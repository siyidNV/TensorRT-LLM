# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Emission-assisted GVR top-k state for the DSA decode path.

Owns the persistent (graph-address-stable) buffers the emission tiers
ride on, the device-side closed-loop seed-row update (pure tensor ops,
CUDA-graph capturable) and the per-step routing decision. Opt-in:
without the flag the DSA decode path is unchanged.

Tier semantics (see gvr_routing):
  * this step's TOP-K consumes what the PREVIOUS step's indexer
    epilogue emitted;
  * this step's INDEXER emits what the routing planned for the NEXT
    step.
"""

import math
from typing import Optional

import torch

from ...cute_dsl_kernels.blackwell.top_k.gvr_routing import (
    LIST_EMIT_MAX_B,
    LIST_EMIT_MIN_N,
    TopkRoute,
    pick_config,
    plan_emission,
)

# Bucketed candidate-list geometry: two tight segments of LIST_SEG_A
# entries plus a LIST_CAP_C-entry loose segment.
LIST_SEG_A = 8192
LIST_CAP_C = 24576
LIST_WIDTH = 2 * LIST_SEG_A + LIST_CAP_C

__all__ = ["GvrEmissionState", "LIST_EMIT_MIN_N", "LIST_PARK_LINE"]

# Closed-loop line placement: fit the slope of log2(count) vs threshold
# from the previous step's (lines, counts) and place the new lines at
# these K-relative target counts (t0 loosest .. t2 tightest).
LINE_TARGETS = (8.0, 5.0, 2.0)
LIST_T0_TARGET = 2.5  # list tier: single collect-line target (xK)
LIST_T0_COUNT_MAX = 6144.0  # keep n0 inside the [K+64, segA] admission band
SLOPE_MIN = 0.05
SLOPE_MAX = 64.0

# No-fit fallback: multiplicative guards around the published k-th value
# (t1 hugs it from below; t0/t2 guard by GUARD_LO/GUARD_HI spans).
FALLBACK_REL = 2.0**0.125 - 1.0
FALLBACK_ABS = 1e-3
GUARD_LO = 2.0
GUARD_HI = 0.5

# List tier only: park the two tight lines above any score so every
# admitted entry lands in the loosest segment. Any finite value above
# the score range works; the kernel's eligibility check only needs the
# three lines increasing and the loosest one finite.
LIST_PARK_LINE = 1.0e30

# Prescore tier: sample = prev top-k plus both neighbors. The sample must
# strictly exceed K (turnover between steps otherwise degrades the bound)
# and must be deduplicated before ranking (duplicates inflate the sample
# k-th above the true k-th, breaking soundness).
PRESCORE_NEIGHBORS = 3

_GATHER_ROWS_KERNEL = None


def _gather_rows_kernel():
    """Lazy-compiled Triton gather: token records from the paged indexer
    K-cache into densely packed mini blocks (same fused record layout, so
    the scoring op reads them bit-identically)."""
    global _GATHER_ROWS_KERNEL
    if _GATHER_ROWS_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _gather_rows(
            pool,
            rows,
            out,
            n_rows,
            S: tl.constexpr,
            TPB: tl.constexpr,
            WPD: tl.constexpr,
            WPB: tl.constexpr,
            ROWS: tl.constexpr,
        ):
            # src record r (= phys_block*TPB + offset): data words at
            # block*WPB + offset*WPD, scale word at block*WPB + TPB*WPD
            # + offset; dst mini block (b*S//TPB + s//TPB) mirrors the
            # same intra-block layout.
            pid = tl.program_id(0)
            r = pid * ROWS + tl.arange(0, ROWS)
            m = r < n_rows
            src = tl.load(rows + r, mask=m, other=0).to(tl.int64)
            sblk = src // TPB
            soff = src % TPB
            b = (r // S).to(tl.int64)
            s = (r % S).to(tl.int64)
            dblk = b * (S // TPB) + s // TPB
            doff = s % TPB
            j = tl.arange(0, WPD)
            v = tl.load(
                pool + (sblk * WPB + soff * WPD)[:, None] + j[None, :],
                mask=m[:, None],
                other=0,
            )
            tl.store(
                out + (dblk * WPB + doff * WPD)[:, None] + j[None, :],
                v,
                mask=m[:, None],
            )
            sv = tl.load(pool + sblk * WPB + TPB * WPD + soff, mask=m, other=0)
            tl.store(out + dblk * WPB + TPB * WPD + doff, sv, mask=m)

        _GATHER_ROWS_KERNEL = _gather_rows
    return _GATHER_ROWS_KERNEL


class GvrEmissionState:
    """Per-attention-backend emission state (persistent buffers)."""

    def __init__(
        self, max_rows: int, top_k: int, device: torch.device, enable_list_tier: bool = True
    ):
        self.max_rows = max_rows
        self.top_k = top_k
        # packed seed row: lines at cols 0..2, counts (emission-filled)
        # at 3..5, adaptive-skip pass count at 6
        self.seed_row = torch.zeros((max_rows, 8), dtype=torch.float32, device=device)
        # contiguous alias of the three lines for the rungs tier (a
        # [rows, 3] column view of the packed row is non-contiguous)
        self.seed_rungs = torch.zeros((max_rows, 3), dtype=torch.float32, device=device)
        self.xstate = torch.zeros((max_rows, 8), dtype=torch.float32, device=device)
        self.cand_vals: Optional[torch.Tensor] = None
        self.cand_idx: Optional[torch.Tensor] = None
        self.cand_ctl: Optional[torch.Tensor] = None
        self.cand_cur: Optional[torch.Tensor] = None
        if enable_list_tier:
            # the routing only ever picks the list tier at
            # batch <= LIST_EMIT_MAX_B, so the wide candidate buffers
            # need that many rows, not max_rows (~0.33 MB/row/layer)
            cand_rows = min(max_rows, LIST_EMIT_MAX_B)
            self.cand_vals = torch.zeros(
                (cand_rows, LIST_WIDTH), dtype=torch.float32, device=device
            )
            self.cand_idx = torch.zeros((cand_rows, LIST_WIDTH), dtype=torch.int32, device=device)
            self.cand_ctl = torch.zeros((cand_rows, 4), dtype=torch.int32, device=device)
            self.cand_cur = torch.zeros((cand_rows, 4), dtype=torch.int32, device=device)
        # previous-step top-k feedback (address-stable; zero-init ->
        # first step's pre_idx points at index 0, a benign candidate)
        self.prev_topk = torch.zeros((max_rows, top_k), dtype=torch.int32, device=device)
        # block_max prefix ([rows, nb_pad*4] fp32 warp-partials),
        # allocated lazily once max_seq_len is known
        self.block_max: Optional[torch.Tensor] = None
        # prescore-tier state (lazy; see ensure_prescore)
        self.mini_fused: Optional[torch.Tensor] = None
        self.mini_bt: Optional[torch.Tensor] = None
        self.mini_meta: Optional[dict] = None
        self._mini_ctx: Optional[torch.Tensor] = None
        self._mini_tpb = 0

    def ensure_prescore(self, tokens_per_block: int, record_bytes: int, num_sms: int) -> None:
        """Lazy prescore buffers: mini K-cache blocks holding the gathered
        sample records plus the (constant) mini block table / context lens
        / DeepGEMM schedules. All shapes depend only on engine-static values, so
        one allocation serves every step; allocation during CUDA graph
        capture fails loudly (same contract as ensure_block_max)."""
        if self.mini_fused is not None and self._mini_tpb == tokens_per_block:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GvrEmissionState.ensure_prescore: (re)allocation requested "
                "during CUDA graph capture; run a warmup step first"
            )
        from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

        cand_rows = min(self.max_rows, LIST_EMIT_MAX_B)
        sample = PRESCORE_NEIGHBORS * self.top_k
        assert sample % tokens_per_block == 0, (
            f"prescore sample {sample} must be a multiple of tokens_per_block {tokens_per_block}"
        )
        mb = sample // tokens_per_block
        device = self.seed_row.device
        self.mini_fused = torch.zeros(
            (cand_rows * mb, tokens_per_block, 1, record_bytes),
            dtype=torch.uint8,
            device=device,
        )
        self.mini_bt = torch.arange(cand_rows * mb, dtype=torch.int32, device=device).view(
            cand_rows, mb
        )
        self._mini_ctx = torch.full((cand_rows,), sample, dtype=torch.int32, device=device)
        # schedule metadata depends on the (python-int) batch size; keep
        # one static buffer per reachable batch so graph capture bakes
        # the right one
        self.mini_meta = {
            b: get_paged_mqa_logits_metadata(
                self._mini_ctx[:b].unsqueeze(-1), tokens_per_block, num_sms
            )
            for b in range(1, cand_rows + 1)
        }
        self._mini_tpb = tokens_per_block

    def prescore_lines(
        self,
        num_rows: int,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        kv_lens: torch.Tensor,
        head_dim: int,
    ) -> None:
        """Sound seed lines from re-scoring the previous step's top-k.

        Gathers prev_topk plus both neighbors (deduplicated) out of the
        paged K-cache into the mini blocks, re-scores them with the SAME
        op on the current query (bit-identical arithmetic), and places
        t0 at the sample k-th: a mathematically sound lower bound of the
        true k-th (the sample is a sub-multiset of the row's scores).
        t1/t2 land at the sample 3K/4- and K/2-th with ascending guards.
        Rows without K distinct finite samples get non-finite lines (the
        kernel's validity guard routes them to the stock path). Pure
        device ops + two kernel launches: graph-capturable.
        """
        tpb = self._mini_tpb
        sample = PRESCORE_NEIGHBORS * self.top_k
        k = self.top_k
        lens = kv_lens[:num_rows].long().clamp_min(1)
        prev = self.prev_topk[:num_rows].long().clamp_min(0)
        last = (lens - 1).unsqueeze(1)
        p0 = torch.minimum(prev, last)
        pos = torch.cat([p0, (p0 - 1).clamp_min(0), torch.minimum(p0 + 1, last)], dim=1)
        # dedup: duplicates would occupy ranking slots and lift the
        # sample k-th above the true k-th (unsound)
        sp, order = pos.sort(dim=1)
        dup = torch.zeros_like(pos, dtype=torch.bool)
        dup.scatter_(1, order[:, 1:], sp[:, 1:] == sp[:, :-1])
        blk = block_table[:num_rows].long().gather(1, pos // tpb)
        rows = (blk * tpb + pos % tpb).to(torch.int32).reshape(-1).contiguous()
        wpd = head_dim // 4
        wpb = self.mini_fused.shape[1] * self.mini_fused.shape[3] // 4
        n_rows = rows.numel()
        _gather_rows_kernel()[((n_rows + 31) // 32,)](
            kv_pool.view(torch.int32).reshape(-1),
            rows,
            self.mini_fused.view(torch.int32).reshape(-1),
            n_rows,
            S=sample,
            TPB=tpb,
            WPD=wpd,
            WPB=wpb,
            ROWS=32,
        )
        mini = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
            q[:num_rows],
            self.mini_fused,
            weights[:num_rows],
            self._mini_ctx[:num_rows],
            self.mini_bt[:num_rows],
            self.mini_meta[num_rows],
            sample,
        )[:, :sample]
        top = torch.topk(mini.masked_fill(dup, float("-inf")), k, dim=1).values
        t0 = top[:, k - 1]
        t1 = torch.maximum(top[:, 3 * k // 4 - 1], t0 + 1e-4)
        t2 = torch.maximum(top[:, k // 2 - 1], t1 + 1e-4)
        valid = torch.isfinite(t0)
        inf = torch.full_like(t0, float("inf"))
        s = self.seed_row[:num_rows]
        s[:, 0] = torch.where(valid, t0, inf)
        s[:, 1] = torch.where(valid, t1, inf)
        s[:, 2] = torch.where(valid, t2, inf)
        s[:, 3:8] = 0.0
        rungs = self.seed_rungs[:num_rows]
        rungs.copy_(s[:, 0:3])
        if self.cand_ctl is not None:
            nc = min(num_rows, self.cand_ctl.shape[0])
            self.cand_ctl[:nc].zero_()
            self.cand_cur[:nc].zero_()

    def ensure_block_max(self, max_seq_len: int) -> torch.Tensor:
        nb4 = ((max_seq_len + 255) // 256 * 256) // 128 * 4
        # exact width: the runner asserts shape == (rows, nrec), so a
        # wider reused buffer would trip it
        if self.block_max is None or self.block_max.shape[1] != nb4:
            # allocating during CUDA graph capture would bake a dangling
            # address into the graph, so fail loudly instead
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "GvrEmissionState.ensure_block_max: (re)allocation requested "
                    "during CUDA graph capture; the block_max buffer must be "
                    "created by a warmup step before capture"
                )
            self.block_max = torch.zeros(
                (self.max_rows, nb4), dtype=torch.float32, device=self.seed_row.device
            )
        return self.block_max

    def plan(
        self, batch: int, n_comp: int, num_sms: int, compress_ratio: int = 4
    ) -> tuple[str, TopkRoute]:
        """Route this step: (tier the epilogue emits, launch knobs the
        top-k consumes it with)."""
        emit_tier = plan_emission(
            batch, n_comp, self.top_k, have_epilogue=True, compress_ratio=compress_ratio
        )
        if emit_tier == "list" and self.cand_vals is None:
            # constructed with enable_list_tier=False: no candidate
            # buffers to emit into, demote to the counts tier
            emit_tier = "counts"
        # emission and consumption happen inside the SAME forward (zero,
        # emit, consume), so the consumer routes on this step's tier
        route = pick_config(emit_tier, batch, n_comp, self.top_k, num_sms)
        return emit_tier, route

    def update_seed_rows(self, num_rows: int, emit_tier: str = "counts") -> None:
        """Device-side closed-loop line update from the last publish.

        Slope-fits log2(count) vs threshold from the previous step's
        (lines, counts) and places the new lines at K-relative target
        counts. Counts come from the packed row (counts/list emission)
        or from the kernel's rung-count publish in xstate cols 4..6
        (rungs tier). Rows without a usable fit get multiplicative
        guards around the published k-th value; rows with invalid
        xstate (col 0 == 0, e.g. cold start) get non-finite lines,
        which the kernel's validity guard routes to the stock path.
        Pure tensor ops (graph-capturable).
        """
        s = self.seed_row[:num_rows]
        x = self.xstate[:num_rows]
        valid = x[:, 0] > 0
        kth = x[:, 1]
        anchor = x[:, 2]
        t_prev0 = s[:, 0]
        t_prev2 = s[:, 2]
        cnts = x[:, 4:7] if emit_tier == "rungs" else s[:, 3:6]
        k = float(self.top_k)
        inf = torch.full_like(kth, float("inf"))
        d_fb = kth.abs() * FALLBACK_REL + FALLBACK_ABS
        if emit_tier == "list":
            # two-point fit (t0_prev, n0) / (kth, K): kth is the exact
            # k-th boundary on list rows
            n0 = cnts[:, 0].clamp_min(1.0)
            dthr = (kth - t_prev0).clamp_min(1e-3)
            slope = ((torch.log2(n0) - math.log2(k)) / dthr).clamp(SLOPE_MIN, SLOPE_MAX)
            tgt0 = min(LIST_T0_TARGET * k, LIST_T0_COUNT_MAX)
            t0 = kth - math.log2(tgt0 / k) / slope
            fit_ok = torch.isfinite(t_prev0) & (n0 > k)
            t0 = torch.where(fit_ok, t0, kth - GUARD_LO * d_fb)
            park = torch.full_like(kth, LIST_PARK_LINE)
            new0 = torch.where(valid, t0, inf)
            new1 = torch.where(valid, park, inf)
            new2 = torch.where(valid, park + park, inf)
        else:
            c0 = cnts[:, 0].clamp_min(1.0)
            c2 = cnts[:, 2].clamp_min(1.0)
            dthr = (t_prev2 - t_prev0).clamp_min(1e-3)
            slope = ((torch.log2(c0) - torch.log2(c2)) / dthr).clamp(SLOPE_MIN, SLOPE_MAX)
            # anchor count estimate: slide the anchor onto the prev line fit
            anch_c = (c2 * torch.exp2(-(anchor - t_prev2) * slope)).clamp(1.0, 1e6)
            t0 = anchor + torch.log2(anch_c / (LINE_TARGETS[0] * k)) / slope
            t1 = anchor + torch.log2(anch_c / (LINE_TARGETS[1] * k)) / slope
            t2 = anchor + torch.log2(anch_c / (LINE_TARGETS[2] * k)) / slope
            # t_prev2 < 1e29 also rejects a parked line left by a tier flip
            fit_ok = torch.isfinite(t_prev0) & (t_prev2 < 1e29) & (c0 > c2)
            t0 = torch.where(fit_ok, t0, kth - GUARD_LO * d_fb)
            t1 = torch.where(fit_ok, t1, kth - 1e-6)
            t2 = torch.where(fit_ok, t2, kth + GUARD_HI * d_fb)
            # strictly ascending (kernel line-validity contract)
            t1 = torch.maximum(t1, t0 + 1e-4)
            t2 = torch.maximum(t2, t1 + 1e-4)
            new0 = torch.where(valid, t0, inf)
            new1 = torch.where(valid, t1, inf)
            new2 = torch.where(valid, t2, inf)
        s[:, 0] = new0
        s[:, 1] = new1
        s[:, 2] = new2
        s[:, 3:8] = 0.0
        rungs = self.seed_rungs[:num_rows]
        rungs[:, 0] = new0
        rungs[:, 1] = new1
        rungs[:, 2] = new2
        if self.cand_ctl is not None:
            nc = min(num_rows, self.cand_ctl.shape[0])
            self.cand_ctl[:nc].zero_()
            self.cand_cur[:nc].zero_()

    def indexer_emit_kwargs(self, emit_tier: str, num_rows: int, single_band: bool = False) -> dict:
        """kwargs for CuteDSLFP4PagedMQALogitsRunner.forward covering the
        planned emission tier (caller merges into its call)."""
        kw: dict = {}
        if emit_tier in ("counts", "list"):
            kw["seed_thr"] = self.seed_row[:num_rows]
        if emit_tier == "list":
            kw.update(
                accept_cap=LIST_SEG_A,
                cand_out=self.cand_vals[:num_rows],
                cand_idx_out=self.cand_idx[:num_rows],
                cand_ctl_out=self.cand_ctl[:num_rows],
                cand_cur_out=self.cand_cur[:num_rows],
            )
            if single_band:
                # every admitted entry goes into the C claim window only
                # (no per-band exact-claim atomics); requires three REAL
                # ascending lines (prescore), not parked ones
                kw["cand_single_band"] = True
        return kw

    def topk_ext_kwargs(
        self,
        route: TopkRoute,
        num_rows: int,
        block_max: Optional[torch.Tensor],
        single_band: bool = False,
    ) -> dict:
        """kwargs for trtllm::cute_dsl_gvr_topk_decode consuming this
        step's emission per the picked route."""
        kw: dict = {
            "xstate": self.xstate[:num_rows],
            "cluster_size": route.cluster_size,
        }
        if route.num_threads is not None:
            kw["num_threads"] = route.num_threads
        if route.tier in ("counts", "list"):
            kw["seed_thr"] = self.seed_row[:num_rows]
        elif route.tier == "rungs":
            # [rows, 3] seed selects the op's ext_rungs variant
            kw["seed_thr"] = self.seed_rungs[:num_rows]
        if route.tier == "list":
            # accept_cap must match the emitter's segment geometry: the
            # buffers are laid out at bases 0 / LIST_SEG_A / 2*LIST_SEG_A,
            # and the consumer derives the C capacity from the tensor
            # width minus 2*accept_cap.
            kw.update(
                cand_vals=self.cand_vals[:num_rows],
                cand_idx=self.cand_idx[:num_rows],
                cand_ctl=self.cand_ctl[:num_rows],
                accept_cap=LIST_SEG_A,
            )
            if single_band:
                kw["cand_single_band"] = True
        if route.attach_block_max and block_max is not None:
            kw["block_max"] = block_max
        return kw
