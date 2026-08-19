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
    PRESCORE_LIST_MAX_B,
    TopkRoute,
    pick_config,
    plan_emission,
)

# Bucketed candidate-list geometry: two tight segments of LIST_SEG_A
# entries plus a LIST_CAP_C-entry loose segment.
LIST_SEG_A = 8192
LIST_CAP_C = 24576
LIST_WIDTH = 2 * LIST_SEG_A + LIST_CAP_C

__all__ = ["GvrEmissionState", "LIST_EMIT_MIN_N", "LIST_PARK_LINE", "PRESCORE_LIST_MAX_B"]

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

# Intra-step scratch shared across layers: emission and consumption
# happen inside one layer's forward, and layers run sequentially on the
# stream, so ONE pool serves every layer (addresses stay stable for
# CUDA-graph capture; growth-only, fail-loud under capture).
_SHARED_SCRATCH: dict = {}


def _shared_scratch(kind: str, shape: tuple, dtype: torch.dtype, device: torch.device):
    key = (kind, device.index)
    need = math.prod(shape)
    t = _SHARED_SCRATCH.get(key)
    if t is None or t.numel() < need:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"gvr_emission shared scratch '{kind}': (re)allocation "
                "requested during CUDA graph capture; run a warmup step first"
            )
        t = torch.zeros(need, dtype=dtype, device=device)
        _SHARED_SCRATCH[key] = t
    return t[:need].view(shape)


_PRESCORE_KERNELS = None


def _prescore_kernels():
    """Lazy-compiled Triton pair for the prescore prepass.

    _mark_prev stamps the current generation byte at each prev position
    (generation tags make the bitmap self-resetting: no clear pass, and
    a stale tag under CUDA-graph replay can only DROP an extra sample -
    the sound direction).

    _score_prep fuses, per sample slot: position derivation (prev and
    both neighbors), duplicate marking (prev entries are distinct, so
    only the +-1 copies collide: with prev itself, or q = p+1 = p'-1 via
    p' = p+2 in prev, where the minus copy wins; boundary/garbage
    entries at i>0 are over-dropped on every copy - sound), physical
    row lookup, and in-register scoring of the fp8 record."""
    global _PRESCORE_KERNELS
    if _PRESCORE_KERNELS is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _mark_prev(prev, lens, mask, gen, npad, n_rows, K: tl.constexpr, ROWS: tl.constexpr):
            pid = tl.program_id(0)
            r = pid * ROWS + tl.arange(0, ROWS)
            m = r < n_rows
            b = (r // K).to(tl.int64)
            last = tl.load(lens + r // K, mask=m, other=1) - 1
            pv = tl.load(prev + r, mask=m, other=0)
            p0 = tl.minimum(tl.maximum(pv, 0), last)
            tl.store(mask + b * npad + p0.to(tl.int64), gen, mask=m)

        @triton.jit
        def _derive_lines(
            scm,
            tk_idx,
            seed,
            rungs,
            ctl,
            cur,
            K: tl.constexpr,
            S: tl.constexpr,
            THREADS: tl.constexpr,
        ):
            # one program per row: min/max over the sample's exact top-K
            # values, slack, line placement, validity, seed/rungs writes
            # and the per-step ctl/cur zeroing - replaces ~20 host-issued
            # elementwise kernels
            b = tl.program_id(0)
            offs = tl.arange(0, THREADS)
            mn = tl.full((THREADS,), float("inf"), tl.float32)
            mx = tl.full((THREADS,), float("-inf"), tl.float32)
            for j in range(0, K, THREADS):
                m = j + offs < K
                idx = tl.load(tk_idx + b * K + j + offs, mask=m, other=0)
                idx = tl.maximum(idx, 0)
                v = tl.load(scm + b * S + idx, mask=m, other=float("inf"))
                mn = tl.minimum(mn, v)
                v2 = tl.where(m, v, float("-inf"))
                mx = tl.maximum(mx, v2)
            t0 = tl.min(mn, axis=0)
            vmax = tl.max(mx, axis=0)
            t0 = t0 - (tl.abs(t0) * (1.0 / 4096.0) + 1e-6)
            spread = tl.maximum(vmax - t0, 4e-4)
            valid = (t0 > -1.0e37) & (t0 < 1.0e37)
            inf = float("inf")
            l0 = tl.where(valid, t0, inf)
            l1 = tl.where(valid, t0 + 0.25 * spread, inf)
            l2 = tl.where(valid, t0 + 0.5 * spread, inf)
            z = tl.program_id(0) * 0
            tl.store(seed + b * 8 + 0, l0)
            tl.store(seed + b * 8 + 1, l1)
            tl.store(seed + b * 8 + 2, l2)
            for jj in range(3, 8):
                tl.store(seed + b * 8 + jj, z.to(tl.float32))
            tl.store(rungs + b * 3 + 0, l0)
            tl.store(rungs + b * 3 + 1, l1)
            tl.store(rungs + b * 3 + 2, l2)
            for jj in range(0, 4):
                tl.store(ctl + b * 4 + jj, z)
                tl.store(cur + b * 4 + jj, z)

        @triton.jit
        def _score_prep(
            prev,
            lens,
            mask,
            bt,
            kdata,
            kscale,
            q,
            w,
            out,
            gen,
            npad,
            maxblk,
            H: tl.constexpr,
            D: tl.constexpr,
            K: tl.constexpr,
            S: tl.constexpr,
            TPB: tl.constexpr,
            BPT: tl.constexpr,
            TS: tl.constexpr,
            TILES: tl.constexpr,
        ):
            # fused gather+score: per (row, TS-sample tile) derive the
            # positions and duplicate flags, read each fp8 record ONCE,
            # score it in-register (fp32 GEMM -> relu -> weighted head
            # sum -> per-token scale, the reference semantics), and store
            # the fp32 score (-inf for duplicates). Skips the mini-block
            # materialization entirely; arithmetic-order divergence from
            # the CUTLASS scorer is orders below the t0 slack.
            pid = tl.program_id(0)
            tiles_per_row: tl.constexpr = S // (TS * TILES)
            b = pid // tiles_per_row
            t0i = (pid % tiles_per_row) * TILES
            hj = tl.arange(0, H)
            dj = tl.arange(0, D)
            qb = tl.load(q + b * H * D + hj[:, None] * D + dj[None, :])
            qbt = tl.trans(qb)
            wb = tl.load(w + b * H + hj)
            last = tl.load(lens + b) - 1
            b64 = b.to(tl.int64) * npad
            for ti in range(TILES):
                sm = (t0i + ti) * TS + tl.arange(0, TS)
                c = sm // K
                i = sm % K
                pv = tl.load(prev + b * K + i)
                p0 = tl.minimum(tl.maximum(pv, 0), last)
                pos = tl.where(
                    c == 0, p0, tl.where(c == 1, tl.maximum(p0 - 1, 0), tl.minimum(p0 + 1, last))
                )
                g = tl.load(mask + b64 + pos.to(tl.int64))
                in_prev = g == gen
                p2 = tl.minimum(pos + 1, last)
                g2 = tl.load(mask + b64 + p2.to(tl.int64))
                dg = ((pv <= 0) | (pv >= last)) & (i > 0)
                dup = dg | tl.where(
                    c == 0,
                    False,
                    tl.where(c == 1, in_prev, in_prev | ((g2 == gen) & (p2 != pos))),
                )
                blk = tl.load(bt + b * maxblk + pos // TPB).to(tl.int64)
                soff = (pos % TPB).to(tl.int64)
                kv = tl.load(kdata + (blk * (TPB * BPT) + soff * D)[:, None] + dj[None, :])
                acc = tl.dot(kv, qbt, out_dtype=tl.float32)
                acc = tl.maximum(acc, 0.0)
                sc = tl.sum(acc * wb[None, :], axis=1)
                scale_i = tl.load(kscale + blk * (TPB * BPT // 4) + TPB * (D // 4) + soff)
                kscl = scale_i.to(tl.float32, bitcast=True)
                sc = sc * kscl
                sc = tl.where(dup, float("-inf"), sc)
                tl.store(out + b * S + sm, sc)

        _PRESCORE_KERNELS = (_mark_prev, _derive_lines, _score_prep)
    return _PRESCORE_KERNELS


class GvrEmissionState:
    """Per-attention-backend emission state (persistent buffers)."""

    def __init__(
        self,
        max_rows: int,
        top_k: int,
        device: torch.device,
        enable_list_tier: bool = True,
        cand_rows_cap: Optional[int] = None,
    ):
        self.max_rows = max_rows
        self.top_k = top_k
        self.cand_rows_cap = LIST_EMIT_MAX_B if cand_rows_cap is None else cand_rows_cap
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
            cand_rows = min(max_rows, self.cand_rows_cap)
            self.cand_vals = _shared_scratch(
                "cand_vals", (cand_rows, LIST_WIDTH), torch.float32, device
            )
            self.cand_idx = _shared_scratch(
                "cand_idx", (cand_rows, LIST_WIDTH), torch.int32, device
            )
            self.cand_ctl = _shared_scratch("cand_ctl", (cand_rows, 4), torch.int32, device)
            self.cand_cur = _shared_scratch("cand_cur", (cand_rows, 4), torch.int32, device)
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
        """Lazy prescore state: the (constant) mini context lens for the
        ranking op plus the tokens-per-block geometry. Scores are computed
        in-place by the fused Triton scorer (no mini-block staging), so
        the only allocation is tiny; capture-guarded like the scratch."""
        del num_sms  # geometry no longer needs a schedule
        cand_rows = min(self.max_rows, self.cand_rows_cap)
        sample = PRESCORE_NEIGHBORS * self.top_k
        assert sample % tokens_per_block == 0, (
            f"prescore sample {sample} must be a multiple of tokens_per_block {tokens_per_block}"
        )
        assert record_bytes == 4 * (record_bytes // 4)
        if self._mini_ctx is not None and self._mini_tpb == tokens_per_block:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GvrEmissionState.ensure_prescore: (re)allocation requested "
                "during CUDA graph capture; run a warmup step first"
            )
        device = self.seed_row.device
        self._mini_ctx = torch.full((cand_rows,), sample, dtype=torch.int32, device=device)
        self._mini_tpb = tokens_per_block
        self._mini_rec = record_bytes

    def prescore_lines(
        self,
        num_rows: int,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        kv_lens: torch.Tensor,
        head_dim: int,
        n_pad: int,
    ) -> None:
        """Sound seed lines from re-scoring the previous step's top-k.

        The fused Triton scorer reads prev_topk plus both neighbors
        (deduplicated) straight out of the paged K-cache and re-scores
        them on the current query with the reference arithmetic; t0
        lands at the sample k-th minus a relative slack that dominates
        any accumulation-order divergence from the CUTLASS scorer by two
        orders of magnitude - a sound lower bound of the true k-th (the
        sample is a sub-multiset of the row's scores). Rows without K
        distinct finite samples get non-finite lines (the kernel's
        validity guard routes them to the stock path). Four launches,
        all on static state: graph-capturable.
        """
        tpb = self._mini_tpb
        sample = PRESCORE_NEIGHBORS * self.top_k
        k = self.top_k
        gen = (getattr(self, "_mask_gen", 0) % 255) + 1
        self._mask_gen = gen
        mask = _shared_scratch("dedup_mask", (num_rows, n_pad), torch.uint8, q.device)
        prev32 = self.prev_topk[:num_rows]
        lens32 = kv_lens[:num_rows].contiguous()
        bt32 = block_table[:num_rows]
        mark, derive, score = _prescore_kernels()
        n_k = num_rows * k
        mark[((n_k + 255) // 256,)](prev32, lens32, mask.view(-1), gen, n_pad, n_k, K=k, ROWS=256)
        heads = q.shape[2]
        scm = _shared_scratch("mini_scores", (num_rows, sample), torch.float32, q.device)
        score[(num_rows * (sample // 512),)](
            prev32,
            lens32,
            mask.view(-1),
            bt32,
            kv_pool.view(torch.float8_e4m3fn).reshape(-1),
            kv_pool.view(torch.int32).reshape(-1),
            q[:num_rows].view(torch.float8_e4m3fn).reshape(-1),
            weights[:num_rows].reshape(-1),
            scm.view(-1),
            gen,
            n_pad,
            bt32.shape[1],
            H=heads,
            D=head_dim,
            K=k,
            S=sample,
            TPB=tpb,
            BPT=self._mini_rec,
            TS=64,
            TILES=8,
        )
        # rank via the GVR top-k op itself (torch sort/topk fall to a
        # per-row radix path, ~15us x rows). t0 = min over the sample's
        # exact top-K = the sample k-th; t1/t2 are heuristic refinement
        # lines by sample spread (only their EXACT emitted counts matter
        # to the consumer, soundness rides on t0 alone). t0 slack: still
        # a lower bound, and it keeps the claimed count above the K+64
        # admission floor on zero-turnover steps.
        tk_idx = _shared_scratch("mini_topk_idx", (num_rows, k), torch.int32, q.device)
        pre0 = _shared_scratch("mini_pre_idx", (num_rows, k), torch.int32, q.device)
        torch.ops.trtllm.cute_dsl_gvr_topk_decode(
            scm, pre0, self._mini_ctx[:num_rows], tk_idx, k, 1, 1, max_seq_len=sample
        )
        derive[(num_rows,)](
            scm,
            tk_idx,
            self.seed_row,
            self.seed_rungs,
            self.cand_ctl,
            self.cand_cur,
            K=k,
            S=sample,
            THREADS=256,
        )

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
        self,
        batch: int,
        n_comp: int,
        num_sms: int,
        compress_ratio: int = 4,
        list_max_b: Optional[int] = None,
    ) -> tuple[str, TopkRoute]:
        """Route this step: (tier the epilogue emits, launch knobs the
        top-k consumes it with)."""
        emit_tier = plan_emission(
            batch,
            n_comp,
            self.top_k,
            have_epilogue=True,
            compress_ratio=compress_ratio,
            list_max_b=list_max_b,
        )
        if emit_tier == "list" and (self.cand_vals is None or batch > self.cand_vals.shape[0]):
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
