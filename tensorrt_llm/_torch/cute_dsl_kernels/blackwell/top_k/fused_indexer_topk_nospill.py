# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused DSv4 decode indexer + exact top-K, single kernel, no logits in GMEM.

One launch computes relu-weighted 64-head MQA logits over the paged FP4
indexer K cache and selects each row's exact top-K, without ever writing
per-token scores to global memory: scores live as signed-monotone 16-bit
keys in (distributed) shared memory; selection is a two-level histogram
descent (2048 coarse / 32 fine bins) with warp-aggregated output claims.
Rows may be split across a thread-block cluster; cross-CTA state moves
through DSMEM only.

Contract
- kv_cache pages are the production planar layout: per page 2048 B of
  packed fp4 data (32 tokens x 64 B) followed by 128 B of scale words
  (one int32 = 4 ue8m0 exponents per token).
- weights are SIGNED fp32; scores may be negative (signed key order).
- Output values are fp16-rounded scores: the selected set is exact up to
  fp16 ordering at the K-th boundary (observed < 1e-4 relative on real
  captures; the score inputs are themselves fp4-derived).
- n_comp (block_table width x 32) must be one of the validated grid
  sizes {8192, 16384, 32768}; callers pad the block table (dummy page
  ids beyond context_lens are never dereferenced past the length mask).
- context_lens are per-row and arbitrary within n_comp.
"""

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr

try:
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass.cutlass_dsl import T as _T

    _ASM = True
except Exception:  # pragma: no cover
    _ASM = False

PFD = 4  # default L2 prefetch distance, in 128-token tiles

FP4 = cutlass.Float4E2M1FN
SF = cutlass.Float8E8M0FNU
U8 = cutlass.Uint8
U16 = cutlass.Uint16
U32 = cutlass.Uint32
F16 = cutlass.Float16
F32 = cutlass.Float32
I32 = cutlass.Int32
GMEM = cute.AddressSpace.gmem

TOK = 128  # tokens per MMA tile (M)
HD = 64  # heads (N) -- exact, no zero padding
DIM = 128  # feature dims (K)
DIMB = 64  # bytes per token row
PAGE = 32  # tokens per KV page
PGB = 2176  # bytes per page
NBINS = 2048  # coarse bins = key >> 5 (signed-monotone 16-bit keys)
NFINE = 32  # fine bins  = key & 31
CAP = 2048
CROW = 32  # rows per gmem copy sub-tile

C_THREADS = 384
P_THREADS = 64  # 2 producer warps: same #cp.async instructions in flight
PPP = 16  # producer threads per KV page
M_WARP = 14
NTHREADS = 480  # 15 warps => 136 regs/thread, enough to keep all 64
# per-head weights resident in registers for the scan
NACC = 3
NGRP = 3  # consumer warp groups (C_THREADS / 128)
MAX_NLOC = 2048  # tiles per CTA (dense prefix + filtered survivors; 1M tokens at CS=1)
RBT_MAX = (
    256  # page-table ring: tiles resident in SMEM (two 128-tile halves, refilled by the producers)
)
NDENSE_MAX = 256  # tiles whose keys are stored densely; later tiles are safe-line filtered
CHUNK = 24  # filtered tiles between two consumer rendezvous (multiple of NACC)
LP = 96  # safe-line refresh period in tiles after the dense prefix (multiple of NACC)
TIECAP = (
    2048  # tie-class members the fp32 refinement can rescore (dead 8 KB buffers hold the lists)
)
# GMEM workspace per row (int32 words) for the cluster-free row split: NREP coarse
# histogram replicas, 32 fine bins and 8 counters on private 128 B lines, tie list


def WS_FINE_OF(nrep, tiecap=None):
    return nrep * NBINS


def WS_CTR_OF(nrep, tiecap=None):
    return nrep * NBINS + 32 * NFINE


def WS_TIE_OF(nrep, tiecap=None):
    return nrep * NBINS + 32 * NFINE + 32 * 8


def WS_CAND_OF(nrep, tiecap=None):
    # the tie cap is a per-configuration constant in the kernel (smaller with several queries)
    return nrep * NBINS + 32 * NFINE + 32 * 8 + (TIECAP if tiecap is None else tiecap)


def WS_ROWW_OF(nrep, tiecap=None):
    return WS_CAND_OF(nrep, tiecap) + 2 * CAPL


CAPL = 4096  # K-th-bin candidates a row lists for the last arriver (bigger bins: old path)


C_CAND = 6
C_ARR1, C_WIN, C_ABV, C_TIE, C_ARR2, C_DONE = (
    0,
    1,
    2,
    3,
    4,
    5,
)  # boundary tie members refined in fp32 (larger tie classes keep the fp16 fill)


@cute.jit
def _pfl2(addr):
    """prefetch.global.L2 [addr] -- fire-and-forget L2 warm-up.

    cp.async.cg misses go all the way to DRAM (~900 cycles); the number of
    outstanding line fills an SM can track is what caps its streaming rate.
    Warming L2 a few tiles ahead turns those misses into ~250-cycle L2 hits,
    so the same number of in-flight requests carries several times the
    bandwidth.  Prefetches do not allocate an L1 fill buffer."""
    _llvm.inline_asm(
        _T.i32(),
        [cutlass.Int64(addr).ir_value()],
        "prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _pick(sHist, sPart, sCtl, tidx, warp_idx, target, out):
    """Find bin b where the descending cumulative count first reaches `target`;
    write (b, count_strictly_above) to sCtl[out], sCtl[out+1]."""
    if tidx < NBINS // 8:
        base = NBINS - 8 * (tidx + 1)
        acc = I32(0)
        for j in cutlass.range_constexpr(8):
            acc = acc + sHist[base + j]
        sPart[tidx] = acc
    cute.arch.barrier()
    if warp_idx == 0:
        lane = tidx
        ls = I32(0)
        for j in cutlass.range_constexpr(NBINS // 256):
            ls = ls + sPart[lane * (NBINS // 256) + j]
        incl = ls
        for d in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(incl, 1 << d, mask_and_clamp=0)
            if lane >= (1 << d):
                incl = incl + other
        excl = incl - ls
        if excl < target and incl >= target:
            acc = excl
            gsel = I32(0)
            done = I32(0)
            for j in cutlass.range_constexpr(NBINS // 256):
                nxt = acc + sPart[lane * (NBINS // 256) + j]
                if done == 0:
                    if nxt >= target:
                        gsel = I32(lane * (NBINS // 256) + j)
                        done = I32(1)
                    else:
                        acc = nxt
            base2 = NBINS - 8 * (gsel + 1)
            bsel = I32(0)
            done2 = I32(0)
            for j in cutlass.range_constexpr(8):
                bb = base2 + 7 - j
                nxt = acc + sHist[bb]
                if done2 == 0:
                    if nxt >= target:
                        bsel = bb
                        done2 = I32(1)
                    else:
                        acc = nxt
            sCtl[out] = bsel
            sCtl[out + 1] = acc


@cute.jit
def _pick32(sFine, sCtl, tidx, warp_idx, target, out):
    """Same descending-cumulative search over the 32 fine bins (one warp)."""
    if warp_idx == 0:
        lane = tidx
        c = sFine[NFINE - 1 - lane]
        incl = c
        for d in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(incl, 1 << d, mask_and_clamp=0)
            if lane >= (1 << d):
                incl = incl + other
        excl = incl - c
        if excl < target and incl >= target:
            sCtl[out] = I32(NFINE - 1) - lane
            sCtl[out + 1] = excl


@cute.jit
def _pick_warp(sH, lane, target):
    """One-warp, barrier-free descending-cumulative search over NBINS bins.
    Safe on a histogram that is still being incremented: every bin read is
    <= its final value, so the result is a lower bound of the final bin.
    Lane l owns bins [NBINS-64*(l+1), NBINS-64*l); reads are lane-rotated to
    spread banks. Returns (found, bin, count_strictly_above), warp-uniform."""
    base = NBINS - 64 * (lane + 1)
    ls = I32(0)
    for k in cutlass.range_constexpr(64):
        ls = ls + sH[base + ((k + lane) & 63)]
    incl = ls
    for d in cutlass.range_constexpr(5):
        oth = cute.arch.shuffle_sync_up(incl, 1 << d, mask_and_clamp=0)
        if lane >= (1 << d):
            incl = incl + oth
    excl = incl - ls
    hit = (excl < target) and (incl >= target)
    hm = cute.arch.vote_ballot_sync(hit)
    found = hm != 0
    hl = cute.arch.popc(hm - 1)
    sel_base = cute.arch.shuffle_sync(base, hl)
    acc0 = cute.arch.shuffle_sync(excl, hl)
    v0 = sH[sel_base + 63 - 2 * lane]
    v1 = sH[sel_base + 62 - 2 * lane]
    ps = v0 + v1
    inc2 = ps
    for d in cutlass.range_constexpr(5):
        oth2 = cute.arch.shuffle_sync_up(inc2, 1 << d, mask_and_clamp=0)
        if lane >= (1 << d):
            inc2 = inc2 + oth2
    exc2 = acc0 + inc2 - ps
    hit2 = (exc2 < target) and (acc0 + inc2 >= target)
    m2 = cute.arch.vote_ballot_sync(hit2)
    l2 = cute.arch.popc(m2 - 1)
    bsel_l = sel_base + 63 - 2 * lane
    above_l = exc2
    if exc2 + v0 < target:
        bsel_l = sel_base + 62 - 2 * lane
        above_l = exc2 + v0
    bsel = cute.arch.shuffle_sync(bsel_l, l2)
    above = cute.arch.shuffle_sync(above_l, l2)
    return found, bsel, above


@cute.jit
def _ntl_of(L, crk, CS):
    ntile = (L + TOK - 1) // TOK
    n = I32(0)
    if ntile > crk:
        n = (ntile - crk + CS - 1) // CS
    return n


@cute.jit
def _compact(
    sHist,
    sKey32,
    sSPos,
    sSKey,
    sFine,
    sCtl,
    tidx,
    wl,
    lmaskw,
    warp_idx,
    crk,
    L,
    ndense,
    n,
    KTOP,
    SCAP,
    CS,
    TIECAP,
):
    """Consumer-only (384 threads, named barrier 1) exact shrink of the
    survivor buffer to <= KTOP + TIECAP entries: keep everything above the K-th
    value seen so far, plus the tie quota at that value extended by TIECAP so a
    tie class the fp32 refinement can rescore never loses a member. sHist also
    counts the keys still parked in the window, so the fine descent may find no
    hit; the whole K-th bin is kept then (fs = -1)."""
    if tidx < 32:
        sFine[tidx] = I32(0)
    if tidx == 0:
        sCtl[6] = I32(0)
        sCtl[7] = I32(0)
        sCtl[14] = I32(-1)
        sCtl[15] = I32(0)
    if warp_idx == 0:
        f, bs, ca = _pick_warp(sHist, wl, I32(KTOP))
        if wl == 0:
            sCtl[12] = I32(-1)
            if f:
                sCtl[12] = bs
            sCtl[13] = ca
    cute.arch.barrier(barrier_id=1, number_of_threads=C_THREADS)
    bs = sCtl[12]
    ca = sCtl[13]
    if bs >= 0:
        r1s = I32(KTOP) - ca
        for w in cutlass.range(tidx, ndense * (TOK // 2), C_THREADS, unroll=1):
            x = I32(sKey32[w])
            k0 = x & 0xFFFF
            k1 = (x >> 16) & 0xFFFF
            p0 = 2 * w
            t0 = (crk + (_ntl_of(L, crk, CS) - 1 - (p0 >> 7)) * CS) * TOK + (p0 & (TOK - 1))
            if (t0 < L) and ((k0 >> 5) == bs):
                cute.arch.atomic_add(sFine.iterator + (k0 & (NFINE - 1)), I32(1), scope="cta")
            if (t0 + 1 < L) and ((k1 >> 5) == bs):
                cute.arch.atomic_add(sFine.iterator + (k1 & (NFINE - 1)), I32(1), scope="cta")
        for q in cutlass.range(tidx, n, C_THREADS, unroll=1):
            kq = I32(sSKey[q])
            if (kq >> 5) == bs:
                cute.arch.atomic_add(sFine.iterator + (kq & (NFINE - 1)), I32(1), scope="cta")
        cute.arch.barrier(barrier_id=1, number_of_threads=C_THREADS)
        _pick32(sFine, sCtl, tidx, warp_idx, r1s, 14)
        cute.arch.barrier(barrier_id=1, number_of_threads=C_THREADS)
        fs = sCtl[14]
        r2s = r1s - sCtl[15] + TIECAP
        nr = (n + C_THREADS - 1) // C_THREADS
        for r in cutlass.range(nr, unroll=1):
            q = r * C_THREADS + tidx
            live = q < n
            kq = I32(0)
            pq = I32(0)
            if live:
                kq = I32(sSKey[q])
                pq = sSPos[q]
            b = kq >> 5
            fb = kq & (NFINE - 1)
            kp = live and ((b > bs) or ((b == bs) and (fb > fs)))
            if live and (b == bs) and (fb == fs):
                tq = cute.arch.atomic_add(sCtl.iterator + 6, I32(1), scope="cta")
                if tq < r2s:
                    kp = True
            cute.arch.barrier(barrier_id=1, number_of_threads=C_THREADS)
            m = cute.arch.vote_ballot_sync(kp)
            base = I32(0)
            if wl == 0:
                if m != 0:
                    base = cute.arch.atomic_add(sCtl.iterator + 7, cute.arch.popc(m), scope="cta")
            base = cute.arch.shuffle_sync(base, 0)
            if kp:
                d = base + cute.arch.popc(m & lmaskw)
                sSPos[d] = pq
                sSKey[d] = U16(kq & 0xFFFF)
        cute.arch.barrier(barrier_id=1, number_of_threads=C_THREADS)
        if tidx == 0:
            sCtl[5] = sCtl[7]
            if bs > sCtl[4]:
                sCtl[4] = bs
        if tidx < 32:
            sFine[tidx] = I32(0)
    cute.arch.barrier(barrier_id=1, number_of_threads=C_THREADS)


@cute.jit
def _filter_window(sWin, woff, sSPos, sSKey, sCtl, tidx, wl, lmaskw, crk, L, ntl, tile0, CS, CHUNK):
    """Consumer-only pass over one chunk window (CHUNK*TOK keys, 8 per
    thread): count survivors per warp first, reserve the slots with ONE atomic
    per warp, then place (position, key). Keeps the rendezvous short."""
    NIT = (CHUNK * TOK) // C_THREADS
    lim0 = L - (crk + (ntl - 1) * CS) * TOK
    keeps = [cutlass.Boolean(False) for _ in range(NIT)]
    nks = [I32(0) for _ in range(NIT)]
    masks = [I32(0) for _ in range(NIT)]
    tot = I32(0)
    for r in cutlass.range_constexpr(NIT):
        w = r * C_THREADS + tidx
        k = I32(sWin[woff + w])
        tl = tile0 + w // TOK
        kp = (tl < ntl) and ((tl > 0) or ((w % TOK) < lim0)) and ((k >> 5) >= sCtl[4])
        mk = cute.arch.vote_ballot_sync(kp)
        keeps[r] = kp
        masks[r] = mk
        nks[r] = tot
        tot = tot + cute.arch.popc(mk)
    sb = I32(0)
    if wl == 0:
        if tot > 0:
            sb = cute.arch.atomic_add(sCtl.iterator + 5, tot, scope="cta")
    sb = cute.arch.shuffle_sync(sb, 0)
    for r in cutlass.range_constexpr(NIT):
        if keeps[r]:
            w = r * C_THREADS + tidx
            tl = tile0 + w // TOK
            qs = sb + nks[r] + cute.arch.popc(masks[r] & lmaskw)
            sSPos[qs] = tl * TOK + (w % TOK)
            sSKey[qs] = U16(I32(sWin[woff + w]) & 0xFFFF)


@cute.jit
def _st_dsmem_u32(addr, val):
    """st.shared::cluster.u32 [addr], val (plain remote SMEM store)."""
    _llvm.inline_asm(
        _T.i32(),
        [I32(addr).ir_value(), I32(val).ir_value()],
        "st.shared::cluster.u32 [$1], $2;\n\tmov.u32 $0, 0;",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )


@cute.kernel
def _dsv4_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_k: cute.CopyAtom,
    mKV: cute.Tensor,
    smem_layout_k_tma: cute.ComposedLayout,
    kv_ptr: cute.Pointer,
    q_ptr: cute.Pointer,
    sfq_ptr: cute.Pointer,
    w_ptr: cute.Pointer,
    clen_ptr: cute.Pointer,
    bt_ptr: cute.Pointer,
    oi_ptr: cute.Pointer,
    ov_ptr: cute.Pointer,
    ws_ptr: cute.Pointer,
    LAY: cutlass.Constexpr,
    SWZ: cutlass.Constexpr,
    NCOMP: cutlass.Constexpr,
    B: cutlass.Constexpr,
    KTOP: cutlass.Constexpr,
    MAXB: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    CS: cutlass.Constexpr,
    NLOC: cutlass.Constexpr,
    NDENSE: cutlass.Constexpr,
    SCAP: cutlass.Constexpr,
    REFINE: cutlass.Constexpr,
    RBT: cutlass.Constexpr,
    GM: cutlass.Constexpr,
    NREP: cutlass.Constexpr,
    NEXT_N: cutlass.Constexpr,
    CRS: cutlass.Constexpr,
    TIECAP: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr,
    NACC: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    g2s = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), FP4, num_bits_per_copy=128
    )
    gcpq = cute.make_tiled_copy_tv(
        g2s,
        cute.make_ordered_layout((8, 4), order=(1, 0)),
        cute.make_ordered_layout((4, 32), order=(1, 0)),
    )
    qcp = cute.make_tiled_copy_tv(
        g2s, cute.make_ordered_layout((CROW, 4), order=(1, 0)), cute.make_layout((1, 32))
    )
    g2s_u32 = cute.make_copy_atom(cpasync.CopyG2SOp(), U32, num_bits_per_copy=32)

    smem = utils.SmemAllocator()
    sA_raw = smem.allocate_array(U32, STAGES * TOK * DIMB // 4, byte_alignment=1024)
    sB_raw = smem.allocate_array(U32, HD * DIMB * NEXT_N // 4, byte_alignment=1024)
    sSFA_raw = smem.allocate_array(U32, STAGES * 128, byte_alignment=128)
    sSFB_raw = smem.allocate_array(U32, 128 * ((HD * NEXT_N + 127) // 128), byte_alignment=128)
    # one buffer per query: views over `iterator + offset` lose the unsigned 16-bit type
    sKey_raw_ = [smem.allocate_array(U16, NDENSE * TOK, byte_alignment=128) for _ in range(NEXT_N)]
    sHist_ = [
        smem.allocate_tensor(I32, cute.make_layout(NBINS), byte_alignment=128)
        for _ in range(NEXT_N)
    ]
    sTot = smem.allocate_tensor(I32, cute.make_layout(NBINS), byte_alignment=128)
    sFTot = smem.allocate_tensor(I32, cute.make_layout(NFINE), byte_alignment=128)
    sW = smem.allocate_tensor(F32, cute.make_layout(64 * NEXT_N), byte_alignment=128)
    sPart = smem.allocate_tensor(I32, cute.make_layout(NBINS // 8), byte_alignment=128)
    sCtl_ = [
        smem.allocate_tensor(I32, cute.make_layout(32), byte_alignment=128) for _ in range(NEXT_N)
    ]
    sFine = smem.allocate_tensor(I32, cute.make_layout(NFINE), byte_alignment=128)
    sBT = smem.allocate_tensor(I32, cute.make_layout(RBT * 4), byte_alignment=128)
    sCand = smem.allocate_tensor(I32, cute.make_layout(CAP), byte_alignment=128)
    SPOSN = max(SCAP, 4)
    SKEYN = max(SCAP, 8)
    SWINN = 2 * CHUNK * TOK if SCAP > 0 else 8
    sSPos_ = [
        smem.allocate_tensor(I32, cute.make_layout(SPOSN), byte_alignment=128)
        for _ in range(NEXT_N)
    ]
    sSKey_ = [
        smem.allocate_tensor(U16, cute.make_layout(SKEYN), byte_alignment=128)
        for _ in range(NEXT_N)
    ]
    sWin_ = [
        smem.allocate_tensor(U16, cute.make_layout(SWINN), byte_alignment=128)
        for _ in range(NEXT_N)
    ]
    tmem_hold = smem.allocate_array(I32, 1, byte_alignment=16)
    mbar = smem.allocate_array(cutlass.Int64, 2 * STAGES + 4 * NACC, byte_alignment=16)

    # per-query key views (dense keys as u16 / f16 / u32 pairs) and weight slices
    sKey_ = [cute.make_tensor(sKey_raw_[t], cute.make_layout(NDENSE * TOK)) for t in range(NEXT_N)]
    sVal_ = [
        cute.make_tensor(cute.recast_ptr(sKey_raw_[t], dtype=F16), cute.make_layout(NDENSE * TOK))
        for t in range(NEXT_N)
    ]
    sKey32_ = [
        cute.make_tensor(
            cute.recast_ptr(sKey_raw_[t], dtype=U32), cute.make_layout(NDENSE * TOK // 2)
        )
        for t in range(NEXT_N)
    ]
    sW_ = [
        cute.make_tensor(cute.recast_ptr(sW.iterator + 64 * t, dtype=F32), cute.make_layout(64))
        for t in range(NEXT_N)
    ]
    ACC_COLS = HD * NEXT_N

    NFULL = NACC if NACC == NGRP else NACC * NGRP
    ab_full = mbar
    ab_empty = mbar + STAGES
    acc_full = mbar + 2 * STAGES
    acc_empty = mbar + 2 * STAGES + NFULL

    if tidx == 0:
        for i in cutlass.range_constexpr(STAGES):
            cute.arch.mbarrier_init(ab_full + i, P_THREADS)
            cute.arch.mbarrier_init(ab_empty + i, 1)
        for i in cutlass.range_constexpr(NFULL):
            cute.arch.mbarrier_init(acc_full + i, 1)
        for i in cutlass.range_constexpr(NACC):
            cute.arch.mbarrier_init(acc_empty + i, 128)
        cute.arch.mbarrier_init_fence()

    sA_swz = cute.make_swizzle(SWZ[0], SWZ[1], SWZ[2])
    sA_ptr = cute.recast_ptr(sA_raw, sA_swz, dtype=FP4)
    sB_ptr = cute.recast_ptr(sB_raw, sA_swz, dtype=FP4)
    sA = cute.make_tensor(sA_ptr, cute.make_layout(LAY[0][0], stride=LAY[0][1]))
    sB = cute.make_tensor(sB_ptr, cute.make_layout(LAY[1][0], stride=LAY[1][1]))
    # Partition the staged UMMA A layout into four independent 32x128 TMA
    # destinations (TOK / PAGE = 4 page slots per MMA tile).  The GMEM tensor
    # keeps the physical page pool as its residual mode, so an arbitrary
    # block-table page ID is a legal runtime coordinate and does not require
    # rebuilding the TMA descriptor.
    sK_tma = cute.make_tensor(cute.recast_ptr(sA_raw, dtype=FP4), smem_layout_k_tma)
    sK_tma, gK_tma = cute.nvgpu.cpasync.tma_partition(
        tma_atom_k,
        0,
        cute.make_layout(1),
        smem_tensor=sK_tma,
        gmem_tensor=cute.group_modes(mKV, 0, 2),
    )
    sSFA = cute.make_tensor(
        cute.recast_ptr(sSFA_raw, dtype=SF), cute.make_layout(LAY[2][0], stride=LAY[2][1])
    )
    sSFB = cute.make_tensor(
        cute.recast_ptr(sSFB_raw, dtype=SF), cute.make_layout(LAY[3][0], stride=LAY[3][1])
    )
    acc_layout = cute.make_layout(LAY[4][0], stride=LAY[4][1])
    sfa_tmem_layout = cute.make_layout(LAY[5][0], stride=LAY[5][1])
    sfb_tmem_layout = cute.make_layout(LAY[6][0], stride=LAY[6][1])
    sSFB_u32 = cute.make_tensor(sSFB_raw, cute.make_layout(128 * ((HD * NEXT_N + 127) // 128)))

    if tidx >= 128:
        ii = tidx - 128
        for i in cutlass.range(ii, NBINS, NTHREADS - 128, unroll=1):
            for t in cutlass.range_constexpr(NEXT_N):
                sHist_[t][i] = I32(0)
            sTot[i] = I32(0)
    if tidx < 32:
        for t in cutlass.range_constexpr(NEXT_N):
            sCtl_[t][tidx] = I32(0)
        sFine[tidx] = I32(0)
        sFTot[tidx] = I32(0)

    if warp_idx == 0:
        cute.arch.alloc_tmem(512, tmem_hold)
    if warp_idx == 11:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)

    b = bidx // CS
    crk = bidx % CS
    CSd = I32(CS)
    Lb = I32(0)
    if cutlass.const_expr(GM == 1):
        # balanced split: the grid's CTAs go to rows in proportion to their tile counts (each
        # pl_row keeps at least ceil(tiles / NLOC)); equal rows keep the uniform mapping. Every
        # warp plans redundantly in registers (rows pl_lane, pl_lane+32, ...): no SMEM, no barrier.
        RPL = (B + 31) // 32
        pl_lane = tidx % 32
        pl_gcl = cute.make_tensor(clen_ptr, cute.make_layout(1 << 20))
        pl_ln = [I32(0)] * RPL
        pl_tb = [I32(0)] * RPL
        pl_tot = I32(0)
        pl_mx = I32(0)
        pl_mn = I32(1 << 30)
        for pr in cutlass.range_constexpr(RPL):
            pl_row = pr * 32 + pl_lane
            pl_t = I32(0)
            pl_lv = I32(0)
            if pl_row < B:
                pl_lv = pl_gcl[pl_row]
                pl_t = ((pl_lv >> CRS) + TOK - 1) // TOK
                if pl_t > pl_mx:
                    pl_mx = pl_t
                if pl_t < pl_mn:
                    pl_mn = pl_t
            pl_ln[pr] = pl_lv
            pl_tb[pr] = pl_t
            pl_tot = pl_tot + pl_t
        pl_tot = cute.arch.warp_redux_sync(pl_tot, "add")
        pl_mx = cute.arch.warp_redux_sync(pl_mx, "max")
        pl_mn = cute.arch.warp_redux_sync(pl_mn, "min")
        pl_summin = I32(0)
        pl_mnb = [I32(0)] * RPL
        for pr in cutlass.range_constexpr(RPL):
            pl_m = I32(0)
            if pr * 32 + pl_lane < B:
                pl_m = (pl_tb[pr] + NLOC - 1) // NLOC
                if pl_m < 1:
                    pl_m = I32(1)
            pl_mnb[pr] = pl_m
            pl_summin = pl_summin + pl_m
        pl_summin = cute.arch.warp_redux_sync(pl_summin, "add")
        pl_uni = I32(0)
        if pl_mx == pl_mn:
            pl_uni = I32(1)
        if pl_summin > I32(B * CS):
            pl_uni = I32(1)
        if pl_uni == 0:
            pl_remc = I32(B * CS) - pl_summin
            pl_sumex = I32(0)
            pl_exb = [I32(0)] * RPL
            for pr in cutlass.range_constexpr(RPL):
                pl_e = I32(0)
                if pr * 32 + pl_lane < B:
                    pl_e = (pl_remc * pl_tb[pr]) // pl_tot
                pl_exb[pr] = pl_e
                pl_sumex = pl_sumex + pl_e
            pl_sumex = cute.arch.warp_redux_sync(pl_sumex, "add")
            pl_rleft = pl_remc - pl_sumex
            pl_base = I32(0)
            for pr in cutlass.range_constexpr(RPL):
                pl_row = pr * 32 + pl_lane
                pl_sv = I32(0)
                if pl_row < B:
                    pl_sv = pl_mnb[pr] + pl_exb[pr]
                    if pl_row < pl_rleft:
                        pl_sv = pl_sv + 1
                pl_incl = pl_sv
                for pd in cutlass.range_constexpr(5):
                    pl_oth = cute.arch.shuffle_sync_up(pl_incl, 1 << pd, mask_and_clamp=0)
                    if pl_lane >= (1 << pd):
                        pl_incl = pl_incl + pl_oth
                pl_st = pl_base + pl_incl - pl_sv
                pl_hit = (pl_row < B) and (pl_st <= bidx) and (bidx < pl_st + pl_sv)
                pl_sel = I32(0)
                if pl_hit:
                    pl_sel = pl_lane + 1
                pl_hm = cute.arch.warp_redux_sync(pl_sel, "max")
                if pl_hm != 0:
                    pl_src = pl_hm - 1
                    b = pr * 32 + pl_src
                    CSd = cute.arch.shuffle_sync(pl_sv, pl_src)
                    crk = bidx - cute.arch.shuffle_sync(pl_st, pl_src)
                    Lb = cute.arch.shuffle_sync(pl_ln[pr], pl_src)
                pl_base = pl_base + cute.arch.shuffle_sync(pl_incl, 31)
        else:
            for pr in cutlass.range_constexpr(RPL):
                pl_v = cute.arch.shuffle_sync(pl_ln[pr], b & 31)
                if (b >> 5) == pr:
                    Lb = pl_v
        b = cute.arch.make_warp_uniform(b)
        CSd = cute.arch.make_warp_uniform(CSd)
        crk = cute.arch.make_warp_uniform(crk)
        Lb = cute.arch.make_warp_uniform(Lb)
    # cluster-free row split: this row's GMEM workspace (zero at launch, re-zeroed by the last arriver)
    wrow_i = cutlass.Int64(0)
    gRow = cute.make_tensor(ws_ptr, cute.make_layout(WS_ROWW_OF(NREP)))
    if cutlass.const_expr(GM == 1):
        wrow_i = ws_ptr.toint() + cutlass.Int64(b * NEXT_N) * (WS_ROWW_OF(NREP, TIECAP) * 4)
        gRow = cute.make_tensor(
            ws_ptr + cutlass.Int64(b * NEXT_N) * WS_ROWW_OF(NREP, TIECAP),
            cute.make_layout(WS_ROWW_OF(NREP, TIECAP)),
        )
    Lraw = cute.make_tensor(clen_ptr, cute.make_layout(1 << 20))[b]
    if cutlass.const_expr(GM == 1):
        Lraw = Lb
    # query t of the sequence sees (kv_len - NEXT_N + t + 1) >> CRS positions; the scan covers
    # the last query's window, the shorter ones only mask the final tile
    L_ = [(Lraw - (NEXT_N - 1 - t)) >> CRS for t in range(NEXT_N)]
    L = L_[NEXT_N - 1]
    ntile = (L + TOK - 1) // TOK
    # this CTA owns the interleaved tile subsequence crk, crk+CS, crk+2CS, ...
    ntl = 0
    if ntile > crk:
        ntl = (ntile - crk + CSd - 1) // CSd

    # Only the pages of this CTA's own tiles (crk, crk+CS, ...) are staged,
    # so the SMEM page table stays NLOC*4 entries however long the row is.
    gBT = cute.make_tensor(bt_ptr + cutlass.Int64(b) * MAXB, cute.make_layout(MAXB))
    if tidx >= 128:
        ii = tidx - 128
        for jj in cutlass.range(ii, RBT * 4, NTHREADS - 128, unroll=1):
            gi = (crk + (ntl - 1 - (jj >> 2)) * CSd) * 4 + (jj & 3)
            if ((jj >> 2) < ntl) and (gi < MAXB):
                sBT[jj] = gBT[gi]

    if tidx < 64 * NEXT_N:
        sW[tidx] = cute.make_tensor(w_ptr, cute.make_layout(1 << 20))[b * (64 * NEXT_N) + tidx]
    if tidx < 128:
        # SFB atom: row n -> word (n // 128) * 128 + (n % 32) * 4 + (n // 32) % 4; rows past
        # N repeat the first ones (the MMA never reads them)
        sfq = cute.make_tensor(sfq_ptr, cute.make_layout(1 << 20))
        for w in cutlass.range_constexpr((HD * NEXT_N + 127) // 128):
            nrow = w * 128 + (tidx // 4) + (tidx % 4) * 32
            if nrow >= HD * NEXT_N:
                nrow = nrow % (HD * NEXT_N)
            sSFB_u32[w * 128 + tidx] = sfq[b * (HD * NEXT_N) + nrow]

    if tidx < 128:
        thr_q = qcp.get_slice(tidx)
        sB_c = cute.make_tensor(sB_ptr, cute.make_layout((HD * NEXT_N, DIM), stride=(DIM, 1)))
        gQ = cute.make_tensor(
            cute.make_ptr(
                FP4,
                q_ptr.toint() + cutlass.Int64(b) * (64 * NEXT_N * DIMB),
                GMEM,
                assumed_align=16,
            ),
            cute.make_layout((64 * NEXT_N, DIM), stride=(DIM, 1)),
        )
        for r in cutlass.range_constexpr(2 * NEXT_N):
            src = cute.local_tile(gQ, (CROW, DIM), (r, 0))
            dst = cute.local_tile(sB_c, (CROW, DIM), (r, 0))
            cute.copy(gcpq, thr_q.partition_S(src), thr_q.partition_D(dst))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

    cute.arch.barrier()

    tmem_ptr = cute.arch.retrieve_tmem_ptr(F32, 16, tmem_hold)
    # One scale-factor TMEM buffer PER accumulator.  With a single shared SFA
    # buffer the tcgen05.cp for tile i+1 cannot start until the MMA of tile i
    # has drained it, which serialises the whole MMA pipeline.
    # columns: NACC accumulators of ACC_COLS, then one SFA slot per accumulator, then SFB
    tSFA_ = [
        cute.make_tensor(
            cute.recast_ptr(tmem_ptr + ACC_COLS * NACC + 32 * a, dtype=SF), sfa_tmem_layout
        )
        for a in range(NACC)
    ]
    tSFB = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + ACC_COLS * NACC + 32 * NACC, dtype=SF), sfb_tmem_layout
    )
    tAcc_ = [cute.make_tensor(tmem_ptr + a * ACC_COLS, acc_layout) for a in range(NACC)]

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)

    ca = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), SF)
    tc_sfa_ = [tcgen05.make_s2t_copy(ca, cute.filter_zeros(tSFA_[a])) for a in range(NACC)]
    tc_sfb = tcgen05.make_s2t_copy(ca, cute.filter_zeros(tSFB))
    ta_ = [tc_sfa_[a].get_slice(0) for a in range(NACC)]
    sfa_src_ = [
        tcgen05.get_s2t_smem_desc_tensor(tc_sfa_[a], ta_[a].partition_S(cute.filter_zeros(sSFA)))
        for a in range(NACC)
    ]
    sfa_dst_ = [ta_[a].partition_D(cute.filter_zeros(tSFA_[a])) for a in range(NACC)]
    tb = tc_sfb.get_slice(0)
    sfb_src = tcgen05.get_s2t_smem_desc_tensor(tc_sfb, tb.partition_S(cute.filter_zeros(sSFB)))
    sfb_dst = tb.partition_D(cute.filter_zeros(tSFB))

    # ---------------- producer warps 12..15 ----------------
    if warp_idx >= 12 and warp_idx < M_WARP:
        lt = tidx - C_THREADS
        pgt = lt // PPP
        tkt = lt % PPP
        # A TMA issue must be warp uniform, but this kernel's producer geometry
        # splits each of the two producer warps into two 16-lane page groups.
        # So the DATA plane is issued per warp: warp w owns the two page slots
        # 2*(w-12) and 2*(w-12)+1, i.e. exactly the pages its two half warps
        # would have fetched with cp.async.  The SCALE plane keeps the original
        # 16-lane-per-page decomposition and all 64 per-thread arrivals.
        pgw = 2 * (warp_idx - 12)
        for j in cutlass.range(ntl, unroll=1):
            i = crk + (ntl - 1 - j) * CSd
            s = j % STAGES
            if j >= STAGES:
                cute.arch.mbarrier_wait(ab_empty + s, ((j // STAGES) - 1) & 1)
            if cutlass.const_expr(NLOC > RBT):
                # refill the ring half that fell dead 64 tiles ago with tiles
                # [j+64, j+192); its first reader (the L2 prefetch) is >= 60
                # tiles away. Both producer warps take the same branch.
                if ((j & 127) == 64) and (j + 64 < ntl):
                    for r in cutlass.range_constexpr(8):
                        e = lt * 8 + r
                        jj = j + 64 + (e >> 2)
                        gi = (crk + (ntl - 1 - jj) * CSd) * 4 + (e & 3)
                        if (jj < ntl) and (gi < MAXB):
                            sBT[(jj & (RBT - 1)) * 4 + (e & 3)] = gBT[gi]
                    cute.arch.barrier(barrier_id=2, number_of_threads=P_THREADS)
            if cutlass.const_expr(_ASM):
                jf = j + PFD
                if cutlass.const_expr(NCOMP >= 32768):
                    jf = j + 2
                if jf < ntl:
                    pgf = sBT[(jf & (RBT - 1)) * 4 + pgt]
                    base = kv_ptr.toint() + cutlass.Int64(pgf) * PGB
                    _pfl2(base + tkt * 128)
                    if tkt == 0:
                        _pfl2(base + 2048)
            # scale plane keeps the per-half-warp page; data plane is per warp
            pg0 = cute.arch.make_warp_uniform(sBT[(j & (RBT - 1)) * 4 + pgw])
            pg1 = cute.arch.make_warp_uniform(sBT[(j & (RBT - 1)) * 4 + pgw + 1])
            pg = pg0
            if pgt != pgw:
                pg = pg1
            # One elected expect_tx per producer warp covers both of that warp's
            # page transactions; two warps therefore account for the full
            # 4 * PAGE * DIMB = 8192 bytes of the tile on the same full barrier.
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(ab_full + s, 2 * PAGE * DIMB)
            cute.copy(
                tma_atom_k,
                gK_tma[(None, pg0)],
                sK_tma[(None, pgw, s)],
                tma_bar_ptr=ab_full + s,
            )
            cute.copy(
                tma_atom_k,
                gK_tma[(None, pg1)],
                sK_tma[(None, pgw + 1, s)],
                tma_bar_ptr=ab_full + s,
            )
            for r in cutlass.range_constexpr(PAGE // PPP):
                tk = tkt + PPP * r
                gsf = cute.make_tensor(
                    cute.make_ptr(
                        U32,
                        kv_ptr.toint() + cutlass.Int64(pg) * PGB + 2048 + tk * 4,
                        GMEM,
                        assumed_align=4,
                    ),
                    cute.make_layout(1),
                )
                dsf = cute.make_tensor(sSFA_raw + (s * 128 + tk * 4 + pgt), cute.make_layout(1))
                cute.copy(g2s_u32, gsf, dsf)
            cute.arch.cp_async_mbarrier_arrive_noinc(ab_full + s)

    # ---------------- mma warp ----------------
    elif warp_idx == M_WARP:
        cute.copy(tc_sfb, sfb_src[(None, None, None, None, 0)], sfb_dst)
        for j in cutlass.range(ntl, unroll=1):
            s = j % STAGES
            a = j % NACC
            cute.arch.mbarrier_wait(ab_full + s, (j // STAGES) & 1)
            if j >= NACC:
                cute.arch.mbarrier_wait(acc_empty + a, ((j // NACC) - 1) & 1)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            if cutlass.const_expr(NACC == 3):
                if a == 0:
                    cute.copy(tc_sfa_[0], sfa_src_[0][(None, None, None, None, s)], sfa_dst_[0])
                    cute.gemm(
                        tiled_mma,
                        tAcc_[0],
                        [tCrA[(None, None, None, s)], tSFA_[0]],
                        [tCrB[(None, None, None, 0)], tSFB],
                        tAcc_[0],
                    )
                elif a == 1:
                    cute.copy(tc_sfa_[1], sfa_src_[1][(None, None, None, None, s)], sfa_dst_[1])
                    cute.gemm(
                        tiled_mma,
                        tAcc_[1],
                        [tCrA[(None, None, None, s)], tSFA_[1]],
                        [tCrB[(None, None, None, 0)], tSFB],
                        tAcc_[1],
                    )
                else:
                    cute.copy(tc_sfa_[2], sfa_src_[2][(None, None, None, None, s)], sfa_dst_[2])
                    cute.gemm(
                        tiled_mma,
                        tAcc_[2],
                        [tCrA[(None, None, None, s)], tSFA_[2]],
                        [tCrB[(None, None, None, 0)], tSFB],
                        tAcc_[2],
                    )
            else:
                if a == 0:
                    cute.copy(tc_sfa_[0], sfa_src_[0][(None, None, None, None, s)], sfa_dst_[0])
                    cute.gemm(
                        tiled_mma,
                        tAcc_[0],
                        [tCrA[(None, None, None, s)], tSFA_[0]],
                        [tCrB[(None, None, None, 0)], tSFB],
                        tAcc_[0],
                    )
                else:
                    cute.copy(
                        tc_sfa_[NACC - 1],
                        sfa_src_[NACC - 1][(None, None, None, None, s)],
                        sfa_dst_[NACC - 1],
                    )
                    cute.gemm(
                        tiled_mma,
                        tAcc_[NACC - 1],
                        [tCrA[(None, None, None, s)], tSFA_[NACC - 1]],
                        [tCrB[(None, None, None, 0)], tSFB],
                        tAcc_[NACC - 1],
                    )
            with cute.arch.elect_one():
                tcgen05.commit(acc_full + (j % NFULL))
                tcgen05.commit(ab_empty + s)

    # ---------------- consumer warps 0..11 ----------------
    elif warp_idx < 12:
        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE)
        atom_t2r = cute.make_copy_atom(op, F32)
        lane = tidx % 128
        gwg = tidx // 128
        lay32 = cute.make_layout(((TOK, 32), 1, 1), stride=((65536, 1), 0, 0))
        a0 = cute.make_tensor(tmem_ptr, lay32)
        tt = tcgen05.make_tmem_copy(atom_t2r, a0)
        thr = tt.get_slice(lane)
        # accumulator aa, query t, half h -> columns aa * ACC_COLS + t * HD + 32 * h
        sS_ = [
            [
                [
                    thr.partition_S(
                        cute.make_tensor(tmem_ptr + aa * ACC_COLS + t * HD + 32 * h, lay32)
                    )
                    for h in range(2)
                ]
                for t in range(NEXT_N)
            ]
            for aa in range(NACC)
        ]
        tD = thr.partition_D(cute.make_identity_tensor(a0.shape))
        frg = cute.make_rmem_tensor(tD.shape, F32)
        # All 64 per-head weights are loop invariant: hoist them into registers
        # once (15 warps => 136 regs/thread), which removes 16 LDS.128 per tile
        # per lane from the innermost loop entirely. With several queries the
        # weights are reloaded per query per tile (the address is made opaque so
        # the loads are not hoisted into 64 * NEXT_N live registers).
        wr = cute.make_rmem_tensor(cute.make_layout(64), F32)
        if cutlass.const_expr(NEXT_N == 1):
            cute.autovec_copy(sW, wr)
        z = F32(0.0)
        # relu is scalar (there is no max.f32x2) but the weighted accumulation
        # runs on the Blackwell packed FP32 datapath: 64 fmax + 32 fma.f32x2
        # instead of 64 fmax + 64 fma per token.
        wl = tidx % 32
        lmaskw = (I32(1) << wl) - I32(1)
        ndense = ntl
        nch = I32(0)
        ntl_pad = ntl
        if cutlass.const_expr(NLOC > NDENSE):
            if ntl > NDENSE:
                ndense = I32(NDENSE)
                nch = (ntl - NDENSE + CHUNK - 1) // CHUNK
                ntl_pad = NDENSE + nch * CHUNK
        wslot = I32(0)
        if cutlass.const_expr(NLOC > NDENSE):
            # window slot of this group's first filtered tile: smallest i >= NDENSE with i % NGRP == gwg
            wslot = ((NDENSE + NGRP - 1) // NGRP) * NGRP + gwg - NDENSE
            if wslot >= NGRP:
                wslot = wslot - NGRP
        lim0_ = [L_[t] - (crk + (ntl - 1) * CSd) * TOK for t in range(NEXT_N)]
        for i in cutlass.range(gwg, ntl_pad, NGRP, unroll=1):
            if cutlass.const_expr(NLOC > NDENSE):
                # chunk rendezvous (consumers only): all 3 groups aligned, the
                # survivor count is quiescent; shrink the buffer if the next
                # chunk could overflow it. Phantom tiles i >= ntl only exist to
                # keep the rendezvous count identical across the 12 warps.
                if (i >= NDENSE) and (((i - NDENSE) % CHUNK) < NGRP):
                    rc = (i - NDENSE) // CHUNK
                    for t in cutlass.range_constexpr(NEXT_N):
                        cute.arch.barrier(barrier_id=1, number_of_threads=C_THREADS)
                        nsv = sCtl_[t][5]
                        cute.arch.barrier(barrier_id=1, number_of_threads=C_THREADS)
                        if nsv > SCAP - CHUNK * TOK:
                            _compact(
                                sHist_[t],
                                sKey32_[t],
                                sSPos_[t],
                                sSKey_[t],
                                sFine,
                                sCtl_[t],
                                tidx,
                                wl,
                                lmaskw,
                                warp_idx,
                                crk,
                                L_[t],
                                ndense,
                                nsv,
                                KTOP,
                                SCAP,
                                CSd,
                                TIECAP,
                            )
                        if rc >= 1:
                            _filter_window(
                                sWin_[t],
                                ((rc - 1) % 2) * (CHUNK * TOK),
                                sSPos_[t],
                                sSKey_[t],
                                sCtl_[t],
                                tidx,
                                wl,
                                lmaskw,
                                crk,
                                L_[t],
                                ntl,
                                NDENSE + (rc - 1) * CHUNK,
                                CSd,
                                CHUNK,
                            )
            if i < ntl:
                a = gwg
                if cutlass.const_expr(NACC != NGRP):
                    a = i % NACC
                if cutlass.const_expr(NFULL == NGRP):
                    cute.arch.mbarrier_wait(acc_full + gwg, (i // NGRP) & 1)
                else:
                    cute.arch.mbarrier_wait(acc_full + (i % NFULL), (i // NFULL) & 1)
                for t in cutlass.range_constexpr(NEXT_N):
                    if cutlass.const_expr(NEXT_N > 1):
                        woff = cute.arch.make_warp_uniform(I32(t * 64))
                        cute.autovec_copy(
                            cute.make_tensor(
                                cute.recast_ptr(sW.iterator + woff, dtype=F32), cute.make_layout(64)
                            ),
                            wr,
                        )
                    if cutlass.const_expr(NACC == 3):
                        if a == 0:
                            cute.copy(tt, sS_[0][t][0], frg)
                        elif a == 1:
                            cute.copy(tt, sS_[1][t][0], frg)
                        else:
                            cute.copy(tt, sS_[2][t][0], frg)
                    else:
                        if a == 0:
                            cute.copy(tt, sS_[0][t][0], frg)
                        else:
                            cute.copy(tt, sS_[NACC - 1][t][0], frg)
                    cute.arch.fence_view_async_tmem_load()
                    acc = [[z, z], [z, z], [z, z], [z, z]]
                    for j in cutlass.range_constexpr(16):
                        k = j & 3
                        r0 = cute.arch.fmax(frg[2 * j], z)
                        r1 = cute.arch.fmax(frg[2 * j + 1], z)
                        acc[k][0], acc[k][1] = cute.arch.fma_packed_f32x2(
                            (wr[2 * j], wr[2 * j + 1]), (r0, r1), (acc[k][0], acc[k][1])
                        )
                    if cutlass.const_expr(NACC == 3):
                        if a == 0:
                            cute.copy(tt, sS_[0][t][1], frg)
                        elif a == 1:
                            cute.copy(tt, sS_[1][t][1], frg)
                        else:
                            cute.copy(tt, sS_[2][t][1], frg)
                    else:
                        if a == 0:
                            cute.copy(tt, sS_[0][t][1], frg)
                        else:
                            cute.copy(tt, sS_[NACC - 1][t][1], frg)
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(t == NEXT_N - 1):
                        cute.arch.mbarrier_arrive(acc_empty + a)
                    for j in cutlass.range_constexpr(16):
                        k = j & 3
                        r0 = cute.arch.fmax(frg[2 * j], z)
                        r1 = cute.arch.fmax(frg[2 * j + 1], z)
                        acc[k][0], acc[k][1] = cute.arch.fma_packed_f32x2(
                            (wr[32 + 2 * j], wr[33 + 2 * j]), (r0, r1), (acc[k][0], acc[k][1])
                        )
                    q0 = cute.arch.add_packed_f32x2((acc[0][0], acc[0][1]), (acc[1][0], acc[1][1]))
                    q1 = cute.arch.add_packed_f32x2((acc[2][0], acc[2][1]), (acc[3][0], acc[3][1]))
                    q2 = cute.arch.add_packed_f32x2(q0, q1)
                    sc = q2[0] + q2[1]
                    # signed monotone key: k = u ^ (0x8000 | sign*0x7FFF). The
                    # buffer stores KEYS, so every downstream ordering site (coarse
                    # bin >> 5, fine bin & 31, comparisons) is unchanged; only the
                    # value OUTPUT sites decode back to fp16 bits.
                    hv0 = sc.to(F16)
                    ui = I32(hv0.bitcast(U16)) & 0xFFFF
                    ki = ui ^ (0x8000 + ((ui >> 15) & 1) * 0x7FFF)
                    hv = U16(ki).bitcast(F16)
                    pos = i * TOK + lane
                    if cutlass.const_expr(NLOC > NDENSE):
                        if i < NDENSE:
                            sVal_[t][pos] = hv
                            if (i > 0) or (lane < lim0_[t]):
                                cute.arch.atomic_add(
                                    sHist_[t].iterator + (ki >> 5), I32(1), scope="cta"
                                )
                        else:
                            # filtered tile: histogram stays complete; the key parks in
                            # the chunk window and is filtered at the next rendezvous
                            if (i > 0) or (lane < lim0_[t]):
                                cute.arch.atomic_add(
                                    sHist_[t].iterator + (ki >> 5), I32(1), scope="cta"
                                )
                            sWin_[t][wslot * TOK + lane] = U16(ki & 0xFFFF)
                    else:
                        sVal_[t][pos] = hv
                        if (i > 0) or (lane < lim0_[t]):
                            cute.arch.atomic_add(
                                sHist_[t].iterator + (ki >> 5), I32(1), scope="cta"
                            )
                    if cutlass.const_expr(NLOC > NDENSE):
                        # safe line: coarse bin of the K-th key seen so far (a lower
                        # bound of the final boundary bin). Refreshed by consumer warp 0
                        # after its accumulator is released: TMA and MMA issue never wait.
                        # first line at the last group-0 tile of the dense prefix, then
                        # every LP tiles: a ~1000-cycle detour on any role stalls the
                        # whole pipeline, so it must stay rare.
                        if warp_idx == 0:
                            if (i >= (NDENSE - 1 - ((NDENSE - 1) % NGRP))) and (
                                ((i - (NDENSE - 1 - ((NDENSE - 1) % NGRP))) % LP) == 0
                            ):
                                fl, bl, _ab = _pick_warp(sHist_[t], wl, I32(KTOP))
                                if fl:
                                    if wl == 0:
                                        if bl > sCtl_[t][4]:
                                            sCtl_[t][4] = bl
                if cutlass.const_expr(NLOC > NDENSE):
                    if i >= NDENSE:
                        wslot = wslot + NGRP
                        if wslot >= 2 * CHUNK:
                            wslot = wslot - 2 * CHUNK

        if cutlass.const_expr(NLOC > NDENSE):
            if ntl > NDENSE:
                for t in cutlass.range_constexpr(NEXT_N):
                    cute.arch.barrier(barrier_id=1, number_of_threads=C_THREADS)
                    nsv2 = sCtl_[t][5]
                    cute.arch.barrier(barrier_id=1, number_of_threads=C_THREADS)
                    if nsv2 > SCAP - CHUNK * TOK:
                        _compact(
                            sHist_[t],
                            sKey32_[t],
                            sSPos_[t],
                            sSKey_[t],
                            sFine,
                            sCtl_[t],
                            tidx,
                            wl,
                            lmaskw,
                            warp_idx,
                            crk,
                            L_[t],
                            ndense,
                            nsv2,
                            KTOP,
                            SCAP,
                            CSd,
                            TIECAP,
                        )
                    _filter_window(
                        sWin_[t],
                        ((nch - 1) % 2) * (CHUNK * TOK),
                        sSPos_[t],
                        sSKey_[t],
                        sCtl_[t],
                        tidx,
                        wl,
                        lmaskw,
                        crk,
                        L_[t],
                        ntl,
                        NDENSE + (nch - 1) * CHUNK,
                        CSd,
                        CHUNK,
                    )

    # Each warp signals cluster arrival as soon as its own scan work is done;
    # the matching wait sits after the CTA barrier, so a CTA only pays the
    # cross-CTA skew instead of a full barrier round trip.
    if cutlass.const_expr(CS > 1 and GM == 0):
        cute.arch.cluster_arrive()
    cute.arch.barrier()
    JP = ntl
    for qq in cutlass.range_constexpr(NEXT_N):
        if cutlass.const_expr(qq > 0):
            # next query: the shared fine bins and (cluster path) the merged histogram start clean
            cute.arch.barrier()
            if tidx < 32:
                sFine[tidx] = I32(0)
                sFTot[tidx] = I32(0)
            if cutlass.const_expr(CS > 1 and GM == 0):
                for i in cutlass.range(tidx, NBINS, NTHREADS, unroll=1):
                    sTot[i] = I32(0)
                cute.arch.barrier()
                cute.arch.cluster_arrive()
            else:
                cute.arch.barrier()
        wrow_i = cutlass.Int64(0)
        gRow = cute.make_tensor(ws_ptr, cute.make_layout(WS_ROWW_OF(NREP, TIECAP)))
        if cutlass.const_expr(GM == 1):
            wrow_i = ws_ptr.toint() + cutlass.Int64(b * NEXT_N + qq) * (
                WS_ROWW_OF(NREP, TIECAP) * 4
            )
            gRow = cute.make_tensor(
                ws_ptr + cutlass.Int64(b * NEXT_N + qq) * WS_ROWW_OF(NREP, TIECAP),
                cute.make_layout(WS_ROWW_OF(NREP, TIECAP)),
            )
        gI = cute.make_tensor(
            oi_ptr + cutlass.Int64(b * NEXT_N + qq) * KTOP, cute.make_layout(KTOP)
        )
        gV = cute.make_tensor(
            ov_ptr + cutlass.Int64(b * NEXT_N + qq) * KTOP, cute.make_layout(KTOP)
        )

        # counters live in cluster-rank-0's SMEM; every CTA of the row claims there
        cnt = sCtl_[qq].iterator + 8
        if cutlass.const_expr(CS > 1 and GM == 0):
            cute.arch.cluster_wait()
            cnt = cute.arch.map_dsmem_ptr(sCtl_[qq].iterator + 8, 0)
            pts = [cute.arch.map_dsmem_ptr(sTot.iterator, c) for c in range(CS)]
            if cutlass.const_expr(CS >= 8):
                # Reduce-scatter to bin owners, then owners broadcast their final
                # segment: 2*NBINS remote adds per CTA instead of NBINS*CS. Below
                # CS=8 the all-to-all traffic is cheaper than the extra rendezvous.
                SEG = NBINS // CS
                for i in cutlass.range(tidx, NBINS, NTHREADS, unroll=1):
                    v = sHist_[qq][i]
                    if v != 0:
                        own = i // SEG
                        for c in cutlass.range_constexpr(CS):
                            if own == c:
                                cute.arch.atomic_add(pts[c] + i, v, scope="cluster")
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                for i in cutlass.range(tidx, SEG, NTHREADS, unroll=1):
                    bi = crk * SEG + i
                    v = sTot[bi]
                    if v != 0:
                        for c in cutlass.range_constexpr(CS):
                            if c != crk:
                                cute.arch.atomic_add(pts[c] + bi, v, scope="cluster")
            else:
                for i in cutlass.range(tidx, NBINS, NTHREADS, unroll=1):
                    v = sHist_[qq][i]
                    if v != 0:
                        for c in cutlass.range_constexpr(CS):
                            cute.arch.atomic_add(pts[c] + i, v, scope="cluster")
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        if cutlass.const_expr(GM == 1):
            # coarse merge through GMEM: relaxed reds into this CTA's replica, one spin barrier,
            # then every CTA reads the summed replicas into sTot
            hrep_i = wrow_i + cutlass.Int64((crk % NREP) * NBINS * 4)
            for i in cutlass.range(tidx, NBINS, NTHREADS, unroll=1):
                v = sHist_[qq][i]
                if v != 0:
                    _red_add_gpu(hrep_i + cutlass.Int64(i * 4), v)
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier()
            if tidx == 0:
                _gm_barrier(
                    wrow_i + cutlass.Int64((WS_CTR_OF(NREP, TIECAP) + 32 * C_ARR1) * 4), CSd
                )
            cute.arch.barrier()
            # plain loads after the acquire (L1 is cold for this row): independent per replica,
            # so the compiler can keep NREP * unroll requests in flight
            for i in cutlass.range(tidx, NBINS, NTHREADS, unroll=2):
                acc = I32(0)
                for q in cutlass.range_constexpr(NREP):
                    acc = acc + gRow[q * NBINS + i]
                sTot[i] = acc
            cute.arch.barrier()
        # ---------------- level 1: coarse descent on the online histogram --------
        if cutlass.const_expr(CS > 1):
            _pick(sTot, sPart, sCtl_[qq], tidx, warp_idx, I32(KTOP), 0)
        else:
            _pick(sHist_[qq], sPart, sCtl_[qq], tidx, warp_idx, I32(KTOP), 0)
        cute.arch.barrier()
        b1 = sCtl_[qq][0]
        cabove = sCtl_[qq][1]
        r1 = I32(KTOP) - cabove

        # ---------------- single streaming pass: claim + fine histogram ----------
        # Single-CTA rows stage winners in SMEM (sHist_[qq] / sTot are dead after the
        # coarse descent; local slot == output slot) and write them out with full
        # lines after the pass, instead of per-warp partial-line stores.
        sWK = cute.make_tensor(
            cute.recast_ptr(sTot.iterator, dtype=U16), cute.make_layout(2 * NBINS)
        )
        ndn = ntl
        nsurv = I32(0)
        if cutlass.const_expr(NLOC > NDENSE):
            if ntl > NDENSE:
                ndn = I32(NDENSE)
            nsurv = sCtl_[qq][5]
        nw = ndn * (TOK // 2)
        ntot = nw + nsurv
        niter = (ntot + NTHREADS - 1) // NTHREADS
        lane = cute.arch.lane_idx()
        lmask = (I32(1) << lane) - I32(1)
        for it in cutlass.range(niter, unroll=1):
            w = it * NTHREADS + tidx
            live = w < ntot
            dense = w < nw
            xi = I32(0)
            p0 = 2 * w
            es0 = p0
            if live:
                if dense:
                    xi = I32(sKey32_[qq][w])
            if cutlass.const_expr(NLOC > NDENSE):
                if live and (w >= nw):
                    xi = I32(sSKey_[qq][w - nw])
                    p0 = sSPos_[qq][w - nw]
                    es0 = NDENSE * TOK + (w - nw)
            kv0 = U16(xi & 0xFFFF)
            kv1 = U16((xi >> 16) & 0xFFFF)
            # local slot -> global kv position
            t0 = (crk + (ntl - 1 - (p0 >> 7)) * CSd) * TOK + (p0 & (TOK - 1))
            t1 = t0 + 1
            n0 = I32(kv0) >> 5
            n1 = I32(kv1) >> 5
            hi0 = live and (t0 < L_[qq]) and (n0 > b1)
            hi1 = live and dense and (t1 < L_[qq]) and (n1 > b1)
            # warp-aggregated claim: lanes of a warp take consecutive output slots,
            # which turns 32 scattered 4B stores into one coalesced pair of stores.
            m0 = cute.arch.vote_ballot_sync(hi0)
            m1 = cute.arch.vote_ballot_sync(hi1)
            tot = cute.arch.popc(m0) + cute.arch.popc(m1)
            base = I32(0)
            if lane == 0:
                base = cute.arch.atomic_add(cnt, tot, scope="cluster")
            base = cute.arch.shuffle_sync(base, 0)
            n0lo = cute.arch.popc(m0 & lmask)
            if hi0:
                p = base + n0lo
                if cutlass.const_expr(CS == 1 or GM == 1):
                    sHist_[qq][p] = t0
                    sWK[p] = kv0
                else:
                    gI[p] = t0
                    u0 = I32(kv0) ^ (0x8000 + ((((I32(kv0) >> 15) & 1) ^ 1) * 0x7FFF))
                    gV[p] = U16(u0 & 0xFFFF).bitcast(F16).to(F32)
            if hi1:
                p = base + cute.arch.popc(m0) + cute.arch.popc(m1 & lmask)
                if cutlass.const_expr(CS == 1 or GM == 1):
                    sHist_[qq][p] = t1
                    sWK[p] = kv1
                else:
                    gI[p] = t1
                    u1 = I32(kv1) ^ (0x8000 + ((((I32(kv1) >> 15) & 1) ^ 1) * 0x7FFF))
                    gV[p] = U16(u1 & 0xFFFF).bitcast(F16).to(F32)
            if live and (t0 < L_[qq]) and (n0 == b1):
                q = cute.arch.atomic_add(sCtl_[qq].iterator + 11, I32(1), scope="cta")
                if q < CAP:
                    sCand[q] = es0
                cute.arch.atomic_add(sFine.iterator + (I32(kv0) & (NFINE - 1)), I32(1), scope="cta")
            if live and dense and (t1 < L_[qq]) and (n1 == b1):
                q = cute.arch.atomic_add(sCtl_[qq].iterator + 11, I32(1), scope="cta")
                if q < CAP:
                    sCand[q] = es0 + 1
                cute.arch.atomic_add(sFine.iterator + (I32(kv1) & (NFINE - 1)), I32(1), scope="cta")
        cute.arch.barrier()

        if cutlass.const_expr(CS == 1):
            nwin = sCtl_[qq][8]
            for j in cutlass.range(tidx, nwin, NTHREADS, unroll=1):
                gI[j] = sHist_[qq][j]
                kw = I32(sWK[j])
                uw = kw ^ (0x8000 + ((((kw >> 15) & 1) ^ 1) * 0x7FFF))
                gV[j] = U16(uw & 0xFFFF).bitcast(F16).to(F32)
        use_list = I32(0)
        if cutlass.const_expr(GM == 1):
            if sTot[b1] <= CAPL:
                use_list = I32(1)
            gb = I32(0)
            nwin = I32(0)
            cb = I32(0)
            ncl = I32(0)
            nsl = I32(0)
            nitl = I32(0)
            lanel = I32(0)
            lml = I32(0)
            if use_list == 1:
                # one atomic reserves the winner range, one the candidate range; no rendezvous
                if tidx == 0:
                    sCtl_[qq][16] = _atom_add_gpu(
                        wrow_i + cutlass.Int64((WS_CTR_OF(NREP, TIECAP) + 32 * C_WIN) * 4),
                        sCtl_[qq][8],
                    )
                    sCtl_[qq][18] = _atom_add_gpu(
                        wrow_i + cutlass.Int64((WS_CTR_OF(NREP, TIECAP) + 32 * C_CAND) * 4),
                        sCtl_[qq][11],
                    )
                cute.arch.barrier()
                gb = sCtl_[qq][16]
                nwin = sCtl_[qq][8]
                for j in cutlass.range(tidx, nwin, NTHREADS, unroll=1):
                    gI[gb + j] = sHist_[qq][j]
                    kw = I32(sWK[j])
                    uw = kw ^ (0x8000 + ((((kw >> 15) & 1) ^ 1) * 0x7FFF))
                    gV[gb + j] = U16(uw & 0xFFFF).bitcast(F16).to(F32)
                cb = sCtl_[qq][18]
                ncl = sCtl_[qq][11]
                nsl = ncl
                if ncl > CAP:
                    nsl = ndn * TOK + nsurv
                nitl = (nsl + NTHREADS - 1) // NTHREADS
                lanel = tidx % 32
                lml = (I32(1) << lanel) - I32(1)
                for it in cutlass.range(nitl, unroll=1):
                    ci = it * NTHREADS + tidx
                    live = ci < nsl
                    es = I32(0)
                    if live:
                        es = ci
                        if ncl <= CAP:
                            es = sCand[ci]
                    pl = es
                    kk = I32(0)
                    if live:
                        if cutlass.const_expr(NLOC > NDENSE):
                            if es < NDENSE * TOK:
                                kk = I32(sKey_[qq][es])
                            else:
                                kk = I32(sSKey_[qq][es - NDENSE * TOK])
                                pl = sSPos_[qq][es - NDENSE * TOK]
                        else:
                            kk = I32(sKey_[qq][es])
                    t = (crk + (ntl - 1 - (pl >> 7)) * CSd) * TOK + (pl & (TOK - 1))
                    take = live and (t < L_[qq]) and ((kk >> 5) == b1)
                    if ncl <= CAP:
                        take = live
                    idx = cb + ci
                    ml = I32(0)
                    bl = I32(0)
                    if ncl > CAP:
                        ml = cute.arch.vote_ballot_sync(take)
                        if lanel == 0:
                            if ml != 0:
                                bl = cute.arch.atomic_add(
                                    sCtl_[qq].iterator + 19, cute.arch.popc(ml), scope="cta"
                                )
                        bl = cute.arch.shuffle_sync(bl, 0)
                        idx = cb + bl + cute.arch.popc(ml & lml)
                    if take:
                        gRow[WS_CAND_OF(NREP, TIECAP) + 2 * idx] = t
                        gRow[WS_CAND_OF(NREP, TIECAP) + 2 * idx + 1] = kk
            else:
                # fine merge through GMEM; the winner range is reserved with one atomic per CTA and
                # written out while the second spin barrier completes
                if tidx < NFINE:
                    v = sFine[tidx]
                    if v != 0:
                        _red_add_gpu(
                            wrow_i + cutlass.Int64((WS_FINE_OF(NREP, TIECAP) + 32 * tidx) * 4), v
                        )
                cute.arch.fence_acq_rel_gpu()
                cute.arch.barrier()
                if tidx == 0:
                    _red_add_gpu(
                        wrow_i + cutlass.Int64((WS_CTR_OF(NREP, TIECAP) + 32 * C_ARR2) * 4), I32(1)
                    )
                    sCtl_[qq][16] = _atom_add_gpu(
                        wrow_i + cutlass.Int64((WS_CTR_OF(NREP, TIECAP) + 32 * C_WIN) * 4),
                        sCtl_[qq][8],
                    )
                cute.arch.barrier()
                gb = sCtl_[qq][16]
                nwin = sCtl_[qq][8]
                for j in cutlass.range(tidx, nwin, NTHREADS, unroll=1):
                    gI[gb + j] = sHist_[qq][j]
                    kw = I32(sWK[j])
                    uw = kw ^ (0x8000 + ((((kw >> 15) & 1) ^ 1) * 0x7FFF))
                    gV[gb + j] = U16(uw & 0xFFFF).bitcast(F16).to(F32)
                if tidx == 0:
                    ctr2 = wrow_i + cutlass.Int64((WS_CTR_OF(NREP, TIECAP) + 32 * C_ARR2) * 4)
                    v2 = _ld_acquire_gpu(ctr2)
                    while v2 < CSd:
                        v2 = _ld_acquire_gpu(ctr2)
                    cute.arch.fence_acq_rel_gpu()
                cute.arch.barrier()
                if tidx < NFINE:
                    sFTot[tidx] = _ld_relaxed_gpu(
                        wrow_i + cutlass.Int64((WS_FINE_OF(NREP, TIECAP) + 32 * tidx) * 4)
                    )
                cute.arch.barrier()
        # No rendezvous before the fine push: peer sFTot buffers were zeroed before
        # the scan-end arrive and nobody reads them until the wait below.
        if cutlass.const_expr(CS > 1 and GM == 0):
            ptf = [cute.arch.map_dsmem_ptr(sFTot.iterator, c) for c in range(CS)]
            if tidx < NFINE:
                v = sFine[tidx]
                if v != 0:
                    for c in cutlass.range_constexpr(CS):
                        cute.arch.atomic_add(ptf[c] + tidx, v, scope="cluster")
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        # ---------------- level 2: fine descent (32 bins) ------------------------
        if cutlass.const_expr(CS > 1):
            _pick32(sFTot, sCtl_[qq], tidx, warp_idx, r1, 2)
        else:
            _pick32(sFine, sCtl_[qq], tidx, warp_idx, r1, 2)
        cute.arch.barrier()
        b2 = sCtl_[qq][2]
        c2above = sCtl_[qq][3]
        r2 = r1 - c2above
        base2 = cabove + c2above

        # ---------------- boundary claim + tie fill ------------------------------
        cnt9 = sCtl_[qq].iterator + 9
        cnt10 = sCtl_[qq].iterator + 10
        if cutlass.const_expr(CS > 1 and GM == 0):
            cnt9 = cute.arch.map_dsmem_ptr(sCtl_[qq].iterator + 9, 0)
            cnt10 = cute.arch.map_dsmem_ptr(sCtl_[qq].iterator + 10, 0)
        sTie = sHist_[qq]
        tie_base = I32(0)
        if cutlass.const_expr(CS > 1 and GM == 0):
            tie_base = I32(cute.arch.map_dsmem_ptr(sHist_[qq].iterator, 0).toint())
        cnt9_i = wrow_i + cutlass.Int64((WS_CTR_OF(NREP, TIECAP) + 32 * C_ABV) * 4)
        cnt10_i = wrow_i + cutlass.Int64((WS_CTR_OF(NREP, TIECAP) + 32 * C_TIE) * 4)
        ncand = sCtl_[qq][11]
        nscan = ncand
        if ncand > CAP:
            nscan = ndn * TOK + nsurv
        if cutlass.const_expr(GM == 1):
            if use_list == 0:
                nitb = (nscan + NTHREADS - 1) // NTHREADS
                lanew = tidx % 32
                lmw = (I32(1) << lanew) - I32(1)
                for it in cutlass.range(nitb, unroll=1):
                    ci = it * NTHREADS + tidx
                    live = ci < nscan
                    es = I32(0)
                    if live:
                        es = ci
                        if ncand <= CAP:
                            es = sCand[ci]
                    pl = es
                    kk = I32(0)
                    if live:
                        if cutlass.const_expr(NLOC > NDENSE):
                            if es < NDENSE * TOK:
                                kk = I32(sKey_[qq][es])
                            else:
                                kk = I32(sSKey_[qq][es - NDENSE * TOK])
                                pl = sSPos_[qq][es - NDENSE * TOK]
                        else:
                            kk = I32(sKey_[qq][es])
                    t = (crk + (ntl - 1 - (pl >> 7)) * CSd) * TOK + (pl & (TOK - 1))
                    take = live and (t < L_[qq]) and ((kk >> 5) == b1)
                    if ncand <= CAP:
                        take = live
                    k2 = kk & (NFINE - 1)
                    ab = take and (k2 > b2)
                    tie = take and (k2 == b2)
                    ma = cute.arch.vote_ballot_sync(ab)
                    mt = cute.arch.vote_ballot_sync(tie)
                    basea = I32(0)
                    baset = I32(0)
                    if lanew == 0:
                        if ma != 0:
                            basea = _atom_add_gpu(cnt9_i, cute.arch.popc(ma))
                        if mt != 0:
                            baset = _atom_add_gpu(cnt10_i, cute.arch.popc(mt))
                    basea = cute.arch.shuffle_sync(basea, 0)
                    baset = cute.arch.shuffle_sync(baset, 0)
                    ub = kk ^ (0x8000 + ((((kk >> 15) & 1) ^ 1) * 0x7FFF))
                    if ab:
                        p = basea + cute.arch.popc(ma & lmw)
                        gI[cabove + p] = t
                        gV[cabove + p] = U16(ub & 0xFFFF).bitcast(F16).to(F32)
                    if tie:
                        p = baset + cute.arch.popc(mt & lmw)
                        if p < r2:
                            gI[base2 + p] = t
                            gV[base2 + p] = U16(ub & 0xFFFF).bitcast(F16).to(F32)
                        if cutlass.const_expr(REFINE == 1):
                            if p < TIECAP:
                                gRow[WS_TIE_OF(NREP, TIECAP) + p] = t
        else:
            for ci in cutlass.range(tidx, nscan, NTHREADS, unroll=2):
                es = ci
                if ncand <= CAP:
                    es = sCand[ci]
                pl = es
                kk = I32(0)
                if cutlass.const_expr(NLOC > NDENSE):
                    if es < NDENSE * TOK:
                        kk = I32(sKey_[qq][es])
                    else:
                        kk = I32(sSKey_[qq][es - NDENSE * TOK])
                        pl = sSPos_[qq][es - NDENSE * TOK]
                else:
                    kk = I32(sKey_[qq][es])
                t = (crk + (ntl - 1 - (pl >> 7)) * CSd) * TOK + (pl & (TOK - 1))
                take = t < L_[qq]
                if ncand <= CAP:
                    take = True
                else:
                    take = take and ((kk >> 5) == b1)
                if take:
                    k2 = kk & (NFINE - 1)
                    if k2 > b2:
                        p = I32(0)
                        if cutlass.const_expr(GM == 1):
                            p = _atom_add_gpu(cnt9_i, I32(1))
                        else:
                            p = cute.arch.atomic_add(cnt9, I32(1), scope="cluster")
                        ub = kk ^ (0x8000 + ((((kk >> 15) & 1) ^ 1) * 0x7FFF))
                        gI[cabove + p] = t
                        gV[cabove + p] = U16(ub & 0xFFFF).bitcast(F16).to(F32)
                    elif k2 == b2:
                        p = I32(0)
                        if cutlass.const_expr(GM == 1):
                            p = _atom_add_gpu(cnt10_i, I32(1))
                        else:
                            p = cute.arch.atomic_add(cnt10, I32(1), scope="cluster")
                        if p < r2:
                            gI[base2 + p] = t
                            ub2 = kk ^ (0x8000 + ((((kk >> 15) & 1) ^ 1) * 0x7FFF))
                            gV[base2 + p] = U16(ub2 & 0xFFFF).bitcast(F16).to(F32)
                        if cutlass.const_expr(REFINE == 1):
                            if p < TIECAP:
                                # tie member list for the fp32 refinement (sPart is dead here)
                                if cutlass.const_expr(GM == 1):
                                    gRow[WS_TIE_OF(NREP, TIECAP) + p] = t
                                elif cutlass.const_expr(CS > 1):
                                    _st_dsmem_u32(tie_base + p * 4, t)
                                else:
                                    sTie[p] = t
        last = crk == 0
        if cutlass.const_expr(GM == 1):
            # the last CTA to finish the boundary pass owns the tail: it takes the tie list and
            # the tie count, and re-zeroes the row's workspace once nobody else touches it
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier()
            if tidx == 0:
                sCtl_[qq][17] = _atom_add_gpu(
                    wrow_i + cutlass.Int64((WS_CTR_OF(NREP, TIECAP) + 32 * C_DONE) * 4), I32(1)
                )
                cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier()
            last = sCtl_[qq][17] == CSd - 1
            ntg = I32(0)
            nall = I32(0)
            if cutlass.const_expr(REFINE == 1):
                if (use_list == 1) and (sCtl_[qq][17] != CSd - 1):
                    # warm L2 with this CTA's K-th-bin candidate pages for the tail CTA's fp32 pass
                    nmine = ncl
                    if ncl > CAP:
                        nmine = sCtl_[qq][19]
                    if nmine > 512:
                        nmine = I32(512)
                    for i in cutlass.range(tidx, nmine, NTHREADS, unroll=1):
                        tc = gRow[WS_CAND_OF(NREP, TIECAP) + 2 * (cb + i)]
                        pgc = kv_ptr.toint() + cutlass.Int64(gBT[tc >> 5]) * PGB
                        for pfl in cutlass.range_constexpr(PGB // 128):
                            _pfl2(pgc + pfl * 128)
            if last:
                if use_list == 1:
                    # the whole K-th bin of the row: fine histogram, descent, write-out, tie list
                    nall = _ld_relaxed_gpu(
                        wrow_i + cutlass.Int64((WS_CTR_OF(NREP, TIECAP) + 32 * C_CAND) * 4)
                    )
                    if tidx < 32:
                        sFine[tidx] = I32(0)
                    if tidx == 0:
                        sCtl_[qq][19] = I32(0)
                        sCtl_[qq][20] = I32(0)
                    cute.arch.barrier()
                    for i in cutlass.range(tidx, nall, NTHREADS, unroll=2):
                        kc = _ld_relaxed_gpu(
                            wrow_i + cutlass.Int64((WS_CAND_OF(NREP, TIECAP) + 2 * i + 1) * 4)
                        )
                        cute.arch.atomic_add(
                            sFine.iterator + (kc & (NFINE - 1)), I32(1), scope="cta"
                        )
                    cute.arch.barrier()
                    _pick32(sFine, sCtl_[qq], tidx, warp_idx, r1, 2)
                    cute.arch.barrier()
                    b2 = sCtl_[qq][2]
                    c2above = sCtl_[qq][3]
                    r2 = r1 - c2above
                    base2 = cabove + c2above
                    for i in cutlass.range(tidx, nall, NTHREADS, unroll=2):
                        tc = _ld_relaxed_gpu(
                            wrow_i + cutlass.Int64((WS_CAND_OF(NREP, TIECAP) + 2 * i) * 4)
                        )
                        kc = _ld_relaxed_gpu(
                            wrow_i + cutlass.Int64((WS_CAND_OF(NREP, TIECAP) + 2 * i + 1) * 4)
                        )
                        k2 = kc & (NFINE - 1)
                        uc = kc ^ (0x8000 + ((((kc >> 15) & 1) ^ 1) * 0x7FFF))
                        p = I32(0)
                        if k2 > b2:
                            p = cute.arch.atomic_add(sCtl_[qq].iterator + 19, I32(1), scope="cta")
                            gI[cabove + p] = tc
                            gV[cabove + p] = U16(uc & 0xFFFF).bitcast(F16).to(F32)
                        elif k2 == b2:
                            p = cute.arch.atomic_add(sCtl_[qq].iterator + 20, I32(1), scope="cta")
                            if p < r2:
                                gI[base2 + p] = tc
                                gV[base2 + p] = U16(uc & 0xFFFF).bitcast(F16).to(F32)
                            if cutlass.const_expr(REFINE == 1):
                                if p < TIECAP:
                                    sTie[p] = tc
                    cute.arch.barrier()
                    if tidx == 0:
                        sCtl_[qq][10] = sCtl_[qq][20]
                else:
                    ntg = _ld_relaxed_gpu(cnt10_i)
                    if ntg > TIECAP:
                        ntg = I32(TIECAP)
                    if tidx == 0:
                        sCtl_[qq][10] = _ld_relaxed_gpu(cnt10_i)
                    if cutlass.const_expr(REFINE == 1):
                        for i in cutlass.range(tidx, ntg, NTHREADS, unroll=1):
                            sTie[i] = _ld_relaxed_gpu(
                                wrow_i + cutlass.Int64((WS_TIE_OF(NREP, TIECAP) + i) * 4)
                            )
            cute.arch.barrier()
            if last:
                for i in cutlass.range(tidx, WS_TIE_OF(NREP, TIECAP) + ntg, NTHREADS, unroll=1):
                    gRow[i] = I32(0)
        elif cutlass.const_expr(CS > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        elif cutlass.const_expr(REFINE == 1):
            cute.arch.barrier()

        # ---------------- fp32 boundary refinement ------------------------------
        # The fp16 key is exact above the boundary; only the tie class at the K-th
        # fp16 value can differ from an fp32 top-K. The recorded tie members are
        # rescored through the scan's own TMA -> MMA -> epilogue path: a virtual
        # tile stages the 32-token pages of four members, so the fp32 score is the
        # one that produced the key. ntie <= r2 (all selected anyway) or a class beyond
        # TIECAP members keeps the fp16 fill.
        if cutlass.const_expr(REFINE == 1):
            ntie = sCtl_[qq][10]
            nvt = I32(0)
            if last:
                # the zero class (fp16 key 0x8000) needs no rescoring: relu-weighted sums that round
                # to fp16 zero are exact fp32 zeros, so its members tie in fp32 too
                if (ntie > r2) and (ntie <= TIECAP) and ((b1 * NFINE + b2) != 0x8000):
                    nvt = (ntie + 3) // 4
            sTS = cute.make_tensor(
                cute.recast_ptr(sCand.iterator, dtype=F32), cute.make_layout(CAP)
            )
            sPG = cute.make_tensor(sTot.iterator, cute.make_layout(TIECAP))
            if warp_idx >= 12 and warp_idx < M_WARP:
                lt = tidx - C_THREADS
                pgt = lt // PPP
                tkt = lt % PPP
                pgw = 2 * (warp_idx - 12)
                gBT = cute.make_tensor(bt_ptr + cutlass.Int64(b) * MAXB, cute.make_layout(MAXB))
                for ti in cutlass.range(lt, nvt * 4, P_THREADS, unroll=1):
                    tm = ti
                    if tm >= ntie:
                        tm = ntie - 1
                    sPG[ti] = gBT[sTie[tm] >> 5]
                cute.arch.barrier(barrier_id=2, number_of_threads=P_THREADS)
                for v in cutlass.range(nvt, unroll=1):
                    j = JP + v
                    s = j % STAGES
                    if j >= STAGES:
                        cute.arch.mbarrier_wait(ab_empty + s, ((j // STAGES) - 1) & 1)
                    pg0 = cute.arch.make_warp_uniform(sPG[v * 4 + pgw])
                    pg1 = cute.arch.make_warp_uniform(sPG[v * 4 + pgw + 1])
                    pg = pg0
                    if pgt != pgw:
                        pg = pg1
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_expect_tx(ab_full + s, 2 * PAGE * DIMB)
                    cute.copy(
                        tma_atom_k,
                        gK_tma[(None, pg0)],
                        sK_tma[(None, pgw, s)],
                        tma_bar_ptr=ab_full + s,
                    )
                    cute.copy(
                        tma_atom_k,
                        gK_tma[(None, pg1)],
                        sK_tma[(None, pgw + 1, s)],
                        tma_bar_ptr=ab_full + s,
                    )
                    for r in cutlass.range_constexpr(PAGE // PPP):
                        tk = tkt + PPP * r
                        gsf = cute.make_tensor(
                            cute.make_ptr(
                                U32,
                                kv_ptr.toint() + cutlass.Int64(pg) * PGB + 2048 + tk * 4,
                                GMEM,
                                assumed_align=4,
                            ),
                            cute.make_layout(1),
                        )
                        dsf = cute.make_tensor(
                            sSFA_raw + (s * 128 + tk * 4 + pgt), cute.make_layout(1)
                        )
                        cute.copy(g2s_u32, gsf, dsf)
                    cute.arch.cp_async_mbarrier_arrive_noinc(ab_full + s)
            elif warp_idx == M_WARP:
                for v in cutlass.range(nvt, unroll=1):
                    j = JP + v
                    s = j % STAGES
                    a = j % NACC
                    cute.arch.mbarrier_wait(ab_full + s, (j // STAGES) & 1)
                    if j >= NACC:
                        cute.arch.mbarrier_wait(acc_empty + a, ((j // NACC) - 1) & 1)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if cutlass.const_expr(NACC == 3):
                        if a == 0:
                            cute.copy(
                                tc_sfa_[0], sfa_src_[0][(None, None, None, None, s)], sfa_dst_[0]
                            )
                            cute.gemm(
                                tiled_mma,
                                tAcc_[0],
                                [tCrA[(None, None, None, s)], tSFA_[0]],
                                [tCrB[(None, None, None, 0)], tSFB],
                                tAcc_[0],
                            )
                        elif a == 1:
                            cute.copy(
                                tc_sfa_[1], sfa_src_[1][(None, None, None, None, s)], sfa_dst_[1]
                            )
                            cute.gemm(
                                tiled_mma,
                                tAcc_[1],
                                [tCrA[(None, None, None, s)], tSFA_[1]],
                                [tCrB[(None, None, None, 0)], tSFB],
                                tAcc_[1],
                            )
                        else:
                            cute.copy(
                                tc_sfa_[2], sfa_src_[2][(None, None, None, None, s)], sfa_dst_[2]
                            )
                            cute.gemm(
                                tiled_mma,
                                tAcc_[2],
                                [tCrA[(None, None, None, s)], tSFA_[2]],
                                [tCrB[(None, None, None, 0)], tSFB],
                                tAcc_[2],
                            )
                    else:
                        if a == 0:
                            cute.copy(
                                tc_sfa_[0], sfa_src_[0][(None, None, None, None, s)], sfa_dst_[0]
                            )
                            cute.gemm(
                                tiled_mma,
                                tAcc_[0],
                                [tCrA[(None, None, None, s)], tSFA_[0]],
                                [tCrB[(None, None, None, 0)], tSFB],
                                tAcc_[0],
                            )
                        else:
                            cute.copy(
                                tc_sfa_[NACC - 1],
                                sfa_src_[NACC - 1][(None, None, None, None, s)],
                                sfa_dst_[NACC - 1],
                            )
                            cute.gemm(
                                tiled_mma,
                                tAcc_[NACC - 1],
                                [tCrA[(None, None, None, s)], tSFA_[NACC - 1]],
                                [tCrB[(None, None, None, 0)], tSFB],
                                tAcc_[NACC - 1],
                            )
                    with cute.arch.elect_one():
                        tcgen05.commit(acc_full + (j % NFULL))
                        tcgen05.commit(ab_empty + s)
            elif warp_idx < 12:
                op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE)
                atom_t2r = cute.make_copy_atom(op, F32)
                lane = tidx % 128
                gwg = tidx // 128
                lay32 = cute.make_layout(((TOK, 32), 1, 1), stride=((65536, 1), 0, 0))
                a0 = cute.make_tensor(tmem_ptr, lay32)
                tt = tcgen05.make_tmem_copy(atom_t2r, a0)
                thr = tt.get_slice(lane)
                sSr_ = [
                    [
                        thr.partition_S(
                            cute.make_tensor(tmem_ptr + aa * ACC_COLS + qq * HD + 32 * h, lay32)
                        )
                        for h in range(2)
                    ]
                    for aa in range(NACC)
                ]
                tD = thr.partition_D(cute.make_identity_tensor(a0.shape))
                frg = cute.make_rmem_tensor(tD.shape, F32)
                wr = cute.make_rmem_tensor(cute.make_layout(64), F32)
                cute.autovec_copy(sW_[qq], wr)
                z = F32(0.0)
                j0 = JP + ((gwg - (JP % NGRP) + NGRP) % NGRP)
                for j in cutlass.range(j0, JP + nvt, NGRP, unroll=1):
                    a = gwg
                    if cutlass.const_expr(NACC != NGRP):
                        a = j % NACC
                    if cutlass.const_expr(NFULL == NGRP):
                        cute.arch.mbarrier_wait(acc_full + gwg, (j // NGRP) & 1)
                    else:
                        cute.arch.mbarrier_wait(acc_full + (j % NFULL), (j // NFULL) & 1)
                    if cutlass.const_expr(NACC == 3):
                        if a == 0:
                            cute.copy(tt, sSr_[0][0], frg)
                        elif a == 1:
                            cute.copy(tt, sSr_[1][0], frg)
                        else:
                            cute.copy(tt, sSr_[2][0], frg)
                    else:
                        if a == 0:
                            cute.copy(tt, sSr_[0][0], frg)
                        else:
                            cute.copy(tt, sSr_[NACC - 1][0], frg)
                    cute.arch.fence_view_async_tmem_load()
                    acc = [[z, z], [z, z], [z, z], [z, z]]
                    for jj in cutlass.range_constexpr(16):
                        kq = jj & 3
                        m0 = cute.arch.fmax(frg[2 * jj], z)
                        m1 = cute.arch.fmax(frg[2 * jj + 1], z)
                        acc[kq][0], acc[kq][1] = cute.arch.fma_packed_f32x2(
                            (wr[2 * jj], wr[2 * jj + 1]), (m0, m1), (acc[kq][0], acc[kq][1])
                        )
                    if cutlass.const_expr(NACC == 3):
                        if a == 0:
                            cute.copy(tt, sSr_[0][1], frg)
                        elif a == 1:
                            cute.copy(tt, sSr_[1][1], frg)
                        else:
                            cute.copy(tt, sSr_[2][1], frg)
                    else:
                        if a == 0:
                            cute.copy(tt, sSr_[0][1], frg)
                        else:
                            cute.copy(tt, sSr_[NACC - 1][1], frg)
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.mbarrier_arrive(acc_empty + a)
                    for jj in cutlass.range_constexpr(16):
                        kq = jj & 3
                        m0 = cute.arch.fmax(frg[2 * jj], z)
                        m1 = cute.arch.fmax(frg[2 * jj + 1], z)
                        acc[kq][0], acc[kq][1] = cute.arch.fma_packed_f32x2(
                            (wr[32 + 2 * jj], wr[33 + 2 * jj]), (m0, m1), (acc[kq][0], acc[kq][1])
                        )
                    q0 = cute.arch.add_packed_f32x2((acc[0][0], acc[0][1]), (acc[1][0], acc[1][1]))
                    q1 = cute.arch.add_packed_f32x2((acc[2][0], acc[2][1]), (acc[3][0], acc[3][1]))
                    q2 = cute.arch.add_packed_f32x2(q0, q1)
                    sc = q2[0] + q2[1]
                    ti = (j - JP) * 4 + lane // 32
                    if ti < ntie:
                        if (lane % 32) == (sTie[ti] & 31):
                            sTS[ti] = sc
            cute.arch.barrier()
            if nvt > 0:
                tv = (
                    U16(
                        (
                            I32(b1 * NFINE + b2)
                            ^ (0x8000 + ((((I32(b1 * NFINE + b2) >> 15) & 1) ^ 1) * 0x7FFF))
                        )
                        & 0xFFFF
                    )
                    .bitcast(F16)
                    .to(F32)
                )
                for ti in cutlass.range(tidx, ntie, NTHREADS, unroll=1):
                    si = sTS[ti]
                    rk = I32(0)
                    for tj in cutlass.range(ntie, unroll=1):
                        sj = sTS[tj]
                        if (sj > si) or ((sj == si) and (tj < ti)):
                            rk = rk + 1
                    if rk < r2:
                        gI[base2 + rk] = sTie[ti]
                        gV[base2 + rk] = tv
        if cutlass.const_expr(REFINE == 1):
            JP = JP + nvt

    cute.arch.barrier()
    if warp_idx == 0:
        cute.arch.dealloc_tmem(tmem_ptr, 512)


@cute.jit
def _red_add_gpu(addr, v):
    """red.relaxed.gpu.global.add.s32 (no return value, pipelines)."""
    _llvm.inline_asm(
        _T.i32(),
        [cutlass.Int64(addr).ir_value(), I32(v).ir_value()],
        "red.relaxed.gpu.global.add.s32 [$1], $2;\n\tmov.u32 $0, 0;",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _atom_add_gpu(addr, v):
    """atom.relaxed.gpu.global.add.s32 -> old value."""
    return I32(
        _llvm.inline_asm(
            _T.i32(),
            [cutlass.Int64(addr).ir_value(), I32(v).ir_value()],
            "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _ld_acquire_gpu(addr):
    return I32(
        _llvm.inline_asm(
            _T.i32(),
            [cutlass.Int64(addr).ir_value()],
            "ld.global.acquire.gpu.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _ld_relaxed_gpu(addr):
    return I32(
        _llvm.inline_asm(
            _T.i32(),
            [cutlass.Int64(addr).ir_value()],
            "ld.global.relaxed.gpu.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _gm_barrier(ctr, target):
    """Thread 0 only: release own writes, arrive, spin until all S CTAs arrived, acquire."""
    cute.arch.fence_acq_rel_gpu()
    _red_add_gpu(ctr, I32(1))
    v = _ld_acquire_gpu(ctr)
    while v < target:
        v = _ld_acquire_gpu(ctr)
    cute.arch.fence_acq_rel_gpu()


@cute.jit
def _launch(
    kv_ptr: cute.Pointer,
    q_ptr: cute.Pointer,
    sfq_ptr: cute.Pointer,
    w_ptr: cute.Pointer,
    clen_ptr: cute.Pointer,
    bt_ptr: cute.Pointer,
    oi_ptr: cute.Pointer,
    ov_ptr: cute.Pointer,
    ws_ptr: cute.Pointer,
    stream: cuda.CUstream,
    B: cutlass.Constexpr,
    NCOMP: cutlass.Constexpr,
    KTOP: cutlass.Constexpr,
    MAXB: cutlass.Constexpr,
    NPAGES: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    CS: cutlass.Constexpr,
    NLOC: cutlass.Constexpr,
    NDENSE: cutlass.Constexpr,
    SCAP: cutlass.Constexpr,
    REFINE: cutlass.Constexpr,
    RBT: cutlass.Constexpr,
    GM: cutlass.Constexpr,
    NREP: cutlass.Constexpr,
    NEXT_N: cutlass.Constexpr,
    CRS: cutlass.Constexpr,
    TIECAP: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr,
    NACC: cutlass.Constexpr,
):
    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        FP4,
        FP4,
        OperandMajorMode.K,
        OperandMajorMode.K,
        SF,
        32,
        tcgen05.CtaGroup.ONE,
        (TOK, HD * NEXT_N),
    )
    mt = (TOK, HD * NEXT_N, DIM)
    sA_layout = sm100_utils.make_smem_layout_a(tiled_mma, mt, FP4, STAGES)
    sB_layout = sm100_utils.make_smem_layout_b(tiled_mma, mt, FP4, 1)
    sSFA_layout = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mt, 32, STAGES)
    sSFB_layout = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mt, 32, 1)

    # Present the paged allocation as (token-within-page, dim, physical-page).
    # PGB is a byte stride, hence 2*PGB in the FP4 element domain.
    mKV = cute.make_tensor(
        cute.recast_ptr(kv_ptr, dtype=FP4),
        cute.make_layout(
            (PAGE, DIM, NPAGES),
            stride=(DIM, 1, PGB * 2),
        ),
    )
    sA_layout_mk = cute.composition(sA_layout, cute.make_layout((TOK, DIM, STAGES)))
    smem_layout_k_tma = cute.tiled_divide(sA_layout_mk, (PAGE, DIM))
    smem_layout_k_tma = cute.select(smem_layout_k_tma, [0, 1, 3])
    tma_atom_k, mKV = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mKV,
        smem_layout_k_tma[0],
        (PAGE, DIM),
    )

    acc_layout = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((TOK, HD * NEXT_N))).layout
    sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma, mt, 32, cute.slice_(sSFA_layout, (None, None, None, 0))
    )
    sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma, mt, 32, cute.slice_(sSFB_layout, (None, None, None, 0))
    )

    def _sl(lay):
        return (lay.shape, lay.stride)

    LAY = (
        _sl(sA_layout.outer),
        _sl(sB_layout.outer),
        _sl(sSFA_layout),
        _sl(sSFB_layout),
        _sl(acc_layout),
        _sl(sfa_tmem_layout),
        _sl(sfb_tmem_layout),
    )
    swz = sA_layout.inner
    SWZ = (swz.num_bits, swz.num_base, swz.num_shift)

    smem_bytes = (
        STAGES * (TOK * DIMB + 512)
        + HD * DIMB * NEXT_N
        + 512 * ((HD * NEXT_N + 127) // 128)
        + NDENSE * TOK * 2 * NEXT_N
        + (NEXT_N + 1) * NBINS * 4
        + 64 * 4 * NEXT_N
        + (NBINS // 8) * 4
        + 32 * 4 * NEXT_N
        + 2 * NFINE * 4
        + 64
        + (2 * STAGES + 4 * NACC) * 8
        + RBT * 4 * 4
        + CAP * 4
        + SCAP * 6 * NEXT_N
        + (2 * CHUNK * TOK * 2 if SCAP > 0 else 16) * NEXT_N
        + 2048
    )

    _dsv4_kernel(
        tiled_mma,
        tma_atom_k,
        mKV,
        smem_layout_k_tma,
        kv_ptr,
        q_ptr,
        sfq_ptr,
        w_ptr,
        clen_ptr,
        bt_ptr,
        oi_ptr,
        ov_ptr,
        ws_ptr,
        LAY,
        SWZ,
        NCOMP,
        B,
        KTOP,
        MAXB,
        STAGES,
        CS,
        NLOC,
        NDENSE,
        SCAP,
        REFINE,
        RBT,
        GM,
        NREP,
        NEXT_N,
        CRS,
        TIECAP,
        CHUNK,
        NACC,
    ).launch(
        grid=[B * CS, 1, 1],
        block=[NTHREADS, 1, 1],
        cluster=[1 if GM else CS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


_cache = {}
_ws = {}
_nsm = {}


def _sm_count():
    dev = torch.cuda.current_device()
    n = _nsm.get(dev)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        n = min(n, int(os.environ.get("TRTLLM_FUSED_TOPK_MAX_CTAS", n)))
        _nsm[dev] = n
    return n


_WS_FRESH = os.environ.get("TRTLLM_FUSED_TOPK_WS_FRESH", "0") == "1"


def _workspace(device, nrep, nsm, key):
    # zero at every launch start: rows re-zero themselves at the end of a launch, so one
    # tensor per compiled kernel serves eager calls and graph replays alike. Graphs of the
    # same shape must not replay concurrently on one device; TRTLLM_FUSED_TOPK_WS_FRESH=1
    # gives every captured launch its own tensor (a memset node per replay) instead.
    if _WS_FRESH and torch.cuda.is_current_stream_capturing():
        return torch.zeros(
            key[0] * key[14] * WS_ROWW_OF(nrep, key[16]), dtype=torch.int32, device=device
        )
    k = (device.index, key)
    ws = _ws.get(k)
    if ws is None:
        ws = torch.zeros(nsm * WS_ROWW_OF(nrep, key[16]), dtype=torch.int32, device=device)
        _ws[k] = ws
    return ws


SMEM_CAP = 231424
STAGE_BYTES = TOK * DIMB + 512 + 16
MIN_STAGES = 4


def _fixed(NDENSE, SCAP, RBT, NEXT_N, CHUNK):
    return (
        HD * DIMB * NEXT_N
        + 512 * ((HD * NEXT_N + 127) // 128)
        + NDENSE * TOK * 2 * NEXT_N
        + SCAP * 6 * NEXT_N
        + (2 * CHUNK * TOK * 2 if SCAP > 0 else 16) * NEXT_N
        + (NEXT_N + 1) * NBINS * 4
        + 64 * 4 * NEXT_N
        + (NBINS // 8) * 4
        + 32 * 4 * NEXT_N
        + 2 * NFINE * 4
        + 64
        + RBT * 4 * 4
        + CAP * 4
        + 2048
    )


def _stages(NDENSE, NLOC, SCAP, RBT, NEXT_N=1, CHUNK=CHUNK, NACC=NACC):
    s = (SMEM_CAP - _fixed(NDENSE, SCAP, RBT, NEXT_N, CHUNK)) // STAGE_BYTES
    if s > 16:
        s = 16
    if s < MIN_STAGES:
        s = MIN_STAGES
    return int(s)


_cfg = {}
_ptrs = {}


def _ptr(t, dtype):
    # pointer objects are cached per (address, dtype): decode steps reuse the same buffers
    k = (t.data_ptr(), dtype)
    p = _ptrs.get(k)
    if p is None:
        if len(_ptrs) > 4096:
            _ptrs.clear()
        p = make_ptr(dtype, t.data_ptr(), GMEM, assumed_align=16)
        _ptrs[k] = p
    return p


def _raw_stream(device):
    try:
        return torch._C._cuda_getCurrentRawStream(device.index)
    except Exception:
        return torch.cuda.current_stream(device).cuda_stream


def _config(B, MAXB, NPAGES, KTOP, NEXT_N=1, CRS=0):
    """Launch configuration for a shape; environment knobs are read on first use."""
    NCOMP = MAXB * 32
    # split one row across a cluster of CTAs when the batch cannot fill the GPU;
    # scores stay in each CTA's SMEM and the top-K is reduced through DSMEM.
    assert NCOMP % TOK == 0, f"block_table width {NCOMP} must be a multiple of {TOK}"
    CS = 1
    # cluster residency (measured): at most ~4 clusters of 16 or ~8 of 8 run in one wave,
    # ~32 of 4 and ~64 of 2 do
    while CS < 16 and (B * CS * 2) <= (64 if CS >= 4 else 128):
        CS *= 2
    while CS < 16 and (NCOMP // TOK + CS - 1) // CS > MAX_NLOC:
        CS *= 2
    # cluster-free split: S co-resident CTAs per row merging through GMEM (all
    # B*S CTAs must be resident: one CTA per SM)
    GM = 0
    NREP = 1
    ntile = NCOMP // TOK
    nsm = _sm_count()
    mode = os.environ.get("TRTLLM_FUSED_TOPK_GMEM_SPLIT", "auto")
    S_gm = min(nsm // B, ntile) if 2 * B <= nsm else 1
    # measured (real rows): the split wins on long rows at any small batch, on rows of
    # 512+ tiles once 8 rows share the chip and on rows of 256+ tiles at 16+ rows; 1-4 rows
    # of up to 512 tiles, 8 rows of up to 256 and 16 rows of up to 128 stay on one cluster each
    # (random-token 4k-64k rows at B=16: split 14-19 us vs cluster 10-14).
    # It also wins wherever it places >= 1.2x the cluster path's CTAs with >= 64 tiles
    # each: batches off the power-of-two grid (19-29, 33-49, 65-74) otherwise leave a
    # third to half of the SMs idle (262k tokens B=70: 481 -> 262 us).
    wide = S_gm * 5 >= CS * 6 and ntile // S_gm >= 64 and (ntile + S_gm - 1) // S_gm <= MAX_NLOC
    if S_gm >= 2 and mode != "0":
        if (
            mode == "1"
            or wide
            or (
                S_gm >= 8
                and (ntile >= 1024 or (B >= 8 and ntile >= 512) or (B >= 16 and ntile >= 256))
            )
        ):
            GM = 1
            CS = S_gm
            NREP = 4 if CS >= 32 else (2 if CS >= 8 else 1)
    NLOC = (NCOMP // TOK + CS - 1) // CS
    # the page-table ring is indexed with a power-of-two mask
    RBT = min(1 << max(2, (NLOC - 1).bit_length()), RBT_MAX)
    assert NLOC <= MAX_NLOC, (
        f"row of {NCOMP} tokens exceeds the supported length (16 CTAs x {MAX_NLOC * TOK})"
    )
    # dense prefix of NDENSE tiles; longer CTAs filter the rest by the safe
    # line into a compact survivor buffer that can never overflow:
    # SCAP >= KTOP + TIECAP + CHUNK*TOK (one chunk of appends after a shrink)
    ndense_max = int(os.environ.get("TRTLLM_FUSED_TOPK_NDENSE", NDENSE_MAX))
    NDENSE = NLOC if NLOC <= ndense_max else min(ndense_max, 128)
    # several queries per sequence: NEXT_N selection states share the SMEM, so the dense
    # prefix, the tie cap and the chunk shrink; two accumulators of 192 columns at NEXT_N=3
    tiecap = TIECAP if NEXT_N == 1 else 1024
    chunk = CHUNK if NEXT_N == 1 else 12
    nacc = NACC if NEXT_N <= 2 else 2
    if NEXT_N > 1:
        NDENSE = min(NDENSE, 64 if NEXT_N == 2 else 32)
    assert 1 <= NEXT_N <= 3, f"next_n {NEXT_N}: the fp4 MMA carries at most 3 queries (TMEM)"
    assert HD * NEXT_N * nacc + 32 * nacc + 32 <= 512
    SCAP = 0
    if NLOC > NDENSE:
        room = SMEM_CAP - MIN_STAGES * STAGE_BYTES
        SCAP = ((KTOP + tiecap + chunk * TOK + 511) // 512) * 512
        if NEXT_N > 1 and _fixed(NDENSE, SCAP, RBT, NEXT_N, chunk) > room:
            # NEXT_N survivor buffers: halve the tie cap and the chunk, then shorten the
            # dense prefix until MIN_STAGES K tiles still fit (fp8 rows, NEXT_N=3, K=2048)
            tiecap, chunk = tiecap // 2, chunk // 2
            SCAP = ((KTOP + tiecap + chunk * TOK + 511) // 512) * 512
            NDENSE = min(NDENSE, (room - _fixed(0, SCAP, RBT, NEXT_N, chunk)) // (TOK * 2 * NEXT_N))
        assert NDENSE >= 4 and _fixed(NDENSE, SCAP, RBT, NEXT_N, chunk) <= room, (
            f"top_k {KTOP} x next_n {NEXT_N}: the selection states do not fit in SMEM"
        )
    STAGES = _stages(NDENSE, NLOC, SCAP, RBT, NEXT_N, chunk, nacc)
    # fp32-exact boundary: the K-th fp16 tie class is rescored through the MMA
    # path (on by default); TRTLLM_FUSED_TOPK_FP32_EXACT=0 keeps the fp16 fill
    REFINE = 0 if os.environ.get("TRTLLM_FUSED_TOPK_FP32_EXACT", "1") == "0" else 1
    assert KTOP <= NBINS and TIECAP <= NBINS and TIECAP <= CAP, (
        "tie lists reuse the histogram and candidate buffers"
    )
    key = (
        B,
        NCOMP,
        KTOP,
        MAXB,
        NPAGES,
        STAGES,
        CS,
        NLOC,
        NDENSE,
        SCAP,
        REFINE,
        RBT,
        GM,
        NREP,
        NEXT_N,
        CRS,
        tiecap,
        chunk,
        nacc,
    )
    return key, NREP, nsm


@torch.no_grad()
def run(
    q_fp4,
    sf_q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    top_k_t,
    indices,
    values,
    cr_shift=0,
):
    """q_fp4 [B, next_n, 64, 64] u8, sf_q [B, next_n, 64] i32, weights [B * next_n, 64] f32,
    context_lens [B] (tokens when cr_shift > 0, else positions), block_table [B, max_blocks],
    indices / values [B * next_n, K]; query t of sequence b fills row b * next_n + t and sees
    (context_lens[b] - next_n + t + 1) >> cr_shift positions."""
    B, MAXB = block_table.shape
    NEXT_N = q_fp4.shape[1]
    assert indices.shape[0] == B * NEXT_N and values.shape[0] == B * NEXT_N, (
        f"outputs need {B * NEXT_N} rows (batch {B} x next_n {NEXT_N}), got {indices.shape[0]}"
    )
    NPAGES = kv_cache.numel() // PGB
    KTOP = indices.shape[1]
    ck = (
        B,
        MAXB,
        NPAGES,
        KTOP,
        os.environ.get("TRTLLM_FUSED_TOPK_GMEM_SPLIT", "auto"),
        os.environ.get("TRTLLM_FUSED_TOPK_NDENSE", ""),
        os.environ.get("TRTLLM_FUSED_TOPK_FP32_EXACT", "1"),
        NEXT_N,
        cr_shift,
    )
    cfg = _cfg.get(ck)
    if cfg is None:
        cfg = _config(B, MAXB, NPAGES, KTOP, NEXT_N, cr_shift)
        _cfg[ck] = cfg
    key, NREP, nsm = cfg
    kv_ptr = _ptr(kv_cache, U8)
    q_ptr = _ptr(q_fp4, U8)
    sfq_ptr = _ptr(sf_q, U32)
    w_ptr = _ptr(weights, F32)
    clen_ptr = _ptr(context_lens, I32)
    bt_ptr = _ptr(block_table, I32)
    oi_ptr = _ptr(indices, I32)
    ov_ptr = _ptr(values, F32)
    ws = _workspace(indices.device, NREP, nsm * NEXT_N, key)
    ws_ptr = _ptr(ws, I32)
    stream = cuda.CUstream(_raw_stream(indices.device))
    fn = _cache.get(key)
    if fn is None:
        (
            B,
            NCOMP,
            KTOP,
            MAXB,
            NPAGES,
            STAGES,
            CS,
            NLOC,
            NDENSE,
            SCAP,
            REFINE,
            RBT,
            GM,
            NREP,
            NEXT_N,
            CRS,
            TIECAP,
            CHUNK,
            NACC,
        ) = key
        fn = cute.compile(
            _launch,
            kv_ptr,
            q_ptr,
            sfq_ptr,
            w_ptr,
            clen_ptr,
            bt_ptr,
            oi_ptr,
            ov_ptr,
            ws_ptr,
            stream,
            B,
            NCOMP,
            KTOP,
            MAXB,
            NPAGES,
            STAGES,
            CS,
            NLOC,
            NDENSE,
            SCAP,
            REFINE,
            RBT,
            GM,
            NREP,
            NEXT_N,
            CRS,
            TIECAP,
            CHUNK,
            NACC,
        )
        _cache[key] = fn
    fn(kv_ptr, q_ptr, sfq_ptr, w_ptr, clen_ptr, bt_ptr, oi_ptr, ov_ptr, ws_ptr, stream)
