# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-only numerical path for the Qwen-facing official ASD rule.

Reference: Kissmetothemoon/ASD, src/asd/budget.py at 9426ce7b3b8cafb2.
Torch is used for tensor allocation/metadata only, never ASD arithmetic.
The target sampler's existing logit processors run before these kernels.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.spec_decode.logit_score import score_logits as score_logits

VOCAB_TILE = 1024


@triton.jit(do_not_specialize=["N"])
def _gather_inputs(input_ids, positions, indices, draft, selected_pos, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = tl.load(indices + offsets, offsets < N, other=0)
    token = tl.load(input_ids + idx, offsets < N, other=0)
    pos = tl.load(positions + idx, offsets < N, other=0)
    tl.store(draft + offsets, token, offsets < N)
    tl.store(selected_pos + offsets, pos, offsets < N)


def gather_inputs(input_ids, positions, indices):
    n = indices.numel()
    draft = torch.empty((n,), dtype=input_ids.dtype, device=input_ids.device)
    selected_pos = torch.empty((n,), dtype=positions.dtype, device=positions.device)
    if n:
        _gather_inputs[(triton.cdiv(n, 256),)](input_ids, positions, indices, draft, selected_pos, n, 256)
    return draft, selected_pos


@triton.jit
def _copy_ledger(old, new, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(old + offsets, offsets < N, other=0.0)
    tl.store(new + offsets, values, offsets < N)


def grow_ledger(old, capacity):
    new = torch.empty((capacity,), device=old.device, dtype=torch.float32)
    if old.numel():
        _copy_ledger[(triton.cdiv(old.numel(), 256),)](old, new, old.numel(), 256)
    return new


@triton.jit
def _select_prefix(
    draft,
    targets,
    regrets,
    cu,
    request_meta,
    state_indices,
    seq_lens,
    prefill_lens,
    ledger,
    output,
    sampled_count,
    STRIDE: tl.constexpr,
    BUDGET: tl.constexpr,
    GATE: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    DIRECT_COMMIT: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(cu + row)
    end = tl.load(cu + row + 1)
    k = end - start - 1
    ledger_slot = tl.load(request_meta + row * 3)
    reset = tl.load(request_meta + row * 3 + 1) != 0
    old = tl.load(ledger + ledger_slot, ~reset, other=0.0)
    # Use the official remaining-budget comparison, not old + cost <= B.
    remaining = tl.maximum(BUDGET - old, 0.0)
    cumulative = 0.0
    charged = old
    mismatches = 0
    accepted = 0
    active = True
    cols = tl.arange(0, BLOCK)
    tl.store(output + row * STRIDE + cols, -1, cols < STRIDE)
    for i in range(k):
        if active:
            candidate = tl.load(draft + start + i + 1)
            target = tl.load(targets + start + i)
            mismatch = candidate != target
            regret = tl.load(regrets + start + i)
            admissible = True
            if mismatch:
                if (BUDGET == 0.0) | (GATE == 0.0) | (CAP == 0):
                    admissible = False
                else:
                    cumulative += regret
                    mismatches += 1
                    value = (k - i).to(tl.float32)
                    # Ascend 3.2.1 does not lower tl.div_rn (compiler abort).
                    # Use its supported FP32 division; see the documented
                    # numerical-boundary difference from Python float64.
                    ratio = regret / value
                    admissible = (regret >= 0.0) & (regret < float("inf"))
                    admissible = admissible & (ratio <= GATE) & (cumulative <= remaining + 1.0e-9)
                    admissible = admissible & (mismatches <= CAP)
            if admissible:
                tl.store(output + row * STRIDE + i, candidate)
                accepted = i + 1
                if mismatch:
                    # Official state.charge is applied sequentially to accepted mismatches.
                    charged += regret
            else:
                active = False
    recovery = tl.load(targets + start + accepted)
    tl.store(output + row * STRIDE + accepted, recovery)
    tl.store(sampled_count + row, accepted + 1)
    if DIRECT_COMMIT:
        committed = tl.load(request_meta + row * 3 + 2) != 0
    else:
        state = tl.load(state_indices + row)
        committed = tl.load(seq_lens + row) >= tl.load(prefill_lens + state)
    tl.store(ledger + ledger_slot, tl.where(committed, charged, old))


def select_prefix(
    draft,
    targets,
    regrets,
    cu,
    request_meta,
    state_indices,
    seq_lens,
    prefill_lens,
    ledger,
    max_draft_tokens,
    *,
    budget,
    gate,
    mismatch_cap,
    direct_commit=False,
    output_dtype=torch.int64,
):
    batch = cu.numel() - 1
    output = torch.empty((batch, max_draft_tokens + 1), dtype=output_dtype, device=draft.device)
    count = torch.empty((batch,), dtype=torch.int32, device=draft.device)
    if batch:
        _select_prefix[(batch,)](
            draft,
            targets,
            regrets,
            cu,
            request_meta,
            state_indices,
            seq_lens,
            prefill_lens,
            ledger,
            output,
            count,
            output.stride(0),
            budget,
            gate,
            mismatch_cap,
            triton.next_power_of_2(max_draft_tokens + 1),
            direct_commit,
            num_warps=1,
        )
    return output, count


@triton.jit
def _gather_rows(source, row_indices, result, STRIDE: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source_row = tl.load(row_indices + row)
    values = tl.load(source + source_row * STRIDE + cols, cols < V, other=0.0).to(tl.float32)
    tl.store(result + row * V + cols, values, cols < V)


def gather_rows(logits, indices):
    rows, vocab = indices.numel(), logits.shape[1]
    output = torch.empty((rows, vocab), device=logits.device, dtype=torch.float32)
    if rows:
        _gather_rows[(rows, triton.cdiv(vocab, VOCAB_TILE))](
            logits, indices, output, logits.stride(0), vocab, VOCAB_TILE, num_warps=4
        )
    return output


@triton.jit
def _pack_v1(
    flat_draft,
    flat_targets,
    flat_regrets,
    bonus,
    cu_draft,
    draft,
    targets,
    regrets,
    cu_logits,
    BATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    begin = tl.load(cu_draft + row - 1, row > 0, other=0)
    end = tl.load(cu_draft + row)
    start = begin + row
    k = end - begin
    offsets = tl.arange(0, BLOCK)
    candidates = tl.load(flat_draft + begin + offsets, offsets < k, other=0)
    selected = tl.load(flat_targets + begin + offsets, offsets < k, other=0)
    values = tl.load(flat_regrets + begin + offsets, offsets < k, other=0.0)
    tl.store(draft + start, 0)
    tl.store(draft + start + offsets + 1, candidates, offsets < k)
    tl.store(targets + start + offsets, selected, offsets < k)
    tl.store(regrets + start + offsets, values, offsets < k)
    tl.store(targets + start + k, tl.load(bonus + row))
    tl.store(regrets + start + k, 0.0)
    tl.store(cu_logits + row, start)
    if row == BATCH - 1:
        tl.store(cu_logits + BATCH, end + BATCH)


def pack_v1(draft, targets, regrets, bonus, cu_draft, max_draft_tokens):
    batch = cu_draft.numel()
    rows = draft.numel() + batch
    packed_draft = torch.empty((rows,), device=draft.device, dtype=torch.int64)
    packed_targets = torch.empty((rows,), device=draft.device, dtype=torch.int64)
    packed_regrets = torch.empty((rows,), device=draft.device, dtype=torch.float32)
    cu_logits = torch.empty((batch + 1,), device=draft.device, dtype=torch.int32)
    if batch:
        _pack_v1[(batch,)](
            draft,
            targets,
            regrets,
            bonus,
            cu_draft,
            packed_draft,
            packed_targets,
            packed_regrets,
            cu_logits,
            batch,
            triton.next_power_of_2(max(1, max_draft_tokens)),
            num_warps=1,
        )
    return packed_draft, packed_targets, packed_regrets, cu_logits
