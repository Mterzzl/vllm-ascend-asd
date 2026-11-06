# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-wise Triton top-1/regret; dynamic batch size, exact finite-score rule."""

import torch
from vllm.triton_utils import tl, triton

VOCAB_BLOCK = 4096


@triton.jit(do_not_specialize=["ROWS"])
def _row_argmax_regret(
    logits,
    draft,
    targets,
    regrets,
    ROWS,
    STRIDE: tl.constexpr,
    VOCAB: tl.constexpr,
    OFFSET: tl.constexpr,
    HAS_CANDIDATE: tl.constexpr,
    WRITE_REGRET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    best_values = tl.full((BLOCK,), -float("inf"), tl.float32)
    best_ids = tl.full((BLOCK,), VOCAB, tl.int32)
    nan_lanes = tl.full((BLOCK,), False, tl.int1)
    # Accumulate a maximum per lane, then reduce once. Unlike a sum, maximum
    # and minimum-ID tie selection do not introduce rounding accumulation.
    for start in range(0, VOCAB, BLOCK):
        cols = start + offsets
        valid = cols < VOCAB
        values = tl.load(logits + row * STRIDE + cols, valid, other=-float("inf")).to(tl.float32)
        nan_lanes = nan_lanes | (values != values)
        take = valid & ((values > best_values) | ((values == best_values) & (cols < best_ids)))
        best_ids = tl.where(take, cols, best_ids)
        best_values = tl.maximum(best_values, values)
    maximum = tl.max(best_values, 0)
    target = tl.min(tl.where(best_values == maximum, best_ids, VOCAB), 0)
    # Preserve the original tiled reduction's NaN sentinel and infinite cost.
    # A later masked load must not erase a NaN observed in an earlier block.
    has_nan = tl.sum(nan_lanes.to(tl.int32), 0) != 0
    target = tl.where(has_nan, VOCAB, target)
    tl.store(targets + row, target)
    if WRITE_REGRET:
        regret = 0.0
        if HAS_CANDIDATE:
            candidate = tl.load(draft + row + OFFSET, row + OFFSET < ROWS, other=-1)
            valid_candidate = (candidate >= 0) & (candidate < VOCAB)
            candidate_logit = tl.load(logits + row * STRIDE + candidate, valid_candidate, other=-float("inf")).to(
                tl.float32
            )
            finite = (~has_nan) & (tl.abs(maximum) < float("inf")) & (tl.abs(candidate_logit) < float("inf"))
            regret = tl.where(valid_candidate & finite, tl.maximum(maximum - candidate_logit, 0.0), float("inf"))
        tl.store(regrets + row, regret)


def _launch(logits, draft=None, *, candidate_offset=1, write_regret=True, block=VOCAB_BLOCK):
    rows, vocab = logits.shape
    targets = torch.empty((rows,), dtype=torch.int64, device=logits.device)
    regrets = torch.empty((rows,), dtype=torch.float32, device=logits.device) if write_regret else None
    if rows:
        _row_argmax_regret[(rows,)](
            logits,
            draft if draft is not None else targets,
            targets,
            regrets if regrets is not None else targets,
            rows,
            logits.stride(0),
            vocab,
            candidate_offset,
            draft is not None,
            write_regret,
            block,
            num_warps=4,
        )
    return targets, regrets


def score_logits(logits, draft=None, *, candidate_offset=1):
    return _launch(logits, draft, candidate_offset=candidate_offset)


def argmax_logits(logits):
    """Draft greedy selection without unused regret output or scratch buffers."""
    return _launch(logits, write_regret=False)[0]
