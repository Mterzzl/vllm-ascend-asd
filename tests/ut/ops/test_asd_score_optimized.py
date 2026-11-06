# SPDX-License-Identifier: Apache-2.0
"""Optimized score/argmax against the archived pre-optimization Triton path."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.asd import gather_inputs
from vllm_ascend.ops.triton.spec_decode.asd_score import argmax_logits, score_logits

spec = importlib.util.spec_from_file_location(
    "asd_pre_optimization", Path(__file__).parent / "fixtures/asd_score_pre_optimization.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
DEVICE = "npu:0"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab", [31, 4097, 151936])
def test_exact_scores_strides_ties_invalid_and_offsets(dtype, vocab):
    generator = torch.Generator().manual_seed(817)
    host = (torch.randint(-64, 64, (7, vocab + 8), generator=generator).float() / 8).to(dtype)
    # Tie maxima across vocabulary blocks; row2 uses the final vocabulary ID.
    host[:, min(vocab - 1, 4095)] = 16
    host[:, min(vocab - 1, 4096)] = 16
    host[2, :vocab] = 0
    host[2, vocab - 1] = 16
    logits = host.to(DEVICE)[:, :vocab]
    candidates = torch.tensor([0, vocab - 1, 4, -1, vocab, 7, 12], dtype=torch.int32, device=DEVICE)
    for offset in [0, 1]:
        old = reference.score_logits(logits, candidates, candidate_offset=offset)
        new = score_logits(logits, candidates, candidate_offset=offset)
        for x, y in zip(old, new):
            torch.testing.assert_close(x.cpu(), y.cpu(), rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(argmax_logits(logits).cpu(), reference.score_logits(logits)[0].cpu(), rtol=0, atol=0)


def test_nonfinite_behavior_matches_original():
    host = torch.full((5, 4097), -float("inf"), dtype=torch.float32)
    host[1, [4, 4096]] = float("inf")
    host[2, 7] = 2
    host[3, 9] = float("nan")
    host[3, 11] = 3
    host[4] = float("nan")
    logits = host.to(DEVICE)
    draft = torch.tensor([0, 4, 8, 11, 0], dtype=torch.int32, device=DEVICE)
    old = reference.score_logits(logits, draft, candidate_offset=0)
    new = score_logits(logits, draft, candidate_offset=0)
    for x, y in zip(old, new):
        torch.testing.assert_close(x.cpu(), y.cpu(), rtol=0, atol=0, equal_nan=True)


def test_empty_and_dynamic_gather():
    empty = torch.empty((0, 31), dtype=torch.float32, device=DEVICE)
    ids, regrets = score_logits(empty)
    assert ids.shape == regrets.shape == (0,)
    assert argmax_logits(empty).shape == (0,)
    source = torch.arange(300, dtype=torch.int32, device=DEVICE)
    positions = source.to(torch.int64)
    for n in [1, 7, 16, 21, 31, 64, 127, 128, 240, 256]:
        idx = torch.arange(n, dtype=torch.int64, device=DEVICE).flip(0)
        actual = gather_inputs(source, positions, idx)
        assert actual[0].cpu().tolist() == list(reversed(range(n)))
        assert actual[1].cpu().tolist() == list(reversed(range(n)))


@pytest.mark.parametrize("rows", [16, 240])
def test_draft_argmax_graph_replay_observes_new_logits(rows):
    logits = torch.zeros((rows, 151936), dtype=torch.float32, device=DEVICE)
    logits[:, -1] = 9
    argmax_logits(logits)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = argmax_logits(logits)
    graph.replay()
    assert result.cpu().tolist() == [151935] * rows
    logits[:, 0] = 10
    graph.replay()
    assert result.cpu().tolist() == [0] * rows
