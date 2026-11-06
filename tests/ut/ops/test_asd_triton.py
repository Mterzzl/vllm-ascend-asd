# SPDX-License-Identifier: Apache-2.0
"""Fixed-input NPU conformance; no model inference or benchmark data.

Run with --confcutdir=tests/ut/ops to avoid the global mocked UT backend.
The bundled pinned ASD author reference is the unchanged CPU oracle.
Torch operations in this file build/check fixtures, not the runtime algorithm.
"""

import random
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from asd_reference.budget import RequestRiskState, TokenScores, choose_prefix  # noqa: E402
from asd_reference.config import ASDConfig as ReferenceConfig  # noqa: E402

from vllm_ascend.ops.triton.spec_decode.asd import gather_inputs, gather_rows, pack_v1, score_logits, select_prefix
from vllm_ascend.sample.asd import RequestLedger

DEVICE = "npu:0"


def tensor(value, dtype=torch.int32):
    return torch.tensor(value, dtype=dtype, device=DEVICE)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab", [31, 1031, 152064])
def test_scores_and_gather(dtype, vocab):
    generator = torch.Generator().manual_seed(704)
    source = (torch.randint(-32, 32, (7, vocab + 8), generator=generator).float() / 8).to(dtype)
    source[:, 0] = 8
    source[:, -9] = 8  # Equal maxima must pick ID 0, including a tile boundary.
    logits = source.to(DEVICE)[:, :vocab]
    candidates = tensor([0, 1, 2, 3, 4, 5, vocab - 1])
    ids, costs = score_logits(logits, candidates, candidate_offset=0)
    host = source[:, :vocab].float()
    expected = host.max(-1)
    assert ids.cpu().tolist() == expected.indices.tolist()
    torch.testing.assert_close(
        costs.cpu(), expected.values - host[torch.arange(7), candidates.cpu().long()], rtol=0, atol=0
    )
    gathered = gather_rows(logits, tensor([6, 0, 3]))
    torch.testing.assert_close(gathered.cpu(), host[[6, 0, 3]], rtol=0, atol=0)
    shifted, shifted_cost = score_logits(logits, candidates)
    assert shifted.cpu().tolist() == expected.indices.tolist()
    torch.testing.assert_close(
        shifted_cost[:-1].cpu(),
        expected.values[:-1] - host[torch.arange(6), candidates.cpu().long()[1:]],
        rtol=0,
        atol=0,
    )
    assert shifted_cost[-1].item() == float("inf")


def run_blocks(drafts, tops, regrets, ledger, ids, config, commit=None):
    lengths = [len(x) for x in drafts]
    endpoints = []
    for length in lengths:
        endpoints.append(length + (endpoints[-1] if endpoints else 0))
    bonus = [71 + i for i in range(len(drafts))]
    packed = pack_v1(
        tensor(sum(drafts, [])),
        tensor(sum(tops, [])),
        tensor(sum(regrets, []), torch.float32),
        tensor(bonus),
        tensor(endpoints),
        max(lengths),
    )
    draft, target, cost, cu = packed
    metadata = ledger.prepare(ids, commit)
    out, count = select_prefix(
        draft,
        target,
        cost,
        cu,
        metadata,
        cu,
        cu,
        cu,
        ledger.values,
        max(lengths),
        budget=config[0],
        gate=config[1],
        mismatch_cap=config[2],
        direct_commit=True,
        output_dtype=torch.int32,
    )
    return out.cpu().tolist(), count.cpu().tolist(), bonus


@pytest.mark.parametrize("config", [(8.0, 0.25, 2), (1.0, 0.5, 4), (0.0, 0.25, 2), (8.0, 0.0, 2), (8.0, 0.25, 0)])
def test_official_oracle_multiple_rounds(config):
    rng = random.Random(617)
    ledger = RequestLedger(2, DEVICE)
    ids = [f"request-{i}" for i in range(24)]
    states = {key: RequestRiskState(config[0]) for key in ids}
    ref_config = ReferenceConfig(*config)
    for round_index in range(12):
        rng.shuffle(ids)  # Active row is not the ledger identity.
        drafts, tops, regrets = [], [], []
        for row, key in enumerate(ids):
            k = [0, 1, 2, 8, 15, 16][row % 6]
            d = [rng.randrange(2) for _ in range(k)]
            r = [0.0 if token == 0 else rng.randrange(17) / 8 for token in d]
            drafts.append(d)
            tops.append([0] * k)
            regrets.append(r)
        output, counts, bonus = run_blocks(drafts, tops, regrets, ledger, ids, config)
        for row, key in enumerate(ids):
            k = len(drafts[row])
            decision = choose_prefix(
                draft_token_ids=drafts[row],
                scores=TokenScores((0.0,) * k, tuple(tops[row]), tuple(-r for r in regrets[row])),
                state=states[key],
                config=ref_config,
            )
            n = decision.accepted_tokens
            expected = drafts[row][:n] + [tops[row][n] if n < k else bonus[row]]
            assert counts[row] == n + 1, (round_index, key, decision, output[row])
            assert output[row] == expected + [-1] * (17 - len(expected))
        actual = ledger.values.cpu().tolist()
        for key in ids:
            assert actual[ledger.slots[key]] == states[key].spent
    # A finished/reused request starts a fresh budget; a preempted one persists.
    ledger.drop_finished([ids[0]])
    states[ids[0]] = RequestRiskState(config[0])
    output, counts, _ = run_blocks([[1, 0], [1, 0]], [[0, 0]] * 2, [[0.25, 0.0]] * 2, ledger, ids[:2], config)
    for row, key in enumerate(ids[:2]):
        decision = choose_prefix(
            draft_token_ids=[1, 0],
            scores=TokenScores((0.0, 0.0), (0, 0), (-0.25, 0.0)),
            state=states[key],
            config=ref_config,
        )
        assert counts[row] == decision.accepted_tokens + 1
        assert ledger.values.cpu()[ledger.slots[key]].item() == states[key].spent


def test_ties_boundaries_discard_and_prefill():
    ledger = RequestLedger(1, DEVICE)
    # Exact budget exhausted; zero-regret ID mismatch still counts toward M.
    run_blocks([[1]], [[0]], [[0.5]], ledger, ["a"], (0.5, 0.5, 2))
    out, count, _ = run_blocks([[1, 1, 1]], [[0, 0, 0]], [[0.0, 0.0, 0.0]], ledger, ["a"], (0.5, 0.5, 2))
    assert count == [3] and out[0] == [1, 1, 0, -1]
    assert ledger.values.cpu()[0].item() == 0.5
    ledger.drop_finished(["a"])
    run_blocks([[1]], [[0]], [[0.5]], ledger, ["a"], (0.5, 0.5, 2), [False])
    assert ledger.values.cpu()[0].item() == 0  # Discarded partial prefill cannot spend.
    # Device-side V2 commit guard must use request-state mapping, not row index.
    draft, target, regret, cu = pack_v1(
        tensor([1, 1]), tensor([0, 0]), tensor([0.25, 0.25], torch.float32), tensor([9, 9]), tensor([1, 2]), 1
    )
    meta = ledger.prepare(["a", "b"])
    select_prefix(
        draft,
        target,
        regret,
        cu,
        meta,
        tensor([2, 0]),
        tensor([9, 5]),
        tensor([5, 0, 10]),
        ledger.values,
        1,
        budget=1.0,
        gate=0.25,
        mismatch_cap=2,
    )
    values = ledger.values.cpu().tolist()
    assert values[ledger.slots["a"]] == 0 and values[ledger.slots["b"]] == 0.25
    got, positions = gather_inputs(tensor([8, 4, 6]), tensor([1, 2, 3]), tensor([2, 0]))
    assert got.cpu().tolist() == [6, 8] and positions.cpu().tolist() == [3, 1]


def test_fp32_reference_boundary_is_explicit():
    # This documents a real limitation rather than claiming float64 identity.
    # Both inputs are exactly representable FP32; their ratio rounds down to g.
    regret = float(torch.tensor(0.1, dtype=torch.float32))
    gate = float(torch.tensor(regret / 5, dtype=torch.float32))
    assert regret / 5 > gate
    ledger = RequestLedger(1, DEVICE)
    _, count, _ = run_blocks(
        [[1, 0, 0, 0, 0]], [[0] * 5], [[regret, 0.0, 0.0, 0.0, 0.0]], ledger, ["boundary"], (1.0, gate, 2)
    )
    state = RequestRiskState(1.0)
    reference = choose_prefix(
        draft_token_ids=[1, 0, 0, 0, 0],
        scores=TokenScores((0.0,) * 5, (0,) * 5, (-regret, 0.0, 0.0, 0.0, 0.0)),
        state=state,
        config=ReferenceConfig(1.0, gate, 2),
    )
    assert reference.accepted_tokens == 0
    assert count == [6]


@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
def test_v1_real_adapter(mode, monkeypatch):
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

    from vllm_ascend.sample.asd_rejection_sampler import ASDRejectionSampler

    monkeypatch.setenv("VLLM_ASCEND_ASD_RISK_BUDGET", "8")
    monkeypatch.setenv("VLLM_ASCEND_ASD_MAX_REGRET_PER_VALUE", ".25")
    monkeypatch.setenv("VLLM_ASCEND_ASD_MAX_RELAXED_TOKENS_PER_BLOCK", "2")
    drafts = [[1, 0, 1], [], [0]]
    meta = SpecDecodeMetadata(
        draft_token_ids=tensor([1, 0, 1, 0]),
        num_draft_tokens=[3, 0, 1],
        cu_num_draft_tokens=tensor([3, 3, 4]),
        cu_num_sampled_tokens=tensor([4, 5, 7]),
        target_logits_indices=tensor([0, 1, 2, 5]),
        bonus_logits_indices=tensor([3, 4, 6]),
        logits_indices=tensor(list(range(7))),
    )
    sampling = SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=1,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=tensor([0.0] * 3, torch.float32),
        presence_penalties=tensor([0.0] * 3, torch.float32),
        repetition_penalties=tensor([1.0] * 3, torch.float32),
        output_token_ids=[[], [], []],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=drafts,
    )
    logits = tensor(
        [
            [1.0, 0.5, -2.0],
            [1.0, 0.0, -2.0],
            [1.0, 0.5, -2.0],
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
            [1.0, 0.0, -2.0],
            [0.0, 0.0, 2.0],
        ],
        torch.float32,
    )
    adapter = ASDRejectionSampler(Sampler(mode), 3, torch.device(DEVICE), 3)
    adapter.set_request_context(["one", "prefill", "strict"], [])
    result = adapter(meta, None, logits, sampling)
    assert result.sampled_token_ids.cpu().tolist() == [[1, 0, 0, -1], [1, -1, -1, -1], [0, 2, -1, -1]]
    assert adapter.ledger.values.cpu().tolist() == [0.5, 0.0, 0.0]
    assert result.logprobs_tensors is not None
    # A missing per-batch context must not silently use stale request identities.
    with pytest.raises(RuntimeError, match="context"):
        adapter(meta, None, logits, sampling)
    # B=0 uses exactly the greedy prefix and charges nothing.
    monkeypatch.setenv("VLLM_ASCEND_ASD_RISK_BUDGET", "0")
    strict = ASDRejectionSampler(Sampler(mode), 3, torch.device(DEVICE), 3)
    strict.set_request_context(["one", "prefill", "strict"], [])
    result = strict(meta, None, logits, sampling)
    assert result.sampled_token_ids.cpu().tolist() == [[0, -1, -1, -1], [1, -1, -1, -1], [0, 2, -1, -1]]
    assert strict.ledger.values.cpu().tolist() == [0.0, 0.0, 0.0]


def test_env_validation(monkeypatch):
    from vllm_ascend import envs
    from vllm_ascend.sample.asd import ASDConfig

    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ASD", "0")
    assert not envs.VLLM_ASCEND_ENABLE_ASD
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ASD", "1")
    assert envs.VLLM_ASCEND_ENABLE_ASD
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ASD", "perhaps")
    with pytest.raises(ValueError):
        _ = envs.VLLM_ASCEND_ENABLE_ASD
    for value in ["nan", "-1", "inf", "1e100", "1e-100"]:
        monkeypatch.setenv("VLLM_ASCEND_ASD_RISK_BUDGET", value)
        with pytest.raises(ValueError):
            ASDConfig.from_env()


@pytest.mark.parametrize("max_tokens,eos,expected", [(10, 1, [1]), (1, 99, [1]), (2, 99, [1, 0])])
def test_upstream_termination(max_tokens, eos, expected):
    from vllm import SamplingParams
    from vllm.v1.core.sched.utils import check_stop
    from vllm.v1.request import Request

    ledger = RequestLedger(1, DEVICE)
    out, count, _ = run_blocks([[1, 0]], [[0, 0]], [[0.25, 0.0]], ledger, ["ending"], (8.0, 0.25, 2))
    params = SamplingParams(temperature=0, max_tokens=max_tokens)
    params.update_from_generation_config({}, eos_token_id=eos)
    request = Request("ending", [8, 9], params, None)
    kept = []
    for token in out[0][: count[0]]:
        request.append_output_token_ids(token)
        kept.append(token)
        if check_stop(request, max_model_len=128):
            break
    assert kept == expected and request.is_finished()
    ledger.drop_finished(["ending"])
    assert not ledger.slots


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--confcutdir=" + str(Path(__file__).parent), "-q", *sys.argv[1:]]))
