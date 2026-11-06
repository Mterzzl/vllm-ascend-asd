# SPDX-License-Identifier: Apache-2.0
"""Actual V2 sampler and DFlash input kernels, fixed fixtures, no model runs."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
import vllm
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

import vllm_ascend.patch.worker  # noqa: F401
from vllm_ascend.sample.asd import ASDConfig
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.spec_decode.asd import ASDRejectionSampler
from vllm_ascend.worker.v2.spec_decode.dflash import prepare_dflash_inputs

DEVICE = torch.device("npu:0")


@pytest.fixture(scope="module", autouse=True)
def initialize_device():
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    torch.npu.set_device(DEVICE)
    init_device_properties_triton()


def tensor(x, dtype=torch.int32):
    return torch.tensor(x, dtype=dtype, device=DEVICE)


def extract(path, name):
    node = next(x for x in ast.walk(ast.parse(path.read_text())) if isinstance(x, ast.FunctionDef) and x.name == name)
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    env = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env[name]


def test_layer_mapping_and_explicit_eagle_override():
    root = Path(vllm.__file__).resolve().parent / "v1/worker"
    v1 = extract(root / "gpu_model_runner.py", "_get_eagle3_aux_layers_from_config")
    v2 = extract(root / "gpu/spec_decode/eagle/eagle3_utils.py", "get_eagle3_aux_layers_from_config")
    cfg = NS(dflash_config={"target_layer_ids": [1, 16, 31, 46, 61]})
    spec = NS(draft_model_config=NS(hf_config=cfg))
    assert v1(NS(speculative_config=spec)) == v2(spec) == (2, 17, 32, 47, 62)
    cfg.eagle_aux_hidden_state_layer_ids = [2, 17, 32, 47, 62]
    assert v1(NS(speculative_config=spec)) == v2(spec) == tuple(cfg.eagle_aux_hidden_state_layer_ids)
    cfg = NS(dflash_config={})
    spec.draft_model_config.hf_config = cfg
    assert v1(NS(speculative_config=spec)) is None and v2(spec) is None


def batch(states, ids, drafts, seq_lens=None):
    slots = [states.req_id_to_index[x] for x in ids]
    lengths = [len(d) + 1 for d in drafts]
    cu = np.asarray([0] + list(np.cumsum(lengths)), dtype=np.int32)
    inputs = [t for d in drafts for t in [2] + d]
    n = len(inputs)
    return NS(
        idx_mapping_np=np.asarray(slots, dtype=np.int32),
        idx_mapping=tensor(slots),
        input_ids=tensor(inputs),
        positions=tensor(list(range(n)), torch.int64),
        logits_indices=tensor(list(range(n))),
        cu_num_logits=tensor(cu),
        cu_num_logits_np=cu,
        expanded_idx_mapping=tensor([s for s, c in zip(slots, lengths) for _ in range(c)]),
        expanded_local_pos=tensor([i for c in lengths for i in range(c)]),
        seq_lens=tensor(seq_lens or [100] * len(ids)),
    )


@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
def test_real_v2_adapter_multiple_rounds(mode):
    states = RequestState(4, 128, 32, 3, 3, DEVICE)
    sampler = Sampler(4, 3, DEVICE, states, mode, 4)
    for name in ["one", "prefill", "exact"]:
        states.add_request(name, 2, [1, 2], 0)
        sampler.add_request(states.req_id_to_index[name], 2, SamplingParams(temperature=0, logprobs=1))
    states.apply_staged_writes()
    sampler.apply_staged_writes()
    adapter = ASDRejectionSampler(
        sampler,
        NS(num_speculative_tokens=3, rejection_sample_method="strict"),
        DEVICE,
        states,
        ASDConfig(0.75, 0.25, 2),
    )
    data = batch(states, ["one", "prefill", "exact"], [[1, 0, 1], [], [0]], [100, 1, 100])
    scores = tensor(
        [[1, 0.5, -2], [1, 0, -2], [1, 0.5, -2], [0, 0, 2], [0, 2, 0], [1, 0, -2], [0, 0, 2]], torch.float32
    )
    out = adapter(scores, data)
    assert out.sampled_token_ids.cpu().tolist() == [[1, 0, 0, -1], [1, -1, -1, -1], [0, 2, -1, -1]]
    assert out.num_sampled.cpu().tolist() == [3, 1, 2]
    assert out.logprobs_tensors is not None
    assert adapter.ledger.values[adapter.ledger.slots["one"]].item() == 0.5
    assert adapter.ledger.values[adapter.ledger.slots["prefill"]].item() == 0
    # Request reordering must retain one’s spent=.5 and reject the next .5 cost.
    data2 = batch(states, ["exact", "one"], [[0], [1, 0, 1]])
    out2 = adapter(torch.cat([scores[5:], scores[:4]]), data2)
    assert out2.sampled_token_ids.cpu().tolist() == [[0, 2, -1, -1], [0, -1, -1, -1]]
    adapter.ledger.drop_finished(["one"])
    out3 = adapter(scores[:4], batch(states, ["one"], [[1, 0, 1]]))
    assert out3.sampled_token_ids.cpu().tolist() == [[1, 0, 0, -1]]
    strict = ASDRejectionSampler(
        sampler, NS(num_speculative_tokens=3, rejection_sample_method="strict"), DEVICE, states, ASDConfig(0, 0.25, 2)
    )
    assert strict(scores[:4], batch(states, ["one"], [[1, 0, 1]])).sampled_token_ids[0, 0].item() == 0
    with pytest.raises(ValueError, match="temperature"):
        adapter.validate_requests([NS(sampling_params=SamplingParams(temperature=1))])


@pytest.mark.parametrize("k", [8, 15])
def test_dflash_positions_slots_rejected_suffix_and_padding(k):
    buf = AscendInputBuffers(3, 64, DEVICE)
    context_pos = tensor([99] * 64, torch.int64)
    context_slots = tensor([99] * 64)
    query_slots = tensor([99] * 64, torch.int64)
    indices = tensor([99] * (3 * k), torch.int64)
    pos = tensor([99] * (3 * k), torch.int64)
    mappings = tensor([99] * (3 * k))
    target = NS(
        num_reqs=2,
        num_scheduled_tokens=np.asarray([4, 2]),
        positions=tensor([0, 1, 2, 3, 10, 11], torch.int64),
        query_start_loc=tensor([0, 4, 6]),
        idx_mapping=tensor([2, 0]),
    )
    temperatures = tensor([0.0, 0.0, 0.0], torch.float32)
    seeds = tensor([0, 0, 0], torch.int64)
    prepare_dflash_inputs(
        buf,
        query_slots,
        context_pos,
        context_slots,
        indices,
        pos,
        mappings,
        temperatures,
        seeds,
        target,
        tensor([2, 1]),
        tensor([2, 0]),
        tensor([7, 8, 9]),
        tensor([4, 5, 6]),
        temperatures,
        seeds,
        tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        16,
        0,
        1,
        1,
        31,
        k + 1,
        k,
        3,
        64,
        64,
    )
    assert buf.input_ids[: k + 1].cpu().tolist() == [9] + [31] * k
    assert buf.input_ids[k + 1 : 2 * (k + 1)].cpu().tolist() == [7] + [31] * k
    assert buf.positions[: k + 1].cpu().tolist() == list(range(2, 3 + k))
    assert context_slots[:6].cpu().tolist() == [16, 17, -1, -1, 90, 91]
    assert indices[:k].cpu().tolist() == list(range(1, k + 1))
    assert mappings[: 2 * k].cpu().tolist() == [2] * k + [0] * k
    assert mappings[2 * k :].cpu().tolist() == [-1] * k
    assert query_slots[2 * (k + 1) :].cpu().tolist() == [-1] * (64 - 2 * (k + 1))
    assert buf.seq_lens.cpu().tolist() == [k + 3, k + 13, 0]


def test_pinned_input_batch_dummy_fields():
    buf = AscendInputBuffers(3, 64, DEVICE)
    b = AscendInputBatch.make_dummy(2, 18, buf)
    assert b.seq_lens_np.tolist() == [9, 9]


@pytest.mark.parametrize("for_capture", [False, True])
def test_dflash_noncausal_metadata_clears_generic_mask(monkeypatch, for_capture):
    import vllm_ascend.worker.v2.spec_decode.dflash as module

    s = module.AscendDFlashSpeculator.__new__(module.AscendDFlashSpeculator)
    s.num_query_per_req = 16
    s.max_model_len = 128
    s.input_buffers = AscendInputBuffers(3, 64, DEVICE)
    s.block_tables = NS(input_block_tables=[tensor([[1], [2], [0]])], slot_mappings=tensor([[-1] * 64]))
    s.attn_groups = []
    s.kv_cache_config = object()
    metadata = NS(causal=False, attn_mask=object())

    def build(**kwargs):
        assert kwargs["causal"] is False
        assert kwargs["for_cudagraph_capture"] == for_capture
        return {"draft": metadata}

    monkeypatch.setattr(module, "build_attn_metadata", build)
    result = s.build_metadata(2, 3, 48, np.asarray([16, 16]), for_capture=for_capture)
    assert result["draft"].causal is False
    assert result["draft"].attn_mask is None
    assert result["draft"].actual_seq_lengths_q == [16, 32, 48]


@pytest.mark.parametrize("num_requests", [1, 16])
def test_noncausal_k15_fia_graph(num_requests):
    # K15 gives sixteen parallel query positions. A uniform zero-logit
    # attention over values 0..15 must return 7.5 at *every* query, including
    # the first; this distinguishes noncausal from causal attention.
    q = torch.zeros((16 * num_requests, 32, 128), dtype=torch.bfloat16, device=DEVICE)
    k = torch.zeros((num_requests + 1, 128, 512), dtype=torch.bfloat16, device=DEVICE)
    v = (
        torch.arange(128, device=DEVICE, dtype=torch.bfloat16)[None, :, None]
        .expand(num_requests + 1, 128, 512)
        .contiguous()
    )
    kwargs = dict(
        query=q,
        key=k,
        value=v,
        block_table=tensor([[i + 1] for i in range(num_requests)]),
        atten_mask=None,
        input_layout="TND",
        block_size=128,
        actual_seq_lengths=[16 * (i + 1) for i in range(num_requests)],
        actual_seq_lengths_kv=[16] * num_requests,
        num_heads=32,
        num_key_value_heads=4,
        sparse_mode=0,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        scale=128**-0.5,
    )
    torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kwargs)
    torch_npu.npu_fused_infer_attention_score(**kwargs)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, _ = torch_npu.npu_fused_infer_attention_score(**kwargs)
    graph.replay()
    torch.testing.assert_close(output.cpu(), torch.full_like(output.cpu(), 7.5), rtol=0, atol=0)


def test_runner_finishes_only_completed_ledgers(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.sample.asd import RequestLedger
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    ledger = RequestLedger(2, DEVICE)
    ledger.prepare(["finished", "preempted"])
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner._asd_sampler = NS(ledger=ledger)
    calls = []
    monkeypatch.setattr(GPUModelRunner, "finish_requests", lambda self, output: calls.append(output))
    output = NS(finished_req_ids={"finished"}, preempted_req_ids={"preempted"})
    runner.finish_requests(output)
    assert set(ledger.slots) == {"preempted"}
    assert calls == [output]


def test_pinned_block_table_and_mixed_prefill_preparation():
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from vllm_ascend.worker.v2.block_table import AscendBlockTables
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner
    from vllm_ascend.worker.v2.states import AscendRequestState

    # Match the real runner's keyword call, not only a positional fixture.
    tables = AscendBlockTables(
        block_sizes=[16],
        max_num_reqs=3,
        max_num_batched_tokens=64,
        max_model_len=128,
        device=DEVICE,
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
    )
    assert tables.block_sizes == [16] and tables.slot_mappings.dtype == torch.int32
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.req_states = AscendRequestState(3, 128, 64, 8, 32, DEVICE)
    runner.req_states.add_request("prefill", 4, [1, 2, 3, 4], 0)
    runner.req_states.add_request("decode", 2, [5, 6], 2)
    runner.req_states.apply_staged_writes()
    runner.input_buffers = AscendInputBuffers(3, 64, DEVICE)
    runner.max_num_reqs = 3
    runner.num_speculative_steps = 8
    runner.device = DEVICE
    runner.vllm_config = NS(
        speculative_config=NS(num_speculative_tokens=8, method="dflash"),
        model_config=NS(runner_type="generate"),
        scheduler_config=NS(enable_chunked_prefill=True),
    )
    runner.num_computed_tokens_event = torch.npu.Event()
    runner.num_computed_tokens_cpu = torch.zeros(3, dtype=torch.int32)
    output = NS(
        total_num_scheduled_tokens=7,
        num_scheduled_tokens={"prefill": 4, "decode": 3},
        scheduled_spec_decode_tokens={"decode": [7, 8]},
        scheduled_cached_reqs=NS(req_ids=[]),
        has_structured_output_requests=False,
    )
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=7, num_reqs=2)
    data = runner.prepare_inputs(output, desc)
    assert data.req_ids == ["decode", "prefill"]
    assert data.cu_num_logits_np.tolist() == [0, 3, 4]
    assert data.positions.cpu().tolist() == [2, 3, 4, 0, 1, 2, 3]
    assert data.seq_lens.cpu().tolist() == [5, 4]
    assert data.input_ids[3:].cpu().tolist() == [1, 2, 3, 4]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--confcutdir=" + str(Path(__file__).parent), "-q", *sys.argv[1:]]))
