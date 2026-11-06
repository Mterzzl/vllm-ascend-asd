# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash single-chain adapter for the pinned vLLM 0.20.2 V2 interfaces.

Input preparation is backported from the existing DFlash V2 implementation.
Context KV is prepared outside the captured parallel-query forward.
"""

from contextlib import contextmanager

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer, init_attn_backend
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.compilation.acl_graph import set_draft_graph_params, update_full_graph_params
from vllm_ascend.ops.triton.spec_decode.logit_score import argmax_logits
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.utils import communicator_switch


@contextmanager
def draft_context():
    previous = (_EXTRA_CTX.is_draft_model, _EXTRA_CTX.is_draft_model_prefill, _EXTRA_CTX.capturing)
    _EXTRA_CTX.is_draft_model = True
    _EXTRA_CTX.is_draft_model_prefill = False
    _EXTRA_CTX.capturing = torch.npu.is_current_stream_capturing()
    try:
        yield
    finally:
        (_EXTRA_CTX.is_draft_model, _EXTRA_CTX.is_draft_model_prefill, _EXTRA_CTX.capturing) = previous


class DFlashAclGraphManager(CudaGraphManager):
    def __init__(self, speculator, mode):
        self.speculator = speculator
        super().__init__(speculator.vllm_config, speculator.device, mode, speculator.num_query_per_req)
        # Do not alias target graph intermediates containing live aux outputs.
        self.pool = torch.npu.graph_pool_handle()
        sizes = sorted({d.num_tokens for ds in self._capture_descs.values() for d in ds})
        if self.needs_capture():
            set_draft_graph_params(sizes)
        logger.info("DFlash V2 draft ACLGraph sizes=%s", sizes)

    def capture_model(self):
        s = self.speculator
        if not self.needs_capture():
            return

        def factory(desc):
            n, t = desc.num_reqs, desc.num_tokens
            batch = AscendInputBatch.make_dummy(n, t, s.input_buffers)
            tables = s.block_tables.get_dummy_block_tables(n)
            slots = s.block_tables.get_dummy_slot_mappings(t)
            metadata = s.build_metadata(n, n, t, batch.seq_lens_np, tables, slots, for_capture=True)
            by_layer = build_slot_mappings_by_layer(slots, s.kv_cache_config)
            s.sample_indices[: n * s.num_speculative_steps].zero_()
            return lambda mode: s.generate_draft(n, t, metadata, by_layer, mode)

        with communicator_switch():
            super().capture(factory, "Capturing DFlash V2 ACL graphs")

    def run_fullgraph(self, desc):
        s = self.speculator
        s.update_stream.wait_stream(torch.npu.current_stream())
        result = super().run_fullgraph(desc)
        with (
            set_forward_context(
                s.current_metadata, s.vllm_config, num_tokens=desc.num_tokens, cudagraph_runtime_mode=desc.cg_mode
            ),
            draft_context(),
        ):
            update_full_graph_params(
                next(iter(s.attn_backends.values())),
                s.update_stream,
                get_forward_context(),
                desc.num_tokens,
                s.vllm_config,
                s.speculative_config,
                desc.num_tokens,
                draft_attn_metadatas=[s.current_metadata],
            )
        return result


class AscendDFlashSpeculator(EagleSpeculator):
    """Qwen DFlash with this release's three-argument set_attn interface."""

    def __init__(self, vllm_config, device):
        pc = vllm_config.parallel_config
        if pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1 or pc.decode_context_parallel_size != 1:
            raise NotImplementedError("This DFlash V2 backport supports TP with PP=DP=CP=1")
        if vllm_config.speculative_config.rejection_sample_method != "strict":
            raise NotImplementedError("DFlash V2 backport requires strict/greedy sampling")
        super().__init__(vllm_config, device)
        self.num_query_per_req = self.num_speculative_steps + 1
        if self.num_query_per_req > 16:
            raise ValueError("Ascend DFlash single-chain attention supports K <= 15")
        if self.max_num_tokens < self.max_num_reqs * self.num_query_per_req:
            raise ValueError("max_num_batched_tokens must cover max_num_seqs * (K+1)")
        self.parallel_drafting_token_id = self.draft_model_config.hf_config.dflash_config["mask_token_id"]
        self.supports_mm_inputs = False
        self.input_buffers = AscendInputBuffers(self.max_num_reqs, self.max_num_tokens, device)
        self.context_positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
        size = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(size, dtype=torch.int64, device=device)
        self.sample_pos = torch.zeros_like(self.sample_indices)
        self.sample_idx_mapping = torch.full((size,), -1, dtype=torch.int32, device=device)
        self.update_stream = torch.npu.Stream()
        self.query_cudagraph_manager = None
        self.draft_logits = None

    def load_model(self, target_model):
        # The existing loader supports DFlashDraftModel and target embedding/head sharing.
        cfg = self.draft_model_config.hf_config
        cfg.is_neox_style = target_model.model.layers[0].self_attn.rotary_emb.is_neox_style
        super().load_model(target_model)

    def init_cudagraph_manager(self, cudagraph_mode):
        mode = (
            CUDAGraphMode.FULL_DECODE_ONLY if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL else CUDAGraphMode.NONE
        )
        self.query_cudagraph_manager = DFlashAclGraphManager(self, mode)

    def capture_model(self):
        self.query_cudagraph_manager.capture_model()

    def set_attn(self, model_state, kv_cache_config, block_tables):
        self.model_state, self.kv_cache_config, self.block_tables = model_state, kv_cache_config, block_tables
        self.attn_backends, self.attn_groups, _ = init_attn_backend(
            kv_cache_config, self.vllm_config, self.device, active_layer_names=self.draft_attn_layer_names
        )
        groups = [i for i, group in enumerate(self.attn_groups) if group]
        if len(groups) != 1:
            raise NotImplementedError("This Qwen DFlash adapter requires one draft KV group")
        self.draft_kv_cache_group_id = groups[0]
        self.context_slots = torch.empty(self.max_num_tokens, dtype=torch.int32, device=self.device)

    def build_metadata(self, n, padded_n, t, upper, tables=None, slots=None, for_capture=False):
        upper = np.asarray(upper[:n], dtype=np.int32)
        lengths = np.zeros(padded_n, dtype=np.int32)
        lengths[:n] = np.minimum(upper + (0 if for_capture else self.num_query_per_req), self.max_model_len)
        q_cpu = torch.arange(padded_n + 1, dtype=torch.int32) * self.num_query_per_req
        if tables is None:
            tables = [table[:padded_n] for table in self.block_tables.input_block_tables]
        if slots is None:
            slots = self.block_tables.slot_mappings[:, :t]
        metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=padded_n,
            num_tokens=t,
            query_start_loc_gpu=self.input_buffers.query_start_loc[: padded_n + 1],
            query_start_loc_cpu=q_cpu,
            max_query_len=self.num_query_per_req,
            seq_lens=self.input_buffers.seq_lens[:padded_n],
            max_seq_len=self.max_model_len,
            block_tables=tables,
            slot_mappings=slots,
            kv_cache_config=self.kv_cache_config,
            seq_lens_np=lengths,
            positions=self.input_buffers.positions[:t],
            attn_state=AscendAttentionState.SpecDecoding,
            causal=False,
            for_cudagraph_capture=for_capture,
        )
        # FIA needs padded cumulative query lengths to match the captured query size.
        for meta in metadata.values():
            # The release's generic builder returns a causal split-fuse mask
            # even for causal=False. Match the existing V1 DFlash proposer:
            # non-causal draft queries use sparse_mode=0 with no mask.
            meta.attn_mask = None
            meta.actual_seq_lengths_q = [(i + 1) * self.num_query_per_req for i in range(padded_n)]
        return metadata

    def generate_draft(self, n, t, metadata, slots, mode=CUDAGraphMode.NONE):
        with (
            set_forward_context(
                metadata,
                self.vllm_config,
                num_tokens=t,
                cudagraph_runtime_mode=mode,
                slot_mapping=slots,
                batch_descriptor=BatchDescriptor(num_tokens=t),
            ),
            draft_context(),
        ):
            hidden = self.model(
                input_ids=self.input_buffers.input_ids[:t],
                positions=self.input_buffers.positions[:t],
                inputs_embeds=None,
            )
        sample_hidden = hidden[self.sample_indices[: n * self.num_speculative_steps]]
        ids = argmax_logits(self.model.compute_logits(sample_hidden))
        self.draft_tokens[:n].copy_(ids.view(n, self.num_speculative_steps))

    def precompute_context(self, num_context_tokens, slots=None):
        # Context KV runs outside the captured query graph, but Ascend's RoPE
        # still reads the forward-context flags (including is_draft_model).
        with set_forward_context(None, self.vllm_config, num_tokens=num_context_tokens), draft_context():
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_context_tokens],
                self.context_positions[:num_context_tokens],
                slots,
            )

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        num_tokens_across_dp=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        n, ctx = input_batch.num_reqs, input_batch.num_tokens
        query = n * self.num_query_per_req
        if not aux_hidden_states:
            raise ValueError("DFlash V2 requires configured target auxiliary hidden states")
        hidden = self.model.combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))
        self.hidden_states[:ctx].copy_(hidden[:ctx])
        if dummy_run and skip_attn_for_dummy_run:
            self.precompute_context(ctx)
            self.generate_draft(n, query, None, None)
            return self.draft_tokens[:n]
        gid = self.draft_kv_cache_group_id
        prepare_dflash_inputs(
            self.input_buffers,
            self.block_tables.slot_mappings[gid],
            self.context_positions,
            self.context_slots,
            self.sample_indices,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            self.block_tables.input_block_tables[gid],
            self.block_tables.block_sizes[gid],
            0,
            1,
            1,
            self.parallel_drafting_token_id,
            self.num_query_per_req,
            self.num_speculative_steps,
            self.max_num_reqs,
            self.max_num_tokens,
            self.max_model_len,
        )
        self.precompute_context(ctx, None if dummy_run else self.context_slots[:ctx])
        desc, _ = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            n,
            query,
            uniform_token_count=self.num_query_per_req,
            dp_size=1,
            dp_rank=0,
            need_eager=is_profile,
        )
        padded_n, t = desc.num_reqs or n, desc.num_tokens
        self.current_metadata = self.build_metadata(n, padded_n, t, input_batch.seq_lens_cpu_upper_bound.numpy())
        slots = build_slot_mappings_by_layer(self.block_tables.slot_mappings[:, :t], self.kv_cache_config)
        if desc.cg_mode == CUDAGraphMode.FULL:
            self.query_cudagraph_manager.run_fullgraph(desc)
        else:
            self.generate_draft(n, t, self.current_metadata, slots, desc.cg_mode)
        return self.draft_tokens[:n]


@triton.jit
def _prepare_dflash_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Sampling params
    temperature_ptr,
    seeds_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    cp_rank,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)

    if block_idx > 0:
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    nrejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - nrejected

    nsampled = tl.load(num_sampled_ptr + req_idx)
    if nsampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req
    num_valid_ctx = valid_ctx_end - ctx_start
    pad_slot = tl.full((), PAD_SLOT_ID, dtype=tl.int64)
    zero_pos = tl.full((), 0, dtype=last_valid_pos.dtype)

    # --- Context positions / slots ---
    # Rejected suffix must be PAD: tree siblings can share RoPE with the
    # accepted prefix, and writing them would clobber draft KV. Matches
    # upstream is_valid_ctx (vllm _prepare_dflash_inputs_kernel).
    for j in range(0, num_ctx):
        ctx_pos_idx = ctx_start + j
        is_valid_ctx = j < num_valid_ctx
        ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx)
        ctx_block_num = ctx_pos // block_size
        ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
        ctx_block_id = tl.load(block_table_ptr + req_idx * block_table_stride + ctx_block_num).to(tl.int64)
        ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
        ctx_slot = tl.where(is_valid_ctx & (ctx_block_id != 0), ctx_slot, pad_slot)
        ctx_pos = tl.where(is_valid_ctx, ctx_pos, zero_pos)
        tl.store(out_context_positions_ptr + ctx_pos_idx, ctx_pos)
        tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, ctx_slot)

    # --- Query positions / input_ids / slots ---
    for q_off in range(0, num_query_per_req):
        query_pos = last_valid_pos + 1 + q_off
        query_idx = query_base + q_off
        if q_off == 0:
            input_id = bonus_token
        else:
            input_id = parallel_drafting_token_id

        q_block_num = query_pos // block_size
        q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
        q_block_id = tl.load(block_table_ptr + req_idx * block_table_stride + q_block_num).to(tl.int64)
        q_slot = tl.where(q_block_id != 0, q_block_id * block_size + (query_pos % block_size), pad_slot)

        tl.store(out_input_ids_ptr + query_idx, input_id)
        clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
        tl.store(out_query_positions_ptr + query_idx, clamped_query_pos)
        tl.store(out_query_slot_mapping_ptr + query_idx, q_slot)

    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    # --- Sample indices / positions / idx_mapping ---
    for s_off in range(sample_off, num_query_per_req):
        sample_idx = req_idx * num_speculative_steps + (s_off - sample_off)
        query_idx = query_base + s_off
        query_pos = last_valid_pos + 1 + s_off
        sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
        tl.store(out_sample_indices_ptr + sample_idx, query_idx)
        tl.store(out_sample_pos_ptr + sample_idx, sample_pos)
        tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx)

    tl.store(out_query_start_loc_ptr + req_idx, query_base)
    # seq_lens is the absolute sequence length the draft attention
    # reads up to (context + query), not just the count of accepted
    # tokens this step.
    tl.store(out_seq_lens_ptr + req_idx, tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len))
    # Copy sampling state (added upstream in vllm-project/vllm#50000).
    tl.store(
        out_temperature_ptr + req_state_idx,
        tl.load(temperature_ptr + req_state_idx),
    )
    tl.store(
        out_seeds_ptr + req_state_idx,
        tl.load(seeds_ptr + req_state_idx),
    )

    if req_idx == num_reqs - 1:
        # Pad per-request buffers to max_num_reqs for CUDA graph safety.
        last_query_end = num_reqs * num_query_per_req
        for i in range(num_reqs, max_num_reqs + 1):
            tl.store(out_query_start_loc_ptr + i, last_query_end)
        for i in range(num_reqs, max_num_reqs):
            tl.store(out_seq_lens_ptr + i, 0)
        # Padded sample slots point at query index 0 (a valid row in
        # last_hidden_states) so CG replay never reads OOB. Padded sample
        # idx mappings point to -1, which is ignored during sampling.
        pad_start = num_reqs * num_speculative_steps
        pad_end = max_num_reqs * num_speculative_steps
        for i in range(pad_start, pad_end):
            tl.store(out_sample_indices_ptr + i, 0)
            tl.store(out_sample_pos_ptr + i, 0)
            tl.store(out_sample_idx_mapping_ptr + i, -1)
        # Pad query slot mappings past num_query_tokens with PAD so the
        # captured CG sees PAD slots (no K/V write) for replay sizes
        # larger than the current request count.
        q_pad_start = num_reqs * num_query_per_req
        for i in range(q_pad_start, max_num_tokens):
            tl.store(out_query_slot_mapping_ptr + i, PAD_SLOT_ID)


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs]
    input_temperature: torch.Tensor,
    # [max_num_reqs]
    input_seeds: torch.Tensor,
    # [max_num_reqs, max_num_blocks]
    block_table: torch.Tensor,
    block_size: int,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    # Cover the longest possible per-request span (ctx + query). Use the max
    # per-request query length, not the total token count across the batch.
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        cp_rank,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        BLOCK_SIZE=BLOCK_SIZE,
    )
