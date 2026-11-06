# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official Qwen-facing ASD policy on the Ascend V2 single-chain runner."""

import numpy as np
from vllm.logger import logger
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from vllm_ascend.ops.triton.spec_decode.asd import gather_inputs, score_logits, select_prefix
from vllm_ascend.sample.asd import RequestLedger, validate_speculative_config


class ASDRejectionSampler(RejectionSampler):
    """ASD replaces only greedy prefix verification; model logit processing stays upstream."""

    def __init__(self, sampler, spec_config, device, req_states, config):
        validate_speculative_config(spec_config)
        super().__init__(sampler, spec_config, device)
        self.req_states = req_states
        self.config = config
        self.ledger = RequestLedger(req_states.max_num_reqs, device)
        logger.info(
            "ASD enabled (Qwen reference): B=%s, g=%s, M=%s; Triton FP32 verification",
            config.budget,
            config.gate,
            config.mismatch_cap,
        )

    def validate_requests(self, new_requests):
        for request in new_requests:
            params = request.sampling_params
            if params is not None and params.temperature != 0:
                raise ValueError("Official ASD requires temperature=0")

    def __call__(self, logits, input_batch, draft_logits=None):
        if logits.shape[1] < self.req_states.vocab_size:
            raise ValueError("ASD requires full-vocabulary target logits")
        counts = np.diff(input_batch.cu_num_logits_np)
        if np.any(counts < 1) or np.any(counts > self.num_speculative_steps + 1):
            raise ValueError("ASD requires packed contiguous single-chain verification rows")
        if np.any(self.sampler.sampling_states.temperature.np[input_batch.idx_mapping_np] != 0):
            raise ValueError("Official ASD requires temperature=0 for every request")
        request_ids = [self.req_states.index_to_req_id[int(i)] for i in input_batch.idx_mapping_np]
        draft, positions = gather_inputs(input_batch.input_ids, input_batch.positions, input_batch.logits_indices)
        processed = self.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            positions,
            draft,
            input_batch.expanded_local_pos,
        )
        targets, regrets = score_logits(processed, draft)
        metadata = self.ledger.prepare(request_ids)
        sampled, count = select_prefix(
            draft,
            targets,
            regrets,
            input_batch.cu_num_logits,
            metadata,
            input_batch.idx_mapping,
            input_batch.seq_lens,
            self.req_states.prefill_len.gpu,
            self.ledger.values,
            self.num_speculative_steps,
            budget=self.config.budget,
            gate=self.config.gate,
            mismatch_cap=self.config.mismatch_cap,
        )
        logprobs: LogprobsTensors | None = self._get_logprobs_tensors(
            input_batch, sampled, count, processed if self.sampler.logprobs_mode == "processed_logprobs" else logits
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            num_sampled=count,
            logprobs_tensors=logprobs,
            num_nans=get_num_nans(logits) if self.sampler.compute_nans else None,
        )
