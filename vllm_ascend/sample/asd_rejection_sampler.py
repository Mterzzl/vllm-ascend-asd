# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton ASD on the Ascend V1 single-chain verification interface."""

import torch
from vllm.logger import logger
from vllm.v1.outputs import SamplerOutput

from vllm_ascend.ops.triton.spec_decode.asd import gather_rows, pack_v1, score_logits, select_prefix
from vllm_ascend.sample.asd import ASDConfig, RequestLedger
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler


class ASDRejectionSampler(AscendRejectionSampler):
    def __init__(self, sampler, capacity, device, vocab_size):
        super().__init__(sampler)
        self.asd_config = ASDConfig.from_env()
        self.ledger = RequestLedger(capacity, device)
        self.vocab_size = vocab_size
        self._context = None
        logger.info(
            "Triton ASD enabled (official Qwen rule): B=%s, g=%s, M=%s",
            self.asd_config.budget,
            self.asd_config.gate,
            self.asd_config.mismatch_cap,
        )

    def set_request_context(self, request_ids, discarded_rows):
        discarded = {int(i) for i in discarded_rows}
        if any(i < 0 or i >= len(request_ids) for i in discarded):
            raise ValueError("ASD received invalid discarded request indices")
        self._context = (list(request_ids), [i not in discarded for i in range(len(request_ids))])

    def forward(self, metadata, draft_probs, logits, sampling_metadata):
        if not sampling_metadata.all_greedy:
            raise ValueError("Official ASD supports temperature=0 only")
        if logits.shape[1] < self.vocab_size:
            raise ValueError("ASD requires full-vocabulary target logits; disable reduced/sharded LM-head sampling")
        if self._context is None:
            raise RuntimeError("ASD request context was not installed by the model runner")
        request_ids, committed = self._context
        self._context = None
        if len(request_ids) != len(metadata.num_draft_tokens):
            raise ValueError("ASD request and speculative block counts differ")

        # Triton gathers/casts each original score row. Existing upstream
        # sampling processors remain responsible for bias, masks and penalties.
        bonus = gather_rows(logits, metadata.bonus_logits_indices)
        targets = gather_rows(logits, metadata.target_logits_indices)
        raw_bonus = raw_targets = None
        want_logprobs = sampling_metadata.max_num_logprobs is not None
        if want_logprobs and not self.is_processed_logprobs_mode:
            raw_bonus = gather_rows(logits, metadata.bonus_logits_indices)
            raw_targets = gather_rows(logits, metadata.target_logits_indices)
        bonus = self.sampler.apply_logits_processors(bonus, sampling_metadata, predict_bonus_token=True)
        if metadata.draft_token_ids.numel():
            targets = self.apply_logits_processors(targets, sampling_metadata, metadata)

        target_ids, regrets = score_logits(targets, metadata.draft_token_ids, candidate_offset=0)
        bonus_ids, _ = score_logits(bonus)
        draft, target_ids, regrets, cu = pack_v1(
            metadata.draft_token_ids,
            target_ids,
            regrets,
            bonus_ids,
            metadata.cu_num_draft_tokens,
            metadata.max_spec_len,
        )
        cfg = self.asd_config
        request_meta = self.ledger.prepare(request_ids, committed)
        output, _ = select_prefix(
            draft,
            target_ids,
            regrets,
            cu,
            request_meta,
            cu,
            cu,
            cu,
            self.ledger.values,
            metadata.max_spec_len,
            budget=cfg.budget,
            gate=cfg.gate,
            mismatch_cap=cfg.mismatch_cap,
            direct_commit=True,
            output_dtype=torch.int32,
        )
        logprobs = None
        if want_logprobs:
            logprobs = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                targets if self.is_processed_logprobs_mode else raw_targets,
                bonus if self.is_processed_logprobs_mode else raw_bonus,
                output,
            )
        return SamplerOutput(sampled_token_ids=output, logprobs_tensors=logprobs)
