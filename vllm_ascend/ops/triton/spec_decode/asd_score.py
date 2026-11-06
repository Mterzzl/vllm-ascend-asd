# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility imports; shared scoring now lives in logit_score."""

from vllm_ascend.ops.triton.spec_decode.logit_score import (
    VOCAB_BLOCK as VOCAB_BLOCK,
)
from vllm_ascend.ops.triton.spec_decode.logit_score import (
    argmax_logits as argmax_logits,
)
from vllm_ascend.ops.triton.spec_decode.logit_score import (
    score_logits as score_logits,
)
