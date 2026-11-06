# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ASD configuration and request identity; numerical operations live in Triton."""

import ast
import math
import struct
from dataclasses import dataclass

import numpy as np
import torch
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

from vllm_ascend import envs
from vllm_ascend.ops.triton.spec_decode.asd import grow_ledger


def validate_speculative_config(spec_config):
    """Validate the verifier's data contract, independently of the drafter name."""
    if spec_config is None:
        raise ValueError("ASD requires speculative decoding to be configured")
    mode = getattr(spec_config, "rejection_sample_method", "strict")
    if mode != "strict":
        raise ValueError(f"ASD requires strict/greedy verification, not {mode!r}")
    tree = getattr(spec_config, "speculative_token_tree", None)
    if tree is None:
        return
    try:
        paths = ast.literal_eval(tree) if isinstance(tree, str) else tree
        if not isinstance(paths, (list, tuple)) or not paths:
            raise ValueError
        if not all(
            isinstance(path, (list, tuple))
            and all(isinstance(node, int) and not isinstance(node, bool) and node >= 0 for node in path)
            for path in paths
        ):
            raise ValueError
        paths = sorted((tuple(path) for path in paths), key=len)
        previous = ()
        for depth, path in enumerate(paths, 1):
            if len(path) != depth or path[:-1] != previous:
                raise ValueError
            previous = path
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError("ASD accepts a single contiguous draft chain, not a branching token tree") from exc


@dataclass(frozen=True)
class ASDConfig:
    budget: float = 8.0
    gate: float = 0.25
    mismatch_cap: int = 2

    @classmethod
    def from_env(cls):
        def scalar(name):
            try:
                value = float(getattr(envs, name))
                represented = struct.unpack("f", struct.pack("f", value))[0]
            except (ValueError, TypeError, OverflowError, struct.error) as exc:
                raise ValueError(f"{name} must be finite, nonnegative and representable in FP32") from exc
            if not math.isfinite(value) or value < 0 or not math.isfinite(represented):
                raise ValueError(f"{name} must be finite, nonnegative and representable in FP32")
            if value > 0 and represented == 0:
                raise ValueError(f"{name} underflows in FP32")
            return value

        cap = str(envs.VLLM_ASCEND_ASD_MAX_RELAXED_TOKENS_PER_BLOCK).strip()
        if not cap.isdecimal():
            raise ValueError("VLLM_ASCEND_ASD_MAX_RELAXED_TOKENS_PER_BLOCK must be a nonnegative integer")
        return cls(scalar("VLLM_ASCEND_ASD_RISK_BUDGET"), scalar("VLLM_ASCEND_ASD_MAX_REGRET_PER_VALUE"), int(cap))


class RequestLedger:
    """Stable identities across row reorder and preemption; reset on ID reuse.

    Python tracks identities, not budget values. Device budget copying and
    updates use Triton. The upstream helper only transfers integer metadata.
    """

    def __init__(self, capacity, device):
        self.device = device
        self.values = torch.empty((max(1, capacity),), dtype=torch.float32, device=device)
        self.slots = {}
        self.free = list(reversed(range(self.values.numel())))

    def prepare(self, request_ids, committed=None):
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("ASD batch has duplicate request IDs")
        if committed is not None and len(committed) != len(request_ids):
            raise ValueError("ASD commit mask length differs from request IDs")
        rows = []
        for row, request_id in enumerate(request_ids):
            fresh = request_id not in self.slots
            if fresh:
                if not self.free:
                    old = self.values.numel()
                    self.values = grow_ledger(self.values, old * 2)
                    self.free.extend(reversed(range(old, old * 2)))
                self.slots[request_id] = self.free.pop()
            commit = True if committed is None else bool(committed[row])
            rows.append((self.slots[request_id], int(fresh), int(commit)))
        metadata = np.asarray(rows, dtype=np.int32).reshape((-1, 3))
        return async_copy_to_gpu(metadata, device=self.device)

    def drop_finished(self, request_ids):
        for request_id in request_ids:
            slot = self.slots.pop(request_id, None)
            if slot is not None:
                self.free.append(slot)
