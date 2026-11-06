# SPDX-License-Identifier: Apache-2.0
"""CPU-only dependency and single-chain contract checks; no model/device runs."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ASCEND = Path(__file__).resolve().parents[3] / "vllm_ascend"


def validator():
    # Extract the actual policy validator without importing tensor runtimes.
    path = ASCEND / "sample/asd.py"
    node = next(
        item
        for item in ast.parse(path.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "validate_speculative_config"
    )
    namespace = {"ast": ast}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize("method", ["dflash", "eagle3", "ngram", "draft_model", "unrelated_drafter"])
@pytest.mark.parametrize("tree", [None, "[(0,), (0, 0), (0, 0, 0)]", [(2, 3), (2,)]])
def test_chain_contract_is_independent_of_method_name(method, tree):
    validator()(SimpleNamespace(method=method, rejection_sample_method="strict", speculative_token_tree=tree))


@pytest.mark.parametrize("tree", ["[(0,), (1,)]", "[(0,), (0, 0), (0, 1)]", "[(0, 0)]", "[]", "broken", "[(-1,)]"])
def test_branching_or_invalid_layout_rejected(tree):
    with pytest.raises(ValueError, match="single contiguous draft chain"):
        validator()(SimpleNamespace(rejection_sample_method="strict", speculative_token_tree=tree))


@pytest.mark.parametrize("mode", ["synthetic", "probabilistic"])
def test_non_greedy_verification_contract_rejected(mode):
    with pytest.raises(ValueError, match="strict/greedy"):
        validator()(SimpleNamespace(rejection_sample_method=mode))


def test_speculation_required():
    with pytest.raises(ValueError, match="requires speculative"):
        validator()(None)


def imports(path):
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


@pytest.mark.parametrize("producer", ["dflash.py", "eagle/speculator.py", "eagle/aclgraph.py"])
def test_producer_does_not_import_asd(producer):
    assert not any(
        "asd" in name.split(".") or "asd_score" in name.split(".")
        for name in imports(ASCEND / "worker/v2/spec_decode" / producer)
    )


@pytest.mark.parametrize(
    "policy",
    [
        "sample/asd.py",
        "sample/asd_rejection_sampler.py",
        "worker/v2/spec_decode/asd.py",
        "ops/triton/spec_decode/asd.py",
    ],
)
def test_asd_does_not_import_a_producer(policy):
    assert not any({"dflash", "eagle", "eagle3"}.intersection(name.split(".")) for name in imports(ASCEND / policy))


def test_shared_score_kernel_body_unchanged():
    before = Path(__file__).parent / "fixtures/asd_score_pre_decoupling.py"
    after = ASCEND / "ops/triton/spec_decode/logit_score.py"

    def functions(path):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }

    assert functions(before) == functions(after)
    assert not any("asd" in name or "worker.v2.spec_decode" in name for name in imports(after))


def test_shared_scorer_cannot_import_acceptance_policy():
    source = (ASCEND / "ops/triton/spec_decode/logit_score.py").read_text()
    assert "RequestLedger" not in source and "ASDConfig" not in source
