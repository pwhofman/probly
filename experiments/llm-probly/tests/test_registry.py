from __future__ import annotations

import json

from common.registry import (
    ACCESS_LEVELS,
    FAMILIES,
    REGISTRY,
    MethodSpec,
    candidates,
    check_registry,
    family_of,
    get,
    load_plugin,
    plugin_names,
)
import pytest


def test_registry_and_plugin_files_agree():
    assert check_registry() == []


def test_every_plugin_file_is_registered():
    assert set(plugin_names()) == set(REGISTRY)


def test_values_match_the_d4_profile_schema():
    # D4 (selector/profile.py) accepts exactly these values.
    assert FAMILIES == ("agreement", "self_eval", "baseline", "hybrid")
    assert ACCESS_LEVELS == ("text_only", "logprobs", "white_box")


def test_greedy_only_methods_cost_one_and_sample_methods_eleven():
    assert get("msp").cost == 1
    assert get("semantic_entropy").cost == 11
    assert get("lexical_similarity").cost == 11
    assert get("p_true_zeroshot").cost == 2


def test_methods_that_need_the_model_are_self_evaluation():
    assert {s.name for s in REGISTRY.values() if s.needs_model} == {
        "p_true_zeroshot",
        "p_true_brainstorm",
        "p_true_empirical",
        "verbalized_2s",
    }


def test_text_only_methods_match_f2():
    # F2 runs exactly these through an OpenAI-compatible API (plus the random reference).
    text_only = {s.name for s in candidates(access="text_only")}
    assert text_only == {
        "lexical_similarity",
        "discrete_semantic_entropy",
        "verbalized_2s",
        "p_true_empirical",
        "dummy_random",
    }


def test_more_access_allows_more_methods():
    sizes = [len(candidates(access=a)) for a in ACCESS_LEVELS]
    assert sizes == sorted(sizes)
    assert sizes[-1] == len(REGISTRY)


def test_candidates_respects_cost_and_sorts_cheapest_first():
    found = candidates(max_cost=2)
    assert found
    assert all(s.cost <= 2 for s in found)
    assert [s.cost for s in found] == sorted(s.cost for s in found)


def test_candidates_filters_family():
    assert {s.family for s in candidates(family="agreement")} == {"agreement"}


def test_candidates_rejects_unknown_access():
    with pytest.raises(ValueError, match="access must be one of"):
        candidates(access="api")


def test_get_unknown_method_lists_known_names():
    with pytest.raises(KeyError, match="semantic_entropy"):
        get("semantic_entropi")


def test_family_of_unregistered_week1_result_is_empty():
    assert family_of("p_true_zeroshot_norm") == ""
    assert family_of("msp") == "baseline"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"family": "agreeement"}, "family"),
        ({"access": "api"}, "access"),
        ({"cost": -1}, "cost"),
        ({"cost": float("nan")}, "cost"),
    ],
)
def test_spec_rejects_bad_values(kwargs, match):
    fields = {"name": "x", "family": "baseline", "access": "text_only", "cost": 1} | kwargs
    with pytest.raises(ValueError, match=match):
        MethodSpec(**fields)


def test_to_dict_is_json_serializable():
    data = json.loads(json.dumps(get("semantic_entropy").to_dict()))
    assert data["access"] == "logprobs"
    assert data["uses_nli"] is True


def test_load_plugin_returns_module_with_score():
    module = load_plugin("msp")
    assert module.NAME == "msp"
    assert callable(module.score)


def test_probability_flag_matches_self_evaluation():
    assert {s.name for s in REGISTRY.values() if s.probability} == {
        s.name for s in REGISTRY.values() if s.family == "self_eval"
    }
