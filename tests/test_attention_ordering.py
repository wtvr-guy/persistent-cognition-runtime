import pytest

from prometheist.attention_ordering import AttentionWorkItem, deterministic_dependency_order


def test_attention_ordering_is_topological_priority_stable_and_input_order_independent():
    items = [
        AttentionWorkItem(item_id="reconcile", priority=5, dependency_ids=["collect"]),
        AttentionWorkItem(item_id="independent", priority=20),
        AttentionWorkItem(item_id="collect", priority=10),
        AttentionWorkItem(item_id="zeta", priority=20),
        AttentionWorkItem(item_id="alpha", priority=20),
    ]

    expected = ["collect", "reconcile", "alpha", "independent", "zeta"]
    assert deterministic_dependency_order(items) == expected
    assert deterministic_dependency_order(list(reversed(items))) == expected


def test_attention_ordering_rejects_missing_dependencies():
    with pytest.raises(ValueError, match="missing dependencies"):
        deterministic_dependency_order(
            [AttentionWorkItem(item_id="dependent", dependency_ids=["missing"])]
        )


def test_attention_ordering_rejects_cycles():
    with pytest.raises(ValueError, match="cycle"):
        deterministic_dependency_order(
            [
                AttentionWorkItem(item_id="alpha", dependency_ids=["beta"]),
                AttentionWorkItem(item_id="beta", dependency_ids=["alpha"]),
            ]
        )
