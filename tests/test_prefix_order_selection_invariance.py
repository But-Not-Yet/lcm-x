"""排序只改渲染顺序，不许改预算选择的结论。

把深度优先（旧）与全局时间序（新）两种 append 顺序喂给 select_prefix，在同一预算下
选中的节点集合必须完全相同——否则修漂移会顺手改掉上下文预算的取舍，那是另一场事故。
"""

from hermes_lcm.assembly_plan import PrefixBudgetFacts, select_prefix
from hermes_lcm.dag import SummaryNode


def _facts(parts, part_keys, budget):
    return PrefixBudgetFacts(
        part_keys=part_keys,
        summary_role="user",
        summary_budget=budget,
        assembly_cap=None,
        survival_fit=False,
        count_message_tokens=lambda msg: len(msg["content"]) // 4 + 1,
    )


def _kept_ids(parts, part_keys, budget):
    selection = select_prefix(parts, _facts(parts, part_keys, budget))
    return {part_keys[i][1] for i in selection.kept_indexes}


def test_selection_set_is_identical_under_both_orders():
    """集合必须相同；列表顺序允许不同，因为渲染顺序正是本次要改的东西。"""
    nodes = [
        SummaryNode(node_id=11, depth=1, created_at=300.0, summary="arc newest " * 40),
        SummaryNode(node_id=12, depth=0, created_at=100.0, summary="leaf oldest " * 40),
        SummaryNode(node_id=13, depth=0, created_at=200.0, summary="leaf middle " * 40),
    ]
    parts = {n.node_id: f"[Summary] {n.summary}" for n in nodes}

    # 旧顺序：深度优先（d1 先，d0 后）
    old_order = [nodes[0], nodes[1], nodes[2]]
    old_keys = [(0, 11), (1, 12), (1, 13)]
    # 新顺序：全局时间序
    new_order = [nodes[1], nodes[2], nodes[0]]
    new_keys = [(1, 12), (1, 13), (0, 11)]

    for budget in (None, 400, 900, 2000):
        old_parts = [parts[n.node_id] for n in old_order]
        new_parts = [parts[n.node_id] for n in new_order]
        assert _kept_ids(old_parts, old_keys, budget) == _kept_ids(new_parts, new_keys, budget), budget
