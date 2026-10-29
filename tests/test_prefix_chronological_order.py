"""摘要前缀的渲染顺序：必须全局按时间，不许按深度分批。

改前引擎按「深度降序、层内旧→新」拼接 frontier，d1 归档块会排到更早的 d0 叶块前面，
模型读到的时间线在每个深度边界倒流一次（实测 56 个可判会话里 9 个、共 46 次）。
这个族属于静默损坏：不报错、不丢内容，只把时间线拧乱，靠人眼很难发现。
"""

import hermes_lcm.engine as lcm_engine
from hermes_lcm.dag import SummaryNode


def test_frontier_renders_chronologically_not_depth_major():
    newer_arc = SummaryNode(node_id=714, depth=1, created_at=300.0)
    older_leaf_a = SummaryNode(node_id=712, depth=0, created_at=100.0)
    older_leaf_b = SummaryNode(node_id=713, depth=0, created_at=200.0)
    ordered = lcm_engine._chronological_frontier(
        [(0, newer_arc), (1, older_leaf_a), (1, older_leaf_b)]
    )
    # 时间序：712 → 713 → 714
    assert [node.node_id for _, node in ordered] == [712, 713, 714]
    # 深度优先级（group）跟着节点走，没有被排序破坏
    assert [group for group, _ in ordered] == [1, 1, 0]


def test_frontier_order_is_stable_for_missing_timestamps():
    zero = SummaryNode(node_id=9, depth=0, created_at=0.0)
    real = SummaryNode(node_id=3, depth=1, created_at=50.0)
    ordered = lcm_engine._chronological_frontier([(0, real), (1, zero)])
    assert [node.node_id for _, node in ordered] == [9, 3]


def test_render_adds_time_span_from_dag_columns():
    """老节点正文里没有时间锚，渲染层从 earliest_at/latest_at 补上。"""
    import datetime as _dt
    start = _dt.datetime(2026, 10, 28, 14, 44).timestamp()
    end = _dt.datetime(2026, 10, 28, 16, 16).timestamp()
    node = SummaryNode(node_id=714, depth=1, created_at=end,
                       earliest_at=start, latest_at=end, summary="正文", expand_hint="x")
    text = lcm_engine._summary_part_text(node)
    assert text.startswith("[Session Arc Summary (d1, node 714)] 2026-10-28 14:44–16:16")
    # 标记本身仍可被既有解析器匹配
    import re
    assert re.match(r"\[Session Arc Summary \(d1, node 714\)\]", text)


def test_render_leaves_header_alone_without_timestamps():
    node = SummaryNode(node_id=5, depth=0, created_at=0.0, summary="正文", expand_hint="x")
    text = lcm_engine._summary_part_text(node)
    assert text.startswith("[Recent Summary (d0, node 5)]\n")
