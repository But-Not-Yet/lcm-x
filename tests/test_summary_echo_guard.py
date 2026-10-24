"""修复甲 预注册判据（指令回显摘要守卫）。

背景：lcm.db 节点 316 的正文以摘要提示词自己的格式说明开头
（"<summary body ending with the required 'Expand for details about:' line>"），
是一代「无视输出契约」的典型残缺产物；它被写入 DAG 后进入每一轮装配。

判据：
  G1 判据函数 `_summary_looks_like_instruction_echo`：
     - 真实残缺形态（node 316 前缀 + 正常正文）→ True
     - 空串 / 纯空白 → True
     - 正常中文摘要 → False
     - 引擎自产合成文本（"Filtered replies derived from ignored messages."）→ False（不误伤）
     - 含 [Expand for details: ...] 尾标的正常摘要 → False（不误伤）
  G2 行为结构：compress 源码里「回显检测的拒绝」出现在 `self._dag.add_node(` 之前
     （拒绝必须发生在写入之前——不写节点、backlog 保留）。

RED 预期：`_summary_looks_like_instruction_echo` 尚不存在 → 收集期失败。
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.hermes/plugins"))
sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))

from hermes_lcm import compaction as _c  # noqa: E402


ECHO_SAMPLE = (
    "<summary body ending with the required 'Expand for details about:' line>\n"
    "\n## 当前任务与目标\n\n主线：验证 Qwen3-TTS 与 VoxCPM2 的替代关系。\n"
)

NORMAL_SAMPLE = (
    "## 当前任务与目标\n\n完成缓存装配漂移的定位与修复。\n\n"
    "[Expand for details: 探针、定罪、三项修复]"
)

SYNTHETIC_SAMPLE = (
    "Filtered replies derived from ignored messages.\n"
    "[Expand for details: ignored-dependent reply]"
)


def test_g1_real_echo_shape_is_detected():
    assert _c._summary_looks_like_instruction_echo(ECHO_SAMPLE) is True


def test_g1_empty_and_blank_are_rejected():
    assert _c._summary_looks_like_instruction_echo("") is True
    assert _c._summary_looks_like_instruction_echo("   \n\t  ") is True


def test_g1_normal_summary_passes():
    assert _c._summary_looks_like_instruction_echo(NORMAL_SAMPLE) is False


def test_g1_engine_synthetic_text_passes():
    assert _c._summary_looks_like_instruction_echo(SYNTHETIC_SAMPLE) is False


def test_g1_non_string_is_not_a_rejection():
    # 非字符串由调用侧的其他检查负责；判据函数本身不把它判成回显
    assert _c._summary_looks_like_instruction_echo(None) is False


def test_g2_rejection_precedes_node_write_in_compress_source():
    # 上游重构后写入被拆进 _publish_leaf_node，_compress_impl 文本里不再有 add_node。
    # 同一不变量在新结构下的三段等价锚点：守卫在 summarize 阶段就 break；
    # 调用方在 publish 之前先看到 break；写入只发生在 publish 里。
    guard_src = inspect.getsource(_c.CompactionMixin._summarize_selected_leaf)
    echo_pos = guard_src.index("_summary_looks_like_instruction_echo(")
    stop_pos = guard_src.index('state.sweep_stop_reason = "summary_result_rejected"', echo_pos)
    pass_pos = guard_src.index('return "next"', stop_pos)
    assert echo_pos < stop_pos < pass_pos, (
        "instruction-echo rejection must set the stop reason and break BEFORE the leaf passes on"
    )
    src = inspect.getsource(_c.CompactionMixin._compress_impl)
    summarize_pos = src.index("_summarize_selected_leaf(")
    # v0.28.1 re-anchor: upstream split publication into _publish_selected_leaf
    # (which calls _publish_leaf_node); the invariant is the same on the new call
    # graph, so the anchor follows the graph, not the old symbol name.
    publish_pos = src.index("self._publish_selected_leaf(", summarize_pos)
    assert summarize_pos < publish_pos, (
        "the summarize call (and its echo rejection) must precede the node publication step"
    )
    publish_src = inspect.getsource(_c.CompactionMixin._publish_leaf_node)
    assert "self._dag.add_node(" in publish_src, (
        "the node write must live in the publication step that the rejection short-circuits"
    )
