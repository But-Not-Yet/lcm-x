"""Golden observations of real compaction; regenerate with this module's CLI.

Only the already-validated provider boundary, clock and offline token estimator
are replaced. SQL tracing starts after scenario setup and includes ingestion,
publication, rollback and the next assembly. Consecutive identical callbacks
use a repeat count, preserving their order and multiplicity. SQL shapes are
defined once at first use; subsequent records reference that shape and hash
the full canonical bound statement.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib
import inspect
import json
import re
import sqlite3
import tempfile
import textwrap
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# The plugin is directory-loaded by Hermes, not installed as a Python package.
# Reuse the suite's registration when invoked with python -m outside pytest.
import tests.conftest  # noqa: F401
from tests.harness import engine_context, pin_char_counter

compaction: Any = importlib.import_module("hermes_lcm.compaction")
escalation: Any = importlib.import_module("hermes_lcm.escalation")
dag: Any = importlib.import_module("hermes_lcm.dag")
tokens: Any = importlib.import_module("hermes_lcm.tokens")
harness: Any = importlib.import_module("tests.harness")
GOLDEN = Path(__file__).parents[1] / "fixtures" / "compress_trace.json"
Trace = list[str]
PAD = " alpha beta gamma delta epsilon" * 40


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    mode: str = "healthy"
    shape: str = "turns"
    turns: int = 8
    sweep: bool = False
    force: bool = True
    current_tokens: int | None = 3000
    overrides: tuple[tuple[str, Any], ...] = ()
    request: str = ""
    seeded_depths: tuple[int, ...] = ()
    retry: bool = False
    focus: str | None = None
    setup: str = ""
    expiry_checkpoint: int = 0


SCENARIOS = (
    Scenario("empty", shape="empty"),
    Scenario("fresh-tail-protected", turns=1, force=False),
    Scenario("below-leaf-minimum", overrides=(("leaf_chunk_tokens", 20000),)),
    Scenario("leaf-pass"),
    Scenario("leaf-no-host-token-count", current_tokens=None),
    Scenario("leaf-automatic-no-sweep", force=False),
    Scenario("dynamic-leaf-passes", overrides=(("dynamic_leaf_chunk_enabled", True),
                                               ("dynamic_leaf_chunk_max", 800))),
    Scenario("dynamic-below-minimum", overrides=(("dynamic_leaf_chunk_enabled", True),
                                                ("leaf_chunk_tokens", 20000))),
    Scenario("condensation-levels", seeded_depths=(0, 0, 1),
             overrides=(("condensation_fanin", 2),)),
    Scenario("sweep-drain-condense", sweep=True, force=False),
    Scenario("sweep-pre-leaf-condensation", sweep=True, force=False, seeded_depths=(0, 0, 0),
             overrides=(("summary_prefix_target_tokens", 100),)),
    Scenario("escalation-level-two", mode="reject-first"),
    Scenario("escalation-level-three", mode="outage", overrides=(("survival_fit", False),)),
    Scenario("escalation-rejected-keeps-backlog", mode="outage"),
    Scenario("context-length-rescue", mode="context-refusal"),
    Scenario("budget-exhausted-before-leaf", mode="timeout", sweep=True, force=False),
    Scenario("budget-partial-pass", mode="slow", sweep=True, force=False),
    Scenario("provider-aborted", mode="cancel-first"),
    Scenario("resume-after-interruption", mode="cancel-second", sweep=True, retry=True),
    Scenario("resume-unadopted-commit", retry=True),
    Scenario("tool-pair-groups", shape="tools"),
    Scenario("objective-anchor", shape="objective", overrides=(("fresh_tail_count", 8),)),
    Scenario("identity-anchor", shape="identity"),
    Scenario("externalized-payloads", shape="tools", overrides=(
        ("large_output_externalization_enabled", True),
        ("large_output_externalization_threshold_chars", 100),
        ("large_output_active_replay_stubbing_enabled", True),
        ("large_output_active_replay_stub_threshold_tokens", 100))),
    Scenario("lifecycle-publication-conflict", mode="conflict"),
    Scenario("child-session-bypass", shape="child", force=False),
    Scenario("child-session-forced", shape="child"),
    Scenario("boundary-cleanup-only", request="cleanup_only_due_to_boundary_cooldown"),
    Scenario("below-threshold-cleanup-only", force=False, current_tokens=1,
             request="below_threshold_cleanup_only"),
    Scenario("hold-fit-only", force=False, request="hold_fit_only_requested"),
    Scenario("ignored-message-dependents", shape="ignored", overrides=(
        ("ignore_message_patterns", ["IGNORE THIS ROW"]),)),
    Scenario("forced-overflow", current_tokens=250000,
             overrides=(("max_assembly_tokens", 2000),)),
    Scenario("hidden-owned-backlog", shape="hidden"),
    Scenario("route-unavailable", mode="open-circuit"),
    Scenario("route-unavailable-full-tail", mode="open-circuit", turns=1),
    Scenario("route-unavailable-adopt-commit", mode="open-after-first", retry=True),
    Scenario("sweep-tail-drained", turns=1, sweep=True, force=False),
    Scenario("fresh-tail-yield", force=False, overrides=(("fresh_tail_count", 100),
             ("fresh_tail_pressure_yield_min_observations", 1))),
    Scenario("dynamic-fresh-tail-yield", force=False, overrides=(("fresh_tail_count", 100),
             ("fresh_tail_pressure_yield_min_observations", 1), ("dynamic_leaf_chunk_enabled", True))),
    Scenario("sweep-fresh-tail-yield", force=False, sweep=True, overrides=(("fresh_tail_count", 100),
             ("fresh_tail_pressure_yield_min_observations", 1))),
    Scenario("dynamic-threshold-reached", current_tokens=250, overrides=(("dynamic_leaf_chunk_enabled", True),)),
    Scenario("dynamic-raw-drained", turns=3, overrides=(("dynamic_leaf_chunk_enabled", True),)),
    Scenario("sweep-budget-entry", mode="spent", sweep=True, force=False),
    Scenario("sweep-pass-budget", turns=40, sweep=True, overrides=(("leaf_chunk_tokens", 1),)),
    Scenario("pre-leaf-condensation-rejected", mode="outage", sweep=True,
             seeded_depths=(0, 0, 0), overrides=(("summary_prefix_target_tokens", 100),)),
    Scenario("post-leaf-condensation-rejected", mode="outage-after-leaves", sweep=True,
             overrides=(("summary_prefix_target_tokens", 20),)),
    Scenario("post-leaf-publication-conflict", mode="conflict-second", sweep=True),
    Scenario("condensation-publication-locked", mode="lock-condensation", seeded_depths=(0, 0, 1),
             overrides=(("condensation_fanin", 2),)),
    Scenario("pre-leaf-publication-locked", mode="lock-condensation", sweep=True,
             seeded_depths=(0, 0, 0), overrides=(("summary_prefix_target_tokens", 100),)),
    Scenario("deferred-maintenance", setup="debt", overrides=(("deferred_maintenance_enabled", True),)),
    Scenario("deferred-critical-small-backlog", setup="debt", current_tokens=190000, overrides=(
        ("deferred_maintenance_enabled", True), ("critical_budget_pressure_ratio", 0.5),
        ("leaf_chunk_tokens", 20000), ("dynamic_leaf_chunk_enabled", True))),
    Scenario("identity-anchor-disabled", setup="no-identity"),
    Scenario("explicit-focus", focus="Preserve source coverage."),
    Scenario("no-system-prompt", shape="no-system"),
    Scenario("retained-tail-occurrence", shape="retained", overrides=(("fresh_tail_count", 1),
             ("leaf_chunk_tokens", 1)), retry=True),
    Scenario("replayed-summary-scaffold", setup="replay", retry=True),
    Scenario("stub-first-exit", shape="tools", force=False, current_tokens=12001, overrides=(
        ("context_threshold", 0.06), ("large_output_externalization_enabled", True),
        ("large_output_active_replay_stubbing_enabled", True),
        ("large_output_externalization_threshold_chars", 100),
        ("large_output_active_replay_stub_threshold_tokens", 100))),
    Scenario("legacy-filtered-backlog", shape="legacy-filter", setup="stored", overrides=(
        ("ignore_message_patterns", ["IGNORE THIS ROW"]),)),
    Scenario("legacy-filtered-tail-dependents", shape="legacy-tail", setup="stored", overrides=(
        ("fresh_tail_count", 3), ("ignore_message_patterns", ["IGNORE THIS ROW"]))),
    Scenario("legacy-filtered-eof-dependents", shape="legacy-eof", setup="stored", overrides=(
        ("ignore_message_patterns", ["IGNORE THIS ROW"]),)),
    Scenario("externalized-transcript-gc", shape="tools", overrides=(
        ("dynamic_leaf_chunk_max", 20000), ("large_output_externalization_enabled", True),
        ("large_output_externalization_threshold_chars", 100), ("large_output_transcript_gc_enabled", True))),
    Scenario("transcript-gc-locked", shape="tools", mode="lock-gc", overrides=(
        ("dynamic_leaf_chunk_max", 20000), ("large_output_externalization_enabled", True),
        ("large_output_externalization_threshold_chars", 100), ("large_output_transcript_gc_enabled", True))),
    Scenario("default-sweep-target", sweep=True, overrides=(("summary_prefix_target_tokens", 0),)),
    Scenario("child-without-usage", shape="child", current_tokens=None),
    Scenario("child-with-cached-usage", shape="child", current_tokens=None, setup="child-usage"),
    Scenario("generated-dependent-replies", setup="dependents"),
    Scenario("dependent-only-leaf", shape="dependent-only", setup="dependents", overrides=(("leaf_chunk_tokens", 1),)),
    Scenario("ignored-dependent-tail-eof", shape="dependent-eof", setup="stored", overrides=(
        ("fresh_tail_count", 2), ("ignore_message_patterns", ["IGNORE THIS ROW"]))),
    Scenario("dynamic-below-minimum-no-fit", force=False, overrides=(
        ("dynamic_leaf_chunk_enabled", True), ("leaf_chunk_tokens", 20000), ("survival_fit", False))),
    Scenario("single-below-minimum-no-fit", force=False, overrides=(
        ("leaf_chunk_tokens", 20000), ("survival_fit", False))),
    *[Scenario(f"deadline-checkpoint-{point}", sweep=True, force=False, expiry_checkpoint=point)
      for point in (4, 6, 8, 10)],
    Scenario("deadline-hidden-backlog-read", shape="hidden", sweep=True, force=False, expiry_checkpoint=6),
    Scenario("unthresholded-default-target", overrides=(("context_threshold", 0.0),
             ("summary_prefix_target_tokens", 0))),
    Scenario("sweep-stub-first-exit", shape="tools", sweep=True, force=False, current_tokens=12001, overrides=(
        ("context_threshold", 0.06), ("large_output_externalization_enabled", True),
        ("large_output_active_replay_stubbing_enabled", True),
        ("large_output_externalization_threshold_chars", 100),
        ("large_output_active_replay_stub_threshold_tokens", 100))),
    Scenario("condensation-spends-soft-budget", mode="very-slow", sweep=True, force=False,
             seeded_depths=(0, 0, 0), overrides=(("summary_prefix_target_tokens", 100),)),
    Scenario("hung-primary-leaves-fallback", mode="timeout", sweep=True, force=False,
             overrides=(("summary_fallback_models", ["trace-fallback"]),)),
    Scenario("dynamic-forced-overflow", current_tokens=250000, overrides=(
        ("max_assembly_tokens", 2000), ("dynamic_leaf_chunk_enabled", True))),
    Scenario("dynamic-small-backlog-yields", force=False, overrides=(
        ("dynamic_leaf_chunk_enabled", True), ("leaf_chunk_tokens", 20000),
        ("survival_fit", False), ("fresh_tail_pressure_yield_min_observations", 1))),
    Scenario("single-small-backlog-yields", force=False, overrides=(
        ("leaf_chunk_tokens", 20000), ("survival_fit", False), ("fresh_tail_pressure_yield_min_observations", 1))),
    Scenario("objective-cap-too-small", shape="objective", overrides=(
        ("fresh_tail_count", 8), ("max_assembly_tokens", 10))),
    Scenario("objective-without-cap", shape="objective", overrides=(
        ("fresh_tail_count", 8), ("max_assembly_tokens", 0))),
    Scenario("optional-extraction-host-unavailable", overrides=(
        ("extraction_enabled", True), ("dynamic_leaf_chunk_enabled", True),
        ("assertion_extraction_enabled", True))),
)


@dataclasses.dataclass
class Clock:
    now: float = 1000.0
    nonce: int = 0
    expiry_checkpoint: int = 0
    checkpoints: int = 0

    def monotonic(self) -> float:
        if self.expiry_checkpoint:
            frame = inspect.currentframe()
            assert frame is not None and frame.f_back is not None
            if frame.f_back.f_code.co_name in {"_compress_impl", "sweep_step_done"}:
                self.checkpoints += 1
                if self.checkpoints == self.expiry_checkpoint:
                    self.now += 121.0
            del frame
        return self.now

    def wall(self) -> float:
        return 1700000000.0 + self.now

    def time_ns(self) -> int:
        self.nonce += 1
        return 1700000000000000000 + self.nonce


class HostInterrupted(BaseException):
    """The host's cancellation is outside the provider's Exception recovery."""


class LiteralPattern:
    """The configured pattern interface without the optional regex dependency."""

    pattern = "IGNORE THIS ROW"

    def search(self, text: str, timeout: float | None = None) -> bool:
        return self.pattern in text


def self_attributes() -> list[str]:
    """Derive the inventory, including getattr(self, ...) and nested closures."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(compaction.CompactionMixin._compress_impl)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            names.add(node.attr)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"getattr", "setattr", "hasattr"} and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name) and node.args[0].id == "self"
                and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            names.add(node.args[1].value)
    return sorted(names)


class Canonical:
    """Preserve semantic values; name only generated IDs and process addresses."""

    def __init__(self, root: Path) -> None:
        self.root = str(root)
        self.ids: dict[str, str] = {}
        self.excluded: dict[str, str] = {}

    def text(self, value: str) -> str:
        value = value.replace(self.root, "<sandbox>")

        def replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            return self.ids.setdefault(raw, f"<generated-{len(self.ids) + 1}>")

        return re.sub(r"(?<![a-zA-Z0-9])[0-9a-f]{32}(?![a-zA-Z0-9])|"
                      r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", replace, value)

    def value(self, value: Any, path: str = "") -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, (str, Path)):
            return self.text(str(value))
        if isinstance(value, bytes):
            return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
        if callable(value):
            return {"callable": value.__qualname__}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {f.name: self.value(getattr(value, f.name), f"{path}.{f.name}")
                    for f in dataclasses.fields(value)}
        if isinstance(value, (escalation.ForegroundBudget, escalation.ForegroundEstimates)):
            return self.value(vars(value), path)
        if isinstance(value, dict):
            # These keys are Python object identities, not durable store IDs.
            identity_keys = path.endswith(("store_ids_by_message_id", "_compress_occurrences"))
            return {(f"<message-{i}>" if identity_keys else str(k)): self.value(v, f"{path}.{k}")
                    for i, (k, v) in enumerate(value.items(), 1)}
        if isinstance(value, (list, tuple)):
            return [self.value(v, path) for v in value]
        if isinstance(value, (set, frozenset)):
            return sorted((self.value(v, path) for v in value), key=lambda x: json.dumps(x, sort_keys=True))
        self.excluded[path] = type(value).__name__ + ": no stable value (connection/lock/service)"
        return {"excluded_type": type(value).__name__}


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(text: str) -> str:
    return f"{hashlib.sha256(text.encode()).hexdigest()[:12]}/{len(text)}"


def compact(value: Any) -> str:
    text = encoded(value)
    return text if len(text) <= 80 else f"sha12/len={fingerprint(text)}"


def sql_event(connection: str, sql: str, shapes: dict[tuple[str, str], int], events: Trace) -> str:
    if sql in {"BEGIN IMMEDIATE", "BEGIN", "COMMIT", "ROLLBACK"}:
        return "sql " + sql
    # Internal FTS callbacks already contain placeholders; quoted names there
    # are identifiers. Other callbacks contain expanded values, which must not
    # be embedded in the golden. Keep every column and expression in the shape.
    shape = sql if sql.startswith("-- ") else re.sub(
        r"'(?:''|[^'])*'|(?<![\w])\d+(?:\.\d+)?(?![\w])", "?", sql)
    key = (connection, shape)
    if key not in shapes:
        shapes[key] = len(shapes) + 1
        events.append(f"sql shape={shapes[key]} conn={connection} {shape}")
    reference = f"sql #{shapes[key]}"
    return reference if shape == sql else f"{reference} sha12/len={fingerprint(sql)}"


def append_sql(events: Trace, event: str) -> None:
    if events:
        previous = events[-1]
        base, separator, count = previous.rpartition(" x")
        if previous == event:
            events[-1] = event + " x2"
            return
        if separator and count.isdigit() and base == event:
            events[-1] = event + f" x{int(count) + 1}"
            return
    events.append(event)


def transcript(case: Scenario) -> list[dict[str, Any]]:
    if case.shape == "empty":
        return []
    rows: list[dict[str, Any]] = [{"role": "system", "content": "Preserve the user's work."}]
    for i in range(case.turns):
        rows.extend([{"role": "user", "content": f"Turn {i}: inspect the source.{PAD}", "timestamp": 10.0 * (i + 1)},
                     {"role": "assistant", "content": f"Turn {i}: recorded evidence.{PAD}"}])
    if case.shape in {"tools", "objective"}:
        if case.shape == "objective":
            rows = [{"role": "user", "content": "Check the named files and preserve every source."}]
        rows.extend(harness.sequential_pairs(4, 1600)[1:])
    if case.shape == "identity":
        rows[1]["content"] = "My preferred name is Reference. Preserve this identity. " + PAD
    if case.shape == "ignored":
        rows[3]["content"] = "IGNORE THIS ROW " + PAD
    if case.shape.startswith("legacy-"):
        index = 3 if case.shape == "legacy-filter" else len(rows) - (5 if case.shape == "legacy-tail" else 3)
        rows[index]["content"] = "IGNORE THIS ROW " + PAD
    if case.shape in {"dependent-only", "dependent-eof"}:
        rows = [{"role": "system", "content": "Preserve the user's work."},
                {"role": "user", "content": "IGNORE THIS ROW " + PAD},
                {"role": "assistant", "content": "Dependent reply. " + PAD},
                {"role": "assistant", "content": "Dependent continuation. " + PAD}]
        if case.shape == "dependent-only":
            rows.extend([{"role": "user", "content": "A new request."},
                         {"role": "assistant", "content": "A fresh reply."}])
    if case.shape in {"no-system", "retained"}:
        rows = rows[1:]
    return rows


class Provider:
    def __init__(self, case: Scenario, clock: Clock, events: Trace, canonical: Canonical, engine: Any) -> None:
        self.case, self.clock, self.events, self.canonical, self.engine = case, clock, events, canonical, engine
        self.calls = 0

    def __call__(self, prompt: str, max_tokens: int, model: str = "", timeout: float | None = None,
                 reasoning_effort: str = "") -> str | None:
        self.calls += 1
        # Prompt headings distinguish L1/L2 and condensation depth; no internal spy.
        level = 2 if "bullet" in prompt.lower().split("\n\n", 1)[0] else 1
        self.events.append(f"call {self.calls} model={model} level={level} budget={max_tokens} "
                           f"timeout={timeout} in={fingerprint(self.canonical.text(prompt))}")
        mode = self.case.mode
        if mode == "cancel-first" or (mode == "cancel-second" and self.calls == 2):
            raise HostInterrupted()
        if mode == "timeout":
            self.clock.now += float(timeout or 120)
            raise TimeoutError("provider timed out")
        if mode in {"slow", "very-slow"}:
            self.clock.now += 29.0 if mode == "slow" else 61.0
        if (mode == "conflict" and self.calls == 1) or (mode == "conflict-second" and self.calls == 2):
            # A competing owner changes a selected row's ownership before
            # publication. Real lifecycle coverage validation must reject it.
            self.engine._store._conn.execute(
                "UPDATE messages SET conversation_id = 'other-conversation' WHERE store_id = ?",
                (2 if self.calls == 1 else 6,))
            self.engine._store._conn.commit()
        if mode == "lock-condensation" and ((self.case.sweep and self.calls == 1)
                                            or (not self.case.sweep and self.calls == 2)):
            self.engine._dag._conn.execute("""CREATE TEMP TRIGGER lock_summary BEFORE INSERT ON summary_nodes
                BEGIN SELECT RAISE(ABORT, 'database is locked'); END""")
        if mode == "lock-gc" and self.calls == 1:
            self.engine._store._conn.execute("""CREATE TEMP TRIGGER lock_gc BEFORE UPDATE OF content ON messages
                BEGIN SELECT RAISE(ABORT, 'database is locked'); END""")
        if mode == "context-refusal" and self.calls <= 2:
            raise RuntimeError("maximum context length exceeded")
        if mode == "outage" or (mode == "outage-after-leaves" and self.calls >= 8):
            return None
        if mode == "reject-first" and self.calls == 1:
            return "not shorter " * 4000
        return f"Earlier turns, call {self.calls}.\nExpand for details about: turns"


def install_clock(patch: pytest.MonkeyPatch, clock: Clock) -> None:
    replacement = SimpleNamespace(**{name: getattr(time, name) for name in dir(time) if not name.startswith("__")})
    replacement.monotonic, replacement.perf_counter, replacement.time = clock.monotonic, clock.monotonic, clock.wall
    replacement.time_ns = clock.time_ns
    replacement.gmtime = lambda: time.gmtime(1700000000)
    for name in ("compaction", "engine", "escalation", "store", "dag", "lifecycle_state",
                 "survival_fit", "externalize"):
        module = importlib.import_module(f"hermes_lcm.{name}")
        if hasattr(module, "time"):
            patch.setattr(module, "time", replacement)


def connections(engine: Any) -> list[tuple[str, sqlite3.Connection]]:
    found: list[tuple[str, sqlite3.Connection]] = []
    seen: set[int] = set()
    for name, service in sorted(vars(engine).items()):
        conn = getattr(service, "_conn", None)
        if isinstance(conn, sqlite3.Connection) and id(conn) not in seen:
            found.append((name, conn))
            seen.add(id(conn))
    return found


def state(engine: Any, canonical: Canonical) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in self_attributes():
        value = getattr(engine, name, {"absent": True})
        if name == "_store":
            value = engine._store.get_session_messages(engine._session_id, limit=10000)
        elif name == "_dag":
            value = engine._dag.get_session_nodes(engine._session_id, limit=10000)
        elif name == "_lifecycle":
            value = engine._lifecycle.get_by_conversation(engine._conversation_id)
        result[name] = canonical.value(value, name)
    return result


def observe(case: Scenario, root: Path) -> Trace:
    root.mkdir(parents=True, exist_ok=True)
    events: Trace = []
    clock, canonical = Clock(expiry_checkpoint=case.expiry_checkpoint), Canonical(root)
    with pytest.MonkeyPatch.context() as patch:
        # Shared harness helpers are deliberately untyped; keep the boundary Any.
        pin: Any = pin_char_counter
        pin(patch)
        install_clock(patch, clock)
        patch.setenv("HERMES_HOME", str(root / "home"))
        if case.setup == "no-identity":
            patch.setenv("LCM_IDENTITY_ANCHOR", "false")
        settings = dict(fresh_tail_count=2, fresh_tail_max_tokens=0, leaf_chunk_tokens=800,
                        dynamic_leaf_chunk_max=1600, context_threshold=0.001,
                        threshold_full_sweep_enabled=case.sweep, max_assembly_tokens=100000,
                        summary_model="trace-model", l3_truncate_tokens=2,
                        temporal_rollups_enabled=False, embeddings_enabled=False,
                        large_output_externalization_enabled=False,
                        large_output_active_replay_stubbing_enabled=False,
                        large_output_externalization_path=str(root / "home" / "payloads"),
                        summary_prefix_target_tokens=1000)
        settings.update(dict(case.overrides))
        context: Any = engine_context
        with context(root, window=200000, conversation_id="trace-conversation", **settings) as engine:
            view = transcript(case)
            if settings.get("ignore_message_patterns"):
                engine._compiled_ignore_message_patterns = [LiteralPattern()]
            if case.setup == "stored":
                engine._store.append_batch("S", view, conversation_id="trace-conversation")
                engine._ingest_cursor = len(view)
            if case.shape == "child":
                engine._mark_thread_context_stateless("trace-child")
                if case.setup == "child-usage":
                    engine.update_from_response({"prompt_tokens": 1000})
            if case.setup == "dependents":
                engine.ingest(view)
                engine._compress_run.store_ids_by_message_id = engine._get_store_id_map_for_messages(view)
                for row in view:
                    if row.get("role") == "assistant":
                        engine._remember_generated_ignored_dependent_reply(row, row["content"])
            for i, depth in enumerate(case.seeded_depths):
                engine._dag.add_node(dag.SummaryNode(
                    session_id="S", depth=depth, summary=f"Seed {i}. " + PAD,
                    token_count=500, source_token_count=1000, source_ids=[],
                    source_type="messages" if depth == 0 else "summaries", created_at=100.0 + i))
            if case.shape == "hidden":
                engine.ingest(view)
                view = view[-2:]
            if case.shape == "retained":
                engine.ingest(view)
                view = [*view[2:], dict(view[0]), dict(view[1])]
                engine._ingest_cursor = len(view)
            if case.setup == "debt":
                engine._lifecycle.record_debt("trace-conversation", kind="raw_backlog", size_estimate=4000)
            if case.mode == "open-circuit":
                for _ in range(3):
                    engine._summary_circuit_breaker.record_failure("trace-model")
            if case.request:
                setattr(engine._compaction_request, case.request, True)
            provider = Provider(case, clock, events, canonical, engine)
            patch.setattr(escalation, "_call_llm_for_summary", provider)
            initial = state(engine, canonical)
            conns = connections(engine)
            shapes: dict[tuple[str, str], int] = {}

            def trace_sql(connection: str, sql: str) -> None:
                normalized = canonical.text(sql.strip())
                if re.match(r"(?:--\s*)?(?:INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER|BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE)\b",
                            normalized, re.IGNORECASE):
                    # Normalize whitespace outside literals only: literal whitespace is data.
                    pieces = re.split(r"('(?:''|[^'])*')", normalized)
                    normalized = "".join(piece if i % 2 else " ".join(piece.split())
                                         for i, piece in enumerate(pieces))
                    append_sql(events, sql_event(connection, normalized, shapes, events))

            for name, conn in conns:
                conn.set_trace_callback(partial(trace_sql, name))
            try:
                for attempt in range(2 if case.retry else 1):
                    events.append(f"compress attempt={attempt + 1}")
                    returned = view
                    # The public wrapper owns these scopes. Invoke the target
                    # directly so its return/state precede survival-fit changes.
                    engine._foreground_budget = engine._new_foreground_budget()
                    if case.mode == "spent":
                        clock.now += 121
                    if case.mode == "open-after-first" and attempt:
                        for _ in range(3):
                            engine._summary_circuit_breaker.record_failure("trace-model")
                    try:
                        with engine._fresh_tail_pressure_yield_invocation():
                            returned = engine._compress_impl(view, current_tokens=case.current_tokens,
                                                             force=case.force, focus_topic=case.focus)
                        events.append("return " + compact(canonical.value(returned)))
                    except BaseException as exc:
                        if not isinstance(exc, (Exception, HostInterrupted)):
                            raise
                        events.append(f"raise {type(exc).__name__} message={compact(canonical.text(str(exc)))}")
                    # This host contract exposes _assemble_context rather than a public assemble().
                    leading = engine._leading_anchor_count(returned)
                    try:
                        assembled = engine._assemble_context(returned[0] if leading else None, returned[leading:],
                            **({"retained_user_message": returned[1]} if leading == 2 else {}))
                        events.append("assemble")
                        events.extend(f"message {row.get('role')} "
                                      f"{fingerprint(canonical.text(str(row.get('content', ''))))}"
                                      for row in assembled)
                    except Exception as exc:
                        events.append(f"assemble raise {type(exc).__name__}")
                    if case.setup == "replay":
                        view = returned
                final = state(engine, canonical)
                for name, value in final.items():
                    if value != initial[name]:
                        events.append(f"attr {name}={compact(value)}")
                events.append(f"attrs count={len(final)} sha256={hashlib.sha256(encoded(final).encode()).hexdigest()}")
                for name, reason in sorted(canonical.excluded.items()):
                    events.append(f"excluded {name}={reason}")
            finally:
                for _, conn in conns:
                    conn.set_trace_callback(None)
                if case.shape == "child":
                    engine._clear_thread_context_stateless("trace-child")
    return events


def first_difference(expected: Trace, actual: Trace) -> str | None:
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else "<end of trace>"
        right = actual[index] if index < len(actual) else "<end of trace>"
        if left != right:
            return f"[{index}]: expected {left!r}, got {right!r}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    if not args.regenerate:
        parser.error("pass --regenerate to intentionally replace the golden fixture")
    with tempfile.TemporaryDirectory(prefix="compress-trace-") as directory:
        observed = {case.name: observe(case, Path(directory) / case.name) for case in SCENARIOS}
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(observed, sort_keys=True, indent=0) + "\n")
    print(f"Wrote {len(observed)} scenarios to {GOLDEN.name}")


if __name__ == "__main__":
    main()
