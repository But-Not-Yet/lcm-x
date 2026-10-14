"""Host live compression-config methods on the LCM engine (t_b6aad2f5).

Root cause under test: tui_gateway/session_compression.py:145-172 writes
compression params onto the SELECTED engine and calls three private methods:

    cc.threshold_percent = cc._effective_threshold_percent(cc.context_length, base)  # :148
    cc.threshold_tokens_cap = cc._coerce_threshold_tokens_cap(...)                    # :167
    cc._threshold_tokens = cc._tail_token_budget = None                               # :171

With tiered(lcm-x) selected, the tiered wrapper (tiered commit 9c6a1d0)
already forwards these names to the inner engine, but the inner LCMEngine
never implemented the three methods: the call raised AttributeError and the
host swallowed the whole live-config apply (agent.log 2026-10-13 18:53:44
"Could not apply live compression config for d2bafe1c:
_coerce_threshold_tokens_cap"), leaving the config stale.

Semantics are copied verbatim from the host (agent/context_compressor.py,
read-only reference, line numbers as of 2026-10-13):
  _coerce_max_tokens            :2731-2738, aliased _coerce_threshold_tokens_cap :2741
  _effective_threshold_cap      :2665-2668 (cap clamped to the window)
  _apply_threshold_tokens_cap   :2743-2752 (minus the _aux_context_ceiling layer,
                                 which lcm-x has no concept of — known limit)
  _effective_threshold_percent  :2754-2759 with module constants
                                 _SMALL_CTX_WINDOW_LIMIT = 512_000 and
                                 _SMALL_CTX_THRESHOLD_PERCENT = 0.75 (:1150-1151)

Pre-registered acceptance (card t_b6aad2f5):
  T1  coerce matrix: 99999->99999; 0/-3/None/"abc"->None; "12345"->12345.
  T2  effective percent: window 262144 base 0.60 -> >=0.75; window 100000
      base 0.60 -> >=0.75; window 262144 base 0.80 -> 0.80 (raise-only,
      never lowered); window >= 512000 -> unchanged.
  T3  apply: threshold_tokens=157286 cap=120000 -> 120000; cap=None ->
      unchanged; cap > threshold -> unchanged.
  T4  host call sequence verbatim against a REAL LCMEngine instance, then
      the same sequence through the tiered wrapper: no AttributeError, and
      the writes land on the inner engine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

# The host ABC import: prefer the real agent.context_engine (tiered.engine
# imports it too); fall back to a minimal stub like sibling tests do.
if "agent.context_engine" not in sys.modules:
    _hermes_agent_root = Path.home() / ".hermes" / "hermes-agent"
    if str(_hermes_agent_root) not in sys.path:
        sys.path.insert(0, str(_hermes_agent_root))
    try:
        import agent.context_engine  # noqa: F401
    except Exception:  # noqa: BLE001 - host tree unavailable in this env
        _agent_mod = ModuleType("agent")
        _agent_mod.__path__ = []
        _ce_mod = ModuleType("agent.context_engine")

        class _StubContextEngine:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self.last_prompt_tokens = 0

            def get_status(self):
                return {}

        _ce_mod.ContextEngine = _StubContextEngine
        sys.modules["agent"] = _agent_mod
        sys.modules["agent.context_engine"] = _ce_mod

_existing = sys.modules.get("hermes_lcm.engine")
if _existing is not None and not hasattr(_existing, "LCMEngine"):
    sys.modules.pop("hermes_lcm.engine", None)

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

# Host constants read from agent/context_compressor.py:1150-1151 (2026-10-13).
_SMALL_CTX_WINDOW_LIMIT = 512_000
_SMALL_CTX_THRESHOLD_PERCENT = 0.75


def _make_engine(tmp_path: Path) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    return LCMEngine(config=config)


# --------------------------------------------------------------------------
# T1 — coerce matrix (host _coerce_max_tokens, context_compressor.py:2731-2741)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (99999, 99999),
        (0, None),
        (-3, None),
        (None, None),
        ("abc", None),
        ("12345", 12345),
    ],
)
def test_t1_coerce_threshold_tokens_cap_matrix(tmp_path, raw, expected):
    engine = _make_engine(tmp_path)
    assert engine._coerce_threshold_tokens_cap(raw) == expected


def test_t1_initial_cap_is_readable_none(tmp_path):
    """Host reads/writes threshold_tokens_cap before its first write must not
    raise: the attribute must exist on the engine with an initial None."""
    engine = _make_engine(tmp_path)
    assert engine.threshold_tokens_cap is None


# --------------------------------------------------------------------------
# T2 — effective percent (host _effective_threshold_percent, :2754-2759)
# --------------------------------------------------------------------------

def test_t2_small_window_floor_and_raise_only():
    # Window below 512K with a sub-75% base is raised to the 75% floor.
    assert LCMEngine._effective_threshold_percent(262144, 0.60) >= 0.75
    assert LCMEngine._effective_threshold_percent(100000, 0.60) >= 0.75
    # Raise-only: a base above the floor is never lowered.
    assert LCMEngine._effective_threshold_percent(262144, 0.80) == 0.80
    # Windows at/above the limit are returned unchanged.
    assert LCMEngine._effective_threshold_percent(_SMALL_CTX_WINDOW_LIMIT, 0.60) == 0.60
    assert LCMEngine._effective_threshold_percent(1_000_000, 0.50) == 0.50
    # Falsy window (unset) returns unchanged, matching the host's `if context_length and ...`.
    assert LCMEngine._effective_threshold_percent(0, 0.60) == 0.60


def test_t2_matches_host_implementation_when_importable():
    """Parity sweep against the live host class when its tree imports here."""
    import os
    _root = os.path.expanduser("~/.hermes/hermes-agent")
    if _root not in sys.path:  # hermes_yaml lives at the tree root
        sys.path.insert(0, _root)
    try:
        import agent.context_compressor as host_cc
    except Exception as _exc:  # noqa: BLE001 - host deps (e.g. ruamel) absent in this env
        pytest.skip(f"host agent.context_compressor not importable in this env: {_exc!r}")
    for window in (64_000, 100_000, 262_144, 512_000, 1_048_576, 0):
        for base in (0.25, 0.5, 0.6, 0.75, 0.8, 0.95):
            assert LCMEngine._effective_threshold_percent(window, base) == (
                host_cc.ContextCompressor._effective_threshold_percent(window, base)
            ), f"parity broke at window={window} base={base}"


# --------------------------------------------------------------------------
# T3 — apply cap (host _apply_threshold_tokens_cap + _effective_threshold_cap,
#      :2665-2668 / :2743-2752)
# --------------------------------------------------------------------------

def test_t3_apply_clamps_only_when_cap_binds(tmp_path):
    engine = _make_engine(tmp_path)
    engine._set_context_length(262144, source="test")

    engine.threshold_tokens = 157286
    engine.threshold_tokens_cap = 120000
    engine._apply_threshold_tokens_cap()
    assert engine.threshold_tokens == 120000

    engine.threshold_tokens = 157286
    engine.threshold_tokens_cap = None
    engine._apply_threshold_tokens_cap()
    assert engine.threshold_tokens == 157286

    engine.threshold_tokens = 157286
    engine.threshold_tokens_cap = 200000
    engine._apply_threshold_tokens_cap()
    assert engine.threshold_tokens == 157286


def test_t3_cap_clamped_to_window_like_host(tmp_path):
    """Host clamps the cap to the window first (_effective_threshold_cap
    :2665-2668): a cap above the window still pulls the trigger down to the
    window when the trigger sits above it."""
    engine = _make_engine(tmp_path)
    engine._set_context_length(100000, source="test")
    engine.threshold_tokens = 130000  # e.g. an absolute pin above the window
    engine.threshold_tokens_cap = 120000
    engine._apply_threshold_tokens_cap()
    assert engine.threshold_tokens == 100000


def test_t3_recompute_honours_a_written_cap(tmp_path):
    """Durable-cap parity (#114707): after the host writes a cap, the next
    trigger recomputation in _set_context_length must re-apply it."""
    engine = _make_engine(tmp_path)
    engine._set_context_length(262144, source="test")
    engine.threshold_tokens_cap = 120000
    engine._set_context_length(262144, source="test-recompute")
    assert engine.threshold_tokens <= 120000


# --------------------------------------------------------------------------
# T4 — host call sequence (session_compression.py:143-171) end to end
# --------------------------------------------------------------------------

def _run_host_sequence(target, *, threshold: float, threshold_tokens) -> None:
    """session_compression.py:143-171 verbatim shape (host DEFAULT_CONFIG
    threshold=0.5 / threshold_tokens=None verified 2026-10-13)."""
    # :143
    target._config_threshold_percent = target._configured_threshold_percent = threshold
    # :144-146 resolve_model_threshold returns pct unchanged when
    # model_thresholds is empty (context_compressor.py:2040).
    base = threshold
    # :147-150
    try:
        target.threshold_percent = target._effective_threshold_percent(target.context_length, base)
    except Exception:  # noqa: BLE001 - host's own except path
        target.threshold_percent = threshold
    # :167-169
    target.threshold_tokens_cap = target._coerce_threshold_tokens_cap(threshold_tokens)
    # The card's sequence additionally exercises the apply verb the host
    # calls from its own recompute paths (context_compressor.py:2223/:2693,
    # tools/delegate_tool.py:191).
    target._apply_threshold_tokens_cap()
    # :171
    target._threshold_tokens = target._tail_token_budget = None


def test_t4_real_engine_survives_host_sequence(tmp_path):
    engine = _make_engine(tmp_path)
    engine._set_context_length(262144, source="test")
    engine.threshold_tokens = 157286  # deterministic pre-cap trigger

    _run_host_sequence(engine, threshold=0.5, threshold_tokens=120000)

    assert engine.threshold_percent == _SMALL_CTX_THRESHOLD_PERCENT
    assert engine.threshold_tokens_cap == 120000
    assert engine.threshold_tokens == 120000
    assert engine._threshold_tokens is None
    assert engine._tail_token_budget is None

    # Host-default shape (threshold_tokens absent -> DEFAULT_CONFIG None):
    # cap coerces to None and the trigger stays ratio-derived.
    _run_host_sequence(engine, threshold=0.5, threshold_tokens=None)
    assert engine.threshold_tokens_cap is None


def test_t4_tiered_wrapper_forwards_to_real_inner(tmp_path):
    tiered_root = Path.home() / ".hermes" / "plugins"
    if str(tiered_root) not in sys.path:
        sys.path.insert(0, str(tiered_root))
    try:
        from tiered.engine import TieredEngine
    except ImportError as exc:  # noqa: BLE001
        pytest.skip(f"tiered wrapper unavailable in this env: {exc}")

    inner = _make_engine(tmp_path)
    inner._set_context_length(262144, source="test")
    inner.threshold_tokens = 157286
    wrapper = TieredEngine(inner)

    _run_host_sequence(wrapper, threshold=0.5, threshold_tokens=120000)

    assert inner.threshold_percent == _SMALL_CTX_THRESHOLD_PERCENT
    assert inner.threshold_tokens_cap == 120000
    assert inner.threshold_tokens == 120000
    assert inner._threshold_tokens is None
    assert inner._tail_token_budget is None
    # Reads through the wrapper must mirror the inner.
    assert wrapper.threshold_tokens_cap == 120000
    assert wrapper.threshold_percent == _SMALL_CTX_THRESHOLD_PERCENT
