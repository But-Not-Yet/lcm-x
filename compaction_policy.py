"""Pure context-window, compaction-trigger and active-assembly policy.

The engine owns parsing, live environment reads, ordered state writes and logs.
Its shells use these same pieces in order; callers with complete facts can use
``resolve_compaction_policy`` without an engine or any side effects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .codex_routing import _codex_oauth_context_cap, _is_codex_gpt55_route

_CODEX_GPT55_COMPACTION_THRESHOLD = 0.85


@dataclass(frozen=True)
class PolicyFacts:
    raw_context_length: int
    model: str | None
    provider: str | None
    configured_ratio: float
    threshold_source: str
    model_thresholds: Mapping[str, float]
    autoraise_enabled: bool
    max_assembly_tokens: int
    reserve_tokens_floor: int
    absolute_threshold_tokens: int
    context_length_source: str


@dataclass(frozen=True)
class CompactionPolicy:
    raw_context_length: int
    context_length: int
    effective_context_length_cap: int | None
    effective_context_length_reason: str
    context_length_source: str
    context_threshold: float
    context_threshold_source: str
    context_threshold_autoraised: dict[str, float] | None
    threshold_percent: float
    threshold_tokens: int
    disable_autoraise: bool
    reserve_floor_disabled: bool


def resolve_threshold(
    configured: float,
    source: str,
    model_thresholds: Mapping[str, float],
    autoraise_enabled: bool,
    model: str | None,
    provider: str | None,
) -> tuple[float, str, dict[str, float] | None]:
    """Resolve longest model override, then route auto-raise, then configuration."""
    configured = float(configured)
    explicit_lcm_override = source in {
        "env:LCM_CONTEXT_THRESHOLD",
        "config_yaml:lcm.context_threshold",
    }
    if model_thresholds and model:
        best_key = ""
        for key in model_thresholds:
            if key in model and len(key) > len(best_key):
                best_key = key
        if best_key:
            override = float(model_thresholds[best_key])
            return override, f"model_thresholds:{best_key}", {"from": configured, "to": override}
    # 5.6 deliberately excluded: extending auto-raise requires its own retention run.
    if (
        _is_codex_gpt55_route(model, provider)
        and autoraise_enabled
        and not explicit_lcm_override
        and configured < _CODEX_GPT55_COMPACTION_THRESHOLD
    ):
        return (
            _CODEX_GPT55_COMPACTION_THRESHOLD,
            "codex_gpt55_autoraise",
            {"from": configured, "to": _CODEX_GPT55_COMPACTION_THRESHOLD},
        )
    return configured, source, None


def effective_context_length(
    raw_context_length: int, model: str | None, provider: str | None,
) -> tuple[int, int | None, str]:
    cap = _codex_oauth_context_cap(model, provider)
    if cap is not None and raw_context_length > cap:
        return cap, cap, "codex_oauth_context_cap"
    return raw_context_length, None, ""


def assembly_token_cap(
    context_length: int, max_assembly_tokens: int, reserve_tokens_floor: int,
) -> tuple[int | None, bool]:
    """Return the cap and whether the reserve floor disables its own constraint."""
    caps: list[int] = []
    reserve_floor_disabled = False
    if max_assembly_tokens > 0:
        caps.append(max_assembly_tokens)
    if context_length > 0 and reserve_tokens_floor > 0:
        reserve_cap = context_length - reserve_tokens_floor
        if reserve_cap > 0:
            caps.append(reserve_cap)
        else:
            reserve_floor_disabled = True
    return (max(1, min(caps)) if caps else None), reserve_floor_disabled


def threshold_tokens(
    context_threshold_tokens: int,
    assembly_cap: int | None,
    absolute_threshold_tokens: int = 0,
) -> int:
    """Expose the stricter assembly trigger unless an absolute override wins."""
    if absolute_threshold_tokens > 0:
        return absolute_threshold_tokens
    if assembly_cap is not None and assembly_cap > 0:
        if context_threshold_tokens > 0:
            return min(context_threshold_tokens, assembly_cap)
        return assembly_cap
    return context_threshold_tokens


def resolve_compaction_policy(facts: PolicyFacts) -> CompactionPolicy:
    """Resolve all setter values from explicit facts, without reading or writing state."""
    raw = facts.raw_context_length if facts.raw_context_length > 0 else 0
    window, cap, reason = (
        effective_context_length(raw, facts.model, facts.provider)
        if raw > 0 else (0, None, "")
    )
    ratio, source, notice = resolve_threshold(
        facts.configured_ratio, facts.threshold_source, facts.model_thresholds,
        facts.autoraise_enabled, facts.model, facts.provider,
    )
    assembly_cap, reserve_floor_disabled = (
        assembly_token_cap(window, facts.max_assembly_tokens, facts.reserve_tokens_floor)
        if raw > 0 else (None, False)
    )
    tokens = (
        threshold_tokens(int(window * ratio), assembly_cap, facts.absolute_threshold_tokens)
        if raw > 0 else 0
    )
    return CompactionPolicy(
        raw_context_length=raw,
        context_length=window,
        effective_context_length_cap=cap,
        effective_context_length_reason=reason,
        context_length_source=facts.context_length_source,
        context_threshold=ratio,
        context_threshold_source=source,
        context_threshold_autoraised=notice,
        threshold_percent=ratio,
        threshold_tokens=tokens,
        disable_autoraise=raw > 0 and facts.absolute_threshold_tokens > 0,
        reserve_floor_disabled=reserve_floor_disabled,
    )
