"""Pure prefix budgeting and selection; assembly owns reads and payload writes."""

from dataclasses import dataclass
from typing import Callable, Sequence

from .extraction import strip_injected_context_blocks


def default_prefix_budget(
    *, ceiling: int, overhead: int, leading_tokens: int, tail_tokens: int,
    repaired_tail_tokens: int, recall_budget_tokens: int | None,
    proactive_recall_enabled: bool, embeddings_enabled: bool, survival_fit: bool,
) -> int:
    """Reserve the measured tail/recall; the quarter floor requires later fitting."""
    return max(
        ceiling - overhead - leading_tokens - max(tail_tokens, repaired_tail_tokens)
        - (max(0, int(recall_budget_tokens or 0))
           if proactive_recall_enabled and embeddings_enabled else 0),
        ceiling // 4 if survival_fit else 0,
    )


@dataclass(frozen=True)
class PrefixBudgetFacts:
    """Resolved budget and ordering facts, after lazy objective fallback preparation.

    None means unbounded; zero is a real budget. The counter is explicit so the
    caller's token-counting policy applies to the planner and the assembly shell.
    """

    part_keys: Sequence[tuple[int, int | None]]
    summary_role: str
    summary_budget: int | None
    assembly_cap: int | None
    survival_fit: bool
    count_message_tokens: Callable[[dict], int]


@dataclass(frozen=True)
class PrefixSelection:
    kept_indexes: tuple[int, ...]
    selected_parts: tuple[str, ...]


def select_prefix(parts: Sequence[str], budget_facts: PrefixBudgetFacts) -> PrefixSelection:
    """Select by group/newness, render in source order, and leave inputs intact.

    The engine resolves the default budget and tries objective fallbacks first:
    preparing those fallbacks can externalize payloads and must remain lazy.
    """
    summary_parts = list(parts)
    part_keys = budget_facts.part_keys
    summary_role = budget_facts.summary_role
    summary_budget = budget_facts.summary_budget
    count_message_tokens = budget_facts.count_message_tokens
    anchor_exempt = (
        summary_budget is not None and budget_facts.assembly_cap is None and budget_facts.survival_fit
        and bool(summary_parts) and part_keys[0] == (-1, None) and count_message_tokens({
            "role": summary_role, "content": strip_injected_context_blocks(summary_parts[0]),
        }) > summary_budget
    )
    kept_indexes = list(range(len(summary_parts)))
    if anchor_exempt:
        summary_parts[0] = strip_injected_context_blocks(summary_parts[0])
        kept_indexes = [0]
    elif summary_budget is not None:
        # Within a depth keep newest parts first; render in the original order.
        kept_indexes = []
        for index in sorted(range(len(summary_parts)), key=lambda i: (part_keys[i][0], -i)):
            candidate = "\n\n---\n\n".join(
                strip_injected_context_blocks(summary_parts[i])
                if part_keys[i] == (-1, None) and count_message_tokens({
                    "role": summary_role, "content": summary_parts[i],
                }) > summary_budget else summary_parts[i]
                for i in sorted(kept_indexes + [index]))
            candidate_msg = {"role": summary_role, "content": candidate}
            if count_message_tokens(candidate_msg) > summary_budget:
                continue
            kept_indexes.append(index)
        kept_indexes.sort()
    return PrefixSelection(tuple(kept_indexes), tuple(summary_parts[i] for i in kept_indexes))
