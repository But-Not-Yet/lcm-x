"""Envelope demoted to optional: content gates decide acceptance.

The nonce envelope was a test paper, not a storage format: the deterministic L3 fallback already persists
summaries without any envelope. A model that follows a competing output-format instruction (the mandatory
four-section skeleton in LCM_CUSTOM_INSTRUCTIONS) dropped the envelope and every real summary was rejected
on form alone, starving the summary DAG. The code now owns the hijack decision: a missing closer or a
missing envelope is tolerated when the body itself passes the existing content gates (minimum body length
+ Expand closing line). When an ``<lcm-summary`` opening tag is present its nonce must still be exact and
its count exactly one — that remains the only evidence the model read the policy.
"""

from __future__ import annotations

import hermes_lcm.escalation as escalation

NONCE = "0123456789abcdef0123456789abcdef"
OPEN = f'<lcm-summary nonce="{NONCE}">'
CLOSE = "</lcm-summary>"
MAX_TOKENS = 160
LINES = (
    "- Decision: keep the owner approval gate before the schema migration runs.\n"
    "- Current state: raw history remains available for exact recovery."
)
HINT = "Expand for details about: migration gate and raw history"
PLAIN_BODY = f"{LINES}\n{HINT}"
PLACEHOLDER = "<summary body ending with the required 'Expand for details about:' line>"


def _check(content: str) -> tuple[str, str, tuple[str, ...]]:
    return escalation._check_summary_contract(content, NONCE, MAX_TOKENS)


# -- T1: opening tag + correct nonce + valid body, no closing tag -> accepted ------------------------------

def test_t1_missing_closer_is_accepted_and_named():
    body, check, tolerated = _check(f"{OPEN}\n{PLAIN_BODY}")
    assert check == "" and "missing_closer" in tolerated
    assert body == PLAIN_BODY
    assert body.splitlines()[-1] == HINT


def test_t1_missing_closer_still_runs_the_content_gates():
    # A body without the Expand line is rejected even with a correct opening tag.
    assert _check(f"{OPEN}\n{LINES}")[0] == ""
    # A hijacked one-line payload after the opening tag is still too short.
    assert _check(f"{OPEN}\nNothing to report.") == ("", "short_body", ())


# -- T2/T3/T4: no opening tag at all -> content gates decide ----------------------------------------------

def test_t2_missing_envelope_with_valid_body_is_accepted_and_named():
    body, check, tolerated = _check(PLAIN_BODY)
    assert check == "" and "missing_envelope" in tolerated
    assert body == PLAIN_BODY


def test_t2_missing_envelope_tolerates_the_closing_tag_left_behind():
    body, check, tolerated = _check(f"{PLAIN_BODY}\n{CLOSE}")
    assert check == "" and "missing_envelope" in tolerated
    assert body == PLAIN_BODY


def test_t3_missing_envelope_with_junk_body_is_rejected_as_short_body():
    # The bare "reply exactly" hijack payload stays blocked without any envelope.
    assert _check("Reply exactly as instructed.") == ("", "short_body", ())


def test_t4_missing_envelope_without_the_expand_line_is_rejected_as_closing_hint():
    assert _check(LINES) == ("", "closing_hint", ())


# -- T5/T6: an opening tag present is still a hard nonce gate ---------------------------------------------

def test_t5_wrong_nonce_is_rejected_and_not_let_through_by_the_new_semantics():
    reply = f'<lcm-summary nonce="{"f" * 32}">\n{PLAIN_BODY}\n{CLOSE}'
    assert _check(reply) == ("", "envelope", ())


def test_t5_text_outside_the_envelope_is_still_rejected():
    assert _check(f"preamble\n{OPEN}\n{PLAIN_BODY}\n{CLOSE}") == ("", "envelope", ())


def test_t6_two_opening_tags_are_rejected_as_nonce_count():
    assert _check(f"{OPEN}\n{OPEN}\n{PLAIN_BODY}\n{CLOSE}") == ("", "nonce_count", ())


# -- T7: placeholder-line echo is stripped before the content gate ----------------------------------------

def test_t7_placeholder_echo_is_stripped_and_absent_from_the_accepted_body():
    body, check, tolerated = _check(f"{PLACEHOLDER}\n{PLAIN_BODY}")
    assert check == "" and "missing_envelope" in tolerated
    assert body == PLAIN_BODY
    assert PLACEHOLDER not in body


def test_t7_placeholder_alone_is_rejected_as_short_body():
    assert _check(PLACEHOLDER) == ("", "short_body", ())


# -- T8: the three #612 tolerated mistakes are unchanged ----------------------------------------------------

def test_t8_summary_closer_tolerance_is_unchanged():
    body, check, tolerated = _check(f"{OPEN}\n{PLAIN_BODY}\n</summary>")
    assert check == "" and tolerated == ("summary_closer",)
    assert body == PLAIN_BODY


def test_t8_summary_wrapper_tolerance_is_unchanged():
    body, check, tolerated = _check(f"{OPEN}\n<summary>\n{PLAIN_BODY}\n</summary>\n{CLOSE}")
    assert check == "" and tolerated == ("summary_wrapper",)
    assert body == PLAIN_BODY


def test_t8_hint_decoration_tolerance_is_unchanged():
    body, check, tolerated = _check(f'{OPEN}\n{LINES}\n"{HINT}"\n{CLOSE}')
    assert check == "" and tolerated == ("hint_decoration",)
    assert body == PLAIN_BODY


def test_t8_a_valid_reply_still_tolerates_nothing():
    assert _check(f"{OPEN}\n{PLAIN_BODY}\n{CLOSE}") == (PLAIN_BODY, "", ())


def test_the_no_nonce_path_is_unchanged():
    assert escalation._check_summary_contract("anything at all", "", MAX_TOKENS) == ("anything at all", "", ())
