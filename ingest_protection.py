"""Storage-boundary protection for payloads that should not live inline in SQLite.

Hermes core may hand LCM messages that already contain inline media/base64
payloads. LCM remains lossless by externalizing those payload strings and
storing compact placeholders in ``messages.content`` / ``messages.tool_calls``.
This avoids duplicating large/binary-ish payloads into SQLite rows, FTS shadow
structures, WAL files, and backups while keeping recovery available through LCM
externalized-payload tools.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import tempfile
from collections import Counter as Counter
from pathlib import Path
from typing import Any, Callable as Callable, Dict, Iterator as Iterator, List, Sequence

from .externalize import (
    externalize_ingest_payload,
    extract_externalized_ref,
    extract_externalized_refs as extract_externalized_refs,
    find_externalized_payload_for_message,
    get_large_output_storage_dir,
    is_externalized_placeholder,
    is_generated_payload_ref_name,
    load_externalized_payload,
    maybe_externalize_payload,
)
from .ingest_content import (
    _MEDIA_TYPE_HINTS as _MEDIA_TYPE_HINTS,
    _MEDIA_VALUE_KEYS as _MEDIA_VALUE_KEYS,
    _contains_media_payload as _contains_media_payload,
    _externalization_kind_for_message as _externalization_kind_for_message,
    _JSON_ESCAPED_SLASH_RE as _JSON_ESCAPED_SLASH_RE,
    _DATA_URI_BASE64_RE as _DATA_URI_BASE64_RE,
    _BASE64_RUN_RE as _BASE64_RUN_RE,
    _WRAPPED_BASE64_MIN_LINE_CHARS as _WRAPPED_BASE64_MIN_LINE_CHARS,
    _WRAPPED_BASE64_MIN_TERMINAL_LINE_CHARS as _WRAPPED_BASE64_MIN_TERMINAL_LINE_CHARS,
    _BASE64_ALPHABET_RE as _BASE64_ALPHABET_RE,
    _BASE64_LINE_ALPHABET_RE as _BASE64_LINE_ALPHABET_RE,
    _EXTERNALIZED_PLACEHOLDER_PREFIX as _EXTERNALIZED_PLACEHOLDER_PREFIX,
    _QUARANTINED_ASSISTANT_KIND as _QUARANTINED_ASSISTANT_KIND,
    _QUARANTINED_ASSISTANT_REASON as _QUARANTINED_ASSISTANT_REASON,
    _QUARANTINED_ASSISTANT_MIN_CHARS as _QUARANTINED_ASSISTANT_MIN_CHARS,
    _QUARANTINED_ASSISTANT_MIN_TOKENS as _QUARANTINED_ASSISTANT_MIN_TOKENS,
    _WORD_TOKEN_RE as _WORD_TOKEN_RE,
    _REPETITION_SEGMENT_SPLIT_RE as _REPETITION_SEGMENT_SPLIT_RE,
    _HEARTBEAT_NOISE_RE as _HEARTBEAT_NOISE_RE,
    _HEARTBEAT_NOISE_MAX_CHARS as _HEARTBEAT_NOISE_MAX_CHARS,
    _GENERIC_BASE64_MIN_CHARS as _GENERIC_BASE64_MIN_CHARS,
    _INGEST_PLACEHOLDER_RE as _INGEST_PLACEHOLDER_RE,
    _is_wrapped_base64_line as _is_wrapped_base64_line,
    _is_wrapped_base64_terminal_line as _is_wrapped_base64_terminal_line,
    _looks_like_hex_hash_inventory as _looks_like_hex_hash_inventory,
    _iter_wrapped_base64_blocks as _iter_wrapped_base64_blocks,
    _replace_wrapped_base64_blocks as _replace_wrapped_base64_blocks,
    is_externalized_ingest_placeholder as is_externalized_ingest_placeholder,
    contains_data_uri_base64 as contains_data_uri_base64,
    contains_long_base64_run as contains_long_base64_run,
    extract_ingest_externalized_refs as extract_ingest_externalized_refs,
    _is_basename_ref as _is_basename_ref,
    extract_all_externalized_payload_refs as extract_all_externalized_payload_refs,
    _safe_placeholder_metadata as _safe_placeholder_metadata,
    _normalized_repetition_segments as _normalized_repetition_segments,
    heartbeat_noise_reason as heartbeat_noise_reason,
    assistant_output_quarantine_reason as assistant_output_quarantine_reason,
    _quarantined_assistant_placeholder as _quarantined_assistant_placeholder,
    _volatile_quarantined_assistant_placeholder as _volatile_quarantined_assistant_placeholder,
    _existing_quarantined_assistant_placeholder as _existing_quarantined_assistant_placeholder,
    looks_like_long_base64 as looks_like_long_base64,
    _maybe_parse_json_string as _maybe_parse_json_string,
    _json_has_duplicate_object_keys as _json_has_duplicate_object_keys,
    _dict_field_path as _dict_field_path,
    _payload_key_field_path as _payload_key_field_path,
)
from .ingest_pem_model import (
    _PRIVATE_KEY_BODY_CHARS_RE as _PRIVATE_KEY_BODY_CHARS_RE,
    _HEX_DIGEST_RE as _HEX_DIGEST_RE,
    _PRIVATE_KEY_STRICT_MIN_CHARS as _PRIVATE_KEY_STRICT_MIN_CHARS,
    _PRIVATE_KEY_INLINE_RUN_RE as _PRIVATE_KEY_INLINE_RUN_RE,
    _split_serialized_pem_line as _split_serialized_pem_line,
    _PRIVATE_KEY_ESCAPED_SEPARATOR_HINT_RE as _PRIVATE_KEY_ESCAPED_SEPARATOR_HINT_RE,
    _normalize_escaped_solidus as _normalize_escaped_solidus,
    _trim_pem_segment as _trim_pem_segment,
    _looks_like_english_token as _looks_like_english_token,
    _PRIVATE_KEY_INLINE_SPAN_RE as _PRIVATE_KEY_INLINE_SPAN_RE,
    _PRIVATE_KEY_ARMOR_HEADER_RE as _PRIVATE_KEY_ARMOR_HEADER_RE,
    _PRIVATE_KEY_MAX_ARMOR_HEADERS as _PRIVATE_KEY_MAX_ARMOR_HEADERS,
    _PRIVATE_KEY_BEGIN_RE as _PRIVATE_KEY_BEGIN_RE,
    _PRIVATE_KEY_END_RE as _PRIVATE_KEY_END_RE,
    _PEM_ANY_MARKER_RE as _PEM_ANY_MARKER_RE,
    _PEM_LINE_KIND_OTHER as _PEM_LINE_KIND_OTHER,
    _PEM_LINE_KIND_BEGIN as _PEM_LINE_KIND_BEGIN,
    _PEM_LINE_KIND_END as _PEM_LINE_KIND_END,
    _PEM_LINE_KIND_STRICT_B64 as _PEM_LINE_KIND_STRICT_B64,
    _PEM_LINE_KIND_SHORT_B64 as _PEM_LINE_KIND_SHORT_B64,
    _PEM_LINE_KIND_END_INLINE as _PEM_LINE_KIND_END_INLINE,
    _PEM_LINE_KIND_ARMOR as _PEM_LINE_KIND_ARMOR,
    _PEM_LINE_KIND_BLANK as _PEM_LINE_KIND_BLANK,
    _PEM_LINE_KIND_END_LOOSE as _PEM_LINE_KIND_END_LOOSE,
    _PEM_LINE_KIND_PREFIXED_B64 as _PEM_LINE_KIND_PREFIXED_B64,
    _PEM_LINE_KIND_LENIENT_B64 as _PEM_LINE_KIND_LENIENT_B64,
    _pem_leading_body_run as _pem_leading_body_run,
    _pem_line_model as _pem_line_model,
)
from .ingest_pem_scan import (
    _redact_private_key_blocks_with as _redact_private_key_blocks_with,
    _has_orphaned_private_key_body as _has_orphaned_private_key_body,
    _pem_marker_with_inline_body as _pem_marker_with_inline_body,
    _has_orphan_full_width_base64_run as _has_orphan_full_width_base64_run,
    _PRIVATE_KEY_PLACEHOLDER_RE as _PRIVATE_KEY_PLACEHOLDER_RE,
    _pem_fragment_near_private_key_placeholder as _pem_fragment_near_private_key_placeholder,
)
from .message_content import normalize_content_value
from .tokens import count_tokens

logger = logging.getLogger(__name__)


# #1016: user/assistant text (``raw_payload``) is externalized only above
# max(large_output_externalization_threshold_chars, this floor). The configured threshold
# (12,000 chars by default) was sized for tool output; below the floor the model and
# recall must see the user's own words. 100,000 chars matches Lossless Claw, which
# externalizes non-tool messages only at 25,000 tokens. Tool results and media payloads
# keep the configured threshold. Not a config key. Dense scripts (CJK: ~1.5 chars/token)
# reach the same 25,000 tokens well under 100k chars, so the floor is also met in tokens,
# counted by ``tokens.count_tokens`` as assembly counts them.
_NON_TOOL_EXTERNALIZATION_FLOOR_CHARS = 100_000
_NON_TOOL_EXTERNALIZATION_FLOOR_TOKENS = 25_000


def _non_tool_text_over_floor(text: str, config, *, context_window_tokens: int | None = 0) -> bool:
    # #1055: use the bound effective window; unknown windows retain #1016's floor.
    window_floor = (max(0, context_window_tokens or 0) + 9) // 10
    char_floor = max(_NON_TOOL_EXTERNALIZATION_FLOOR_CHARS, 4 * window_floor)
    token_floor = max(_NON_TOOL_EXTERNALIZATION_FLOOR_TOKENS, window_floor)
    if len(text) > char_floor:
        return True
    # Count tokens only where externalization could apply (enabled, over the configured threshold).
    if not getattr(config, "large_output_externalization_enabled", False):
        return False
    if len(text) <= int(getattr(config, "large_output_externalization_threshold_chars", 0) or 0):
        return False
    return count_tokens(text) >= token_floor
_EXTERNALIZED_PAYLOAD_PLACEHOLDER_RE = re.compile(
    r"\[(?:Externalized|GC'd externalized) (?:tool output|payload):.*?;\s*ref=([^;\]\s]+)\]"
)
_SOURCE_LITERAL_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:(?:\d+[|:])|(?:[+-](?![+-])))?\s*"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\r\n]+\])?\s*=\s*(?:[rubf]{0,2})?[\"']\s*$",
    re.IGNORECASE,
)
_SOURCE_LITERAL_LINE_RE = re.compile(
    r"^\s*(?:(?:\d+[|:])|(?:[+-](?![+-])))\s*(?:[rubf]{0,2})?[\"']\s*$",
    re.IGNORECASE,
)
_SOURCE_DIFF_LITERAL_RE = re.compile(
    r"^\s*[+-](?![+-])\s*(?:[rubf]{0,2})?[\"'][^\r\n]*$",
    re.IGNORECASE,
)
_PERSISTED_OUTPUT_TAG = "<persisted-output>"
_PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
_PERSISTED_OUTPUT_SAVED_TO_RE = re.compile(r"^Full output saved to:\s*(?P<path>.+?)\s*$", re.MULTILINE)
_PERSISTED_OUTPUT_PREVIEW_RE = re.compile(
    r"^Preview \(first \d+ chars\):\s*\r?\n(?P<preview>.*?)\r?\n</persisted-output>\s*$",
    re.MULTILINE | re.DOTALL,
)
_PERSISTED_OUTPUT_CHAR_COUNT_RE = re.compile(r"too large\s*\((?P<count>[\d,]+)\s+characters\b", re.IGNORECASE)
_PERSISTED_OUTPUT_INLINE_PREVIEW_SHA256_RE = re.compile(
    r"\r?\n\[LCM persisted-output marker identity: preview_sha256=(?P<sha256>[0-9a-f]{64})\]"
    r"(?:\r?\n\[LCM persisted-output file generation: size=\d+; mtime_ns=\d+; ctime_ns=\d+\])?"
    r"\r?\n</persisted-output>\s*$"
)
_PERSISTED_OUTPUT_INLINE_GENERATION_RE = re.compile(
    r"\r?\n\[LCM persisted-output file generation: size=(?P<size>\d+); mtime_ns=(?P<mtime_ns>\d+); ctime_ns=(?P<ctime_ns>\d+)\]\r?\n</persisted-output>\s*$"
)
_PERSISTED_OUTPUT_INLINE_METADATA_RE = re.compile(
    r"\r?\n\[LCM persisted-output (?:file generation|marker identity):[^\r\n]*\]\s*$"
)
_UNRECOVERABLE_TRUNCATION_RE = re.compile(
    r"\[Truncated:\s*tool response was [\d,]+ chars\.\s*Full output could not be saved to sandbox\.\]",
    re.IGNORECASE,
)
_HERMES_RESULTS_DIRNAME = "hermes-results"
_MAX_RECOVERED_PERSISTED_OUTPUT_BYTES = 64 * 1024 * 1024
_SENSITIVE_PLACEHOLDER_PREFIX = "[LCM sensitive redaction:"
_EMBEDDING_PRIVACY_PLACEHOLDER_PREFIX = "[LCM embedding privacy:"
# v3: single-pass PEM scanner (short-terminal-line bound, body-adjacent END
# pairing, linear on pathological inputs) — output differs from v2 on
# truncated/multi-key material, so v2-era vectors must re-embed.
_EMBEDDING_PRIVACY_TRANSFORM_VERSION = "privacy:v3"
_EMBEDDING_PRIVACY_PLACEHOLDER_RE = re.compile(
    r"\[LCM (?:sensitive redaction|embedding privacy):\s*"
    r"name=(?P<name>[a-z0-9_-]+)[^\]]*\]",
    re.IGNORECASE,
)
# Includes the resolver's aliases (embedding_provider.resolve_provider maps
# "openai"/"siliconflow" to the openai-compatible cloud provider): the
# privacy predicate must hold for every spelling a CALLER might pass, not
# just the canonical id the resolved provider object carries — a config
# string alias must never bypass cloud privacy (#370 review).
_EMBEDDING_PRIVACY_CLOUD_PROVIDERS = frozenset(
    {"voyage", "voyageai", "openai-compatible", "openai", "siliconflow"}
)
_SENSITIVE_PATTERN_CATALOG: dict[str, re.Pattern[str]] = {
    "api_key": re.compile(
        r"(?P<prefix>(?:\\?[\"']?)\b(?:api[_-]?key|api[_-]?token|access[_-]?token|secret[_-]?key|client[_-]?secret)\b\s*(?:\\?[\"']?)\s*[:=]\s*(?:\\?[\"']?))"
        r"(?P<secret>[A-Za-z0-9._~+/=-]{12,})"
        r"(?P<suffix>\\?[\"']?)",
        re.IGNORECASE,
    ),
    "bearer_token": re.compile(
        r"(?P<prefix>\bBearer\s+)"
        r"(?P<secret>[A-Za-z0-9._~+/=-]{12,})",
        re.IGNORECASE,
    ),
    "password_assignment": re.compile(
        r"(?P<prefix>\b(?:password|passwd|pwd|passphrase)\b\s*[\"']?\s*[:=]\s*)"
        r"(?:(?P<quote>[\"'])(?P<secret_quoted>[^\r\n\]\}]{6,}?)(?P=quote)|"
        r"(?P<secret_unquoted>[^\s,\"'\]}]{6,}))",
        re.IGNORECASE,
    ),
    "private_key": re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.IGNORECASE | re.DOTALL,
    ),
}

# Sensitive redaction runs synchronously in the ingest path. The private_key
# catalog regex is retained for configuration compatibility, but substitution
# always bypasses it in favor of the linear scanner below.
try:  # pragma: no cover - exercised when the optional dependency is absent
    import regex as _regex_engine
except Exception:  # pragma: no cover - keep the plugin importable in minimal installs
    _regex_engine = None

_SENSITIVE_MATCH_TIMEOUT_SECONDS = 1.0
# Legitimate PEM keys are a few KB; above this a DOTALL rescan is the attack, not
# a real key, so fail-open rather than block ingest.
_SENSITIVE_STDLIB_MAX_CHARS = 262_144
_BACKTRACKING_RISKY_SENSITIVE_PATTERNS = frozenset({"private_key"})
_SENSITIVE_TIMEOUT_WARNED: set[str] = set()
_SENSITIVE_STDLIB_SKIP_WARNED: set[str] = set()
_SENSITIVE_REGEX_CATALOG: dict[str, Any] = {}


def _regex_engine_flags(re_flags: int) -> int:
    mapped = 0
    if re_flags & re.IGNORECASE:
        mapped |= _regex_engine.IGNORECASE
    if re_flags & re.DOTALL:
        mapped |= _regex_engine.DOTALL
    if re_flags & re.MULTILINE:
        mapped |= _regex_engine.MULTILINE
    if re_flags & re.VERBOSE:
        mapped |= _regex_engine.VERBOSE
    return mapped


def _regex_pattern_for(name: str) -> Any:
    """Lazily compile the timeout-capable `regex` mirror of a catalog pattern."""
    if _regex_engine is None:
        return None
    cached = _SENSITIVE_REGEX_CATALOG.get(name)
    if cached is not None:
        return cached
    stdlib_pattern = _SENSITIVE_PATTERN_CATALOG[name]
    compiled = _regex_engine.compile(
        stdlib_pattern.pattern, _regex_engine_flags(stdlib_pattern.flags)
    )
    _SENSITIVE_REGEX_CATALOG[name] = compiled
    return compiled


def _apply_sensitive_pattern(name: str, repl, text: str) -> str:
    """Substitute one sensitive pattern with a ReDoS-safe strategy.

    Private keys always use the linear scanner; other catalog patterns use
    their bounded character-class regexes.
    """
    # Only the private_key pattern (lazy `.*?` under DOTALL, which rescans to
    # end-of-string per unmatched BEGIN header) is O(n^2) and needs a guard.
    # The other patterns are character-class-bounded and linear, so they always
    # run via stdlib and never fail open - a redaction bypass under CPU load
    # would be a silent secret leak, so we restrict fail-open to the one
    # pattern that genuinely requires it.
    if name == "private_key":
        return _redact_private_key_blocks(text)
    if name in _BACKTRACKING_RISKY_SENSITIVE_PATTERNS:
        regex_pattern = _regex_pattern_for(name)
        if regex_pattern is not None:
            try:
                return regex_pattern.sub(
                    repl, text, timeout=_SENSITIVE_MATCH_TIMEOUT_SECONDS
                )
            except TimeoutError:
                if name not in _SENSITIVE_TIMEOUT_WARNED:
                    _SENSITIVE_TIMEOUT_WARNED.add(name)
                    logger.warning(
                        "LCM sensitive redaction %r timed out after %.3gs; leaving "
                        "span unredacted for this input",
                        name,
                        _SENSITIVE_MATCH_TIMEOUT_SECONDS,
                    )
                return _redact_private_key_blocks(text)
    return _SENSITIVE_PATTERN_CATALOG[name].sub(repl, text)


def _redact_private_key_blocks(text: str) -> str:
    """Redact PEM private-key blocks with a linear scanner.

    This keeps large valid keys protected even when the optional ``regex``
    package is unavailable, without running the stdlib DOTALL private-key
    pattern over a pathological multi-MB input. A truncated block (BEGIN with
    no matching END) is redacted only through its contiguous PEM-shaped base64
    body, preserving unrelated trailing prose (#365).
    """
    return _redact_private_key_blocks_with(
        text, lambda secret: _sensitive_placeholder("private_key", secret)
    )


def _is_unrecoverable_tool_truncation_marker(text: str | None) -> bool:
    return isinstance(text, str) and bool(_UNRECOVERABLE_TRUNCATION_RE.search(text))


def _expected_persisted_output_chars(text: str | None) -> int | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_CHAR_COUNT_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group("count").replace(",", ""))
    except ValueError:
        return None


def _persisted_output_preview_prefix(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_PREVIEW_RE.search(text.strip())
    if not match:
        return None
    preview = match.group("preview")
    if preview.endswith("\r\n..."):
        preview = preview[: -len("\r\n...")]
    elif preview.endswith("\n..."):
        preview = preview[: -len("\n...")]
    return preview


def _persisted_output_preview_prefix_digest(text: str | None) -> str | None:
    preview_prefix = _persisted_output_preview_prefix(text)
    if not preview_prefix:
        return None
    return hashlib.sha256(
        preview_prefix.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _persisted_output_inline_preview_sha256(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_INLINE_PREVIEW_SHA256_RE.search(text)
    if not match:
        return None
    return match.group("sha256")


def _inline_persisted_output_generation_metadata(text: str | None) -> dict[str, int] | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_INLINE_GENERATION_RE.search(text)
    if not match:
        return None
    try:
        return {
            "size": int(match.group("size")),
            "mtime_ns": int(match.group("mtime_ns")),
            "ctime_ns": int(match.group("ctime_ns")),
        }
    except (TypeError, ValueError):
        return None


def _has_inline_persisted_output_generation_metadata(text: str | None) -> bool:
    return _inline_persisted_output_generation_metadata(text) is not None


def _persisted_output_marker_identity_digest(text: str | None) -> str | None:
    return _persisted_output_inline_preview_sha256(text) or _persisted_output_preview_prefix_digest(text)


def _has_lossy_sensitive_redaction(text: str | None) -> bool:
    if not isinstance(text, str) or _SENSITIVE_PLACEHOLDER_PREFIX not in text:
        return False
    for match in re.finditer(r"\[LCM sensitive redaction: (?P<body>[^\]]+)\]", text):
        body = match.group("body")
        fields = {
            key: value
            for key, value in re.findall(r"([A-Za-z0-9_]+)=([^;]+)", body)
        }
        if fields.get("name") == "password_assignment" and "sha256" not in fields:
            return True
    return False


def _persisted_output_saved_path(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_SAVED_TO_RE.search(text.strip())
    if not match:
        return None
    raw_path = match.group("path").strip()
    if not raw_path or "\x00" in raw_path:
        return None
    return raw_path


def _safe_temp_hermes_results_file(path: Path) -> Path | None:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        return None
    parent = path.parent
    if parent.name != _HERMES_RESULTS_DIRNAME:
        return None
    try:
        expected_parent = (Path(tempfile.gettempdir()) / _HERMES_RESULTS_DIRNAME).resolve()
        parent_is_valid_dir = parent.exists() and parent.is_dir() and not parent.is_symlink()
        if not parent_is_valid_dir or parent.resolve() != expected_parent:
            return None
        return expected_parent / path.name
    except OSError:
        return None


def _is_hermes_persisted_output_marker(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    marker = text.strip()
    return (
        marker.startswith(_PERSISTED_OUTPUT_TAG)
        and marker.endswith(_PERSISTED_OUTPUT_CLOSING_TAG)
        and _expected_persisted_output_chars(marker) is not None
        and _PERSISTED_OUTPUT_SAVED_TO_RE.search(marker) is not None
    )


def _stat_generation_metadata(stats: os.stat_result) -> dict[str, int]:
    return {
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
        "ctime_ns": int(stats.st_ctime_ns),
    }


def _read_regular_file_no_symlink(path: Path) -> tuple[str, dict[str, int]] | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    fd: int | None = None
    try:
        lstat_result = os.lstat(str(path))
        if not stat.S_ISREG(lstat_result.st_mode):
            return None
        if lstat_result.st_size > _MAX_RECOVERED_PERSISTED_OUTPUT_BYTES:
            return None
        fd = os.open(str(path), flags)
        stats_before = os.fstat(fd)
        if not stat.S_ISREG(stats_before.st_mode):
            return None
        if stats_before.st_size > _MAX_RECOVERED_PERSISTED_OUTPUT_BYTES:
            return None
        with os.fdopen(fd, "rb") as handle:
            fd = None
            raw = handle.read()
            stats_after = os.fstat(handle.fileno())
        if _stat_generation_metadata(stats_before) != _stat_generation_metadata(stats_after):
            return None
        return raw.decode("utf-8"), _stat_generation_metadata(stats_after)
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def recover_hermes_persisted_output_with_file_stat(text: str | None) -> tuple[str, dict[str, int]] | None:
    """Recover Hermes host `<persisted-output>` content when the backing file is safe.

    Recovery is intentionally conservative: the marker must include Hermes'
    character count, the file path must be an absolute basename under a
    `hermes-results` temp directory, the target must be a regular non-symlink
    file, and the recovered character count must match the marker. If any check
    fails, callers should keep the marker/preview instead of claiming lossless
    recovery from an unsafe or stale file.
    """
    if not isinstance(text, str) or not _is_hermes_persisted_output_marker(text):
        return None
    expected_chars = _expected_persisted_output_chars(text)
    if expected_chars is None:
        return None
    raw_path = _persisted_output_saved_path(text)
    if raw_path is None:
        return None
    path = Path(raw_path)
    safe_path = _safe_temp_hermes_results_file(path)
    if safe_path is None:
        return None
    recovered_with_stat = _read_regular_file_no_symlink(safe_path)
    if recovered_with_stat is None:
        return None
    recovered, file_stat = recovered_with_stat
    if len(recovered) != expected_chars:
        return None
    preview_prefix = _persisted_output_preview_prefix(text)
    if not preview_prefix or not recovered.startswith(preview_prefix):
        return None
    return recovered, file_stat


def recover_hermes_persisted_output(text: str | None) -> str | None:
    recovered_with_stat = recover_hermes_persisted_output_with_file_stat(text)
    if recovered_with_stat is None:
        return None
    recovered, _file_stat = recovered_with_stat
    return recovered


def _add_inline_persisted_output_generation_metadata(text: str, file_stat: dict[str, int] | None) -> str:
    if not file_stat or not isinstance(text, str) or "</persisted-output>" not in text:
        return text
    generation = (
        "[LCM persisted-output file generation: "
        f"size={file_stat['size']}; "
        f"mtime_ns={file_stat['mtime_ns']}; "
        f"ctime_ns={file_stat['ctime_ns']}]"
    )
    if generation in text:
        return text
    return text.replace("</persisted-output>", f"{generation}\n</persisted-output>", 1)


def _add_inline_persisted_output_identity_metadata(text: str, preview_sha256: str | None) -> str:
    if (
        not isinstance(text, str)
        or "</persisted-output>" not in text
        or not isinstance(preview_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", preview_sha256)
    ):
        return text
    if _has_lossy_sensitive_redaction(text) or _persisted_output_inline_preview_sha256(text):
        return text
    identity = f"[LCM persisted-output marker identity: preview_sha256={preview_sha256}]"
    return text.replace("</persisted-output>", f"{identity}\n</persisted-output>", 1)


def sensitive_pattern_status(config: Any) -> dict[str, Any]:
    """Return metadata-only status for durable and provider-copy protection."""
    configured, active, unknown = _configured_sensitive_pattern_names(config)
    enabled = bool(getattr(config, "sensitive_patterns_enabled", False))
    privacy_setting = getattr(config, "embedding_privacy_enabled", None)
    privacy_provider = str(getattr(config, "embedding_provider", "") or "")
    privacy_required = embedding_provider_requires_privacy(privacy_provider)
    privacy_enabled = privacy_required and privacy_setting is not False
    return {
        "sensitive_patterns_enabled": enabled,
        "enabled": enabled,
        "sensitive_patterns": configured,
        "patterns": configured,
        "active_patterns": active if enabled else [],
        "unknown_patterns": unknown,
        "source": getattr(config, "sensitive_patterns_source", "default"),
        "embedding_privacy_setting": (
            "auto" if privacy_setting is None else "on" if privacy_setting else "off"
        ),
        "embedding_privacy_enabled": privacy_enabled,
        "embedding_privacy_provider_requires_privacy": privacy_required,
        "embedding_privacy_revision": (
            None
            if not privacy_required
            else "active" if privacy_enabled else "privacy:off"
        ),
        "placeholder_format": "[LCM sensitive redaction: name=<pattern>; chars=<n>; bytes=<n>; sha256=<16 for non-password>]",
        "lossless_recovery": False if enabled and active else None,
    }


def _configured_sensitive_pattern_names(config: Any) -> tuple[list[str], list[str], list[str]]:
    raw = getattr(config, "sensitive_patterns", []) or []
    if isinstance(raw, str):
        names = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in raw if str(part).strip()]
    if not names:
        return [], [], []
    configured: list[str] = []
    active: list[str] = []
    unknown: list[str] = []
    for name in names:
        normalized = name.lower().strip()
        if normalized in {"all", "default"}:
            for catalog_name in _SENSITIVE_PATTERN_CATALOG:
                if catalog_name not in configured:
                    configured.append(catalog_name)
                if catalog_name not in active:
                    active.append(catalog_name)
            continue
        configured.append(normalized)
        if normalized in _SENSITIVE_PATTERN_CATALOG:
            if normalized not in active:
                active.append(normalized)
        elif normalized not in unknown:
            unknown.append(normalized)
    return configured, active, unknown


class EmbeddingPrivacyPolicyError(RuntimeError):
    """Cloud embedding input cannot be proven safe under the active policy."""


class EmbeddingIdentityStaleError(EmbeddingPrivacyPolicyError):
    """The configured cloud privacy revision has no matching active vector identity (re-register with warmup)."""


def embedding_provider_requires_privacy(provider_id: str) -> bool:
    """Return whether a provider may send embedding input off-machine."""
    return str(provider_id or "").strip().lower() in _EMBEDDING_PRIVACY_CLOUD_PROVIDERS


def embedding_privacy_revision(config: Any) -> str:
    """Return the canonical identity revision for cloud embedding input.

    The active revision contains only a transform version and a digest of sorted
    pattern *names*. Explicit opt-out returns ``privacy:off``. Neither posture
    depends on matched text or secret bytes.
    """
    if getattr(config, "embedding_privacy_enabled", None) is False:
        return "privacy:off"
    configured, active, unknown = _configured_sensitive_pattern_names(config)
    if not configured or not active:
        raise EmbeddingPrivacyPolicyError(
            "cloud embedding privacy requires a nonempty sensitive-pattern policy"
        )
    if unknown:
        raise EmbeddingPrivacyPolicyError(
            "cloud embedding privacy policy contains unknown pattern names: "
            + ", ".join(sorted(unknown))
        )
    names = sorted(set(active))
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return f"{_EMBEDDING_PRIVACY_TRANSFORM_VERSION}:{digest}"


def _embedding_privacy_placeholder(pattern_name: str) -> str:
    return (
        f"{_EMBEDDING_PRIVACY_PLACEHOLDER_PREFIX} "
        f"name={_safe_placeholder_metadata(pattern_name)}]"
    )


def _canonicalize_embedding_privacy_placeholders(text: str) -> str:
    return _EMBEDDING_PRIVACY_PLACEHOLDER_RE.sub(
        lambda match: _embedding_privacy_placeholder(
            str(match.group("name") or "unknown").strip().lower()
        ),
        text,
    )


def _embedding_privacy_redact_match(
    pattern_name: str, match: re.Match[str]
) -> str:
    group_names = match.re.groupindex
    secret_group = None
    for candidate in ("secret", "secret_quoted", "secret_unquoted"):
        if candidate in group_names and match.groupdict().get(candidate) is not None:
            secret_group = candidate
            break
    placeholder = _embedding_privacy_placeholder(pattern_name)
    if secret_group is None:
        return placeholder
    relative_start = match.start(secret_group) - match.start(0)
    relative_end = match.end(secret_group) - match.start(0)
    full = match.group(0)
    return full[:relative_start] + placeholder + full[relative_end:]


def _private_key_first(names: Sequence[str]) -> list[str]:
    """Order pattern names so private_key runs before any assignment pattern.

    An assignment pattern whose secret group matches the literal ``-----BEGIN``
    would otherwise consume the PEM begin marker and blind the private-key
    redactor, leaking the key body (#365). Relative order of the rest is kept.
    """
    ordered = [n for n in names if n == "private_key"]
    ordered.extend(n for n in names if n != "private_key")
    return ordered


def _embedding_privacy_redact_private_keys(text: str) -> str:
    """Replace complete or truncated PEM private-key blocks with placeholders."""
    return _redact_private_key_blocks_with(
        text, lambda _secret: _embedding_privacy_placeholder("private_key")
    )


def _embedding_privacy_residual_patterns(
    text: str, active_names: Sequence[str]
) -> list[str]:
    residual: list[str] = []
    for name in active_names:
        if name == "private_key":
            # Transform-INDEPENDENT check (#365): a surviving PEM END marker means
            # an earlier pattern consumed the BEGIN marker the redactor keys on, so
            # the key body shipped. Flag it even though the BEGIN-based redactor is
            # now blind. (BEGIN redaction still flagged for truncated/no-END blocks.)
            # Backstops (#383 rounds 4-6, #391): a surviving marker beside a
            # base64 run, >=2 contiguous full-width markerless base64 body
            # lines, and a 16+ base64 run in a private_key placeholder's
            # raw-text adjacency window — all block fail-closed, so the
            # validator no longer shares the transform's marker/placeholder-
            # keyed blind spot. The proximity backstop excludes pure-hex runs
            # (git SHAs / digests) so "redacted key + nearby git SHA in prose"
            # embeds (#389) — three line-model rewrites of it each re-opened a
            # leak shape, so the raw-text scan is authoritative (#391).
            # The marker-independent backstops scan for base64 runs; escaped
            # solidi (\/, any depth) split a run into sub-16 chunks and hide it,
            # so also run them on the solidi-normalized text (#383 round-7,
            # chwFQ). Placeholders and markers carry no solidi, so normalization
            # only rejoins base64 — strictly more sensitive, detection-only.
            normalized = _normalize_escaped_solidus(text)
            if (
                _embedding_privacy_redact_private_keys(text) != text
                or _has_orphaned_private_key_body(text)
                or _pem_marker_with_inline_body(text)
                or _pem_marker_with_inline_body(normalized)
                or _has_orphan_full_width_base64_run(text)
                or _has_orphan_full_width_base64_run(normalized)
                or _pem_fragment_near_private_key_placeholder(text)
                or _pem_fragment_near_private_key_placeholder(normalized)
            ):
                residual.append(name)
            continue
        if _SENSITIVE_PATTERN_CATALOG[name].search(text) is not None:
            residual.append(name)
    return residual


def protect_embedding_text(
    text: str,
    config,
    *,
    expected_revision: str | None = None,
) -> tuple[str, str, bool]:
    """Return residual-clean provider input without mutating durable source."""
    revision = embedding_privacy_revision(config)
    if expected_revision is not None and revision != str(expected_revision):
        raise EmbeddingPrivacyPolicyError(
            "cloud embedding privacy policy differs from registered vector identity; "
            "run `/lcm embed warmup` before dispatch"
        )
    original = str(text)
    if revision == "privacy:off":
        return original, revision, False
    protected = _canonicalize_embedding_privacy_placeholders(original)
    _configured, active, _unknown = _configured_sensitive_pattern_names(config)
    # private_key MUST run first: an assignment pattern (e.g. password_assignment,
    # whose secret_unquoted group matches the literal "-----BEGIN") can otherwise
    # consume the PEM begin marker and leave the key body unredacted (#365).
    for name in _private_key_first(sorted(set(active))):
        if name == "private_key":
            protected = _embedding_privacy_redact_private_keys(protected)
        else:
            protected = _SENSITIVE_PATTERN_CATALOG[name].sub(
                lambda match, pattern_name=name: _embedding_privacy_redact_match(
                    pattern_name, match
                ),
                protected,
            )
    residual = _embedding_privacy_residual_patterns(protected, active)
    if residual:
        raise EmbeddingPrivacyPolicyError(
            "cloud embedding privacy residual detector blocked pattern names: "
            + ", ".join(sorted(set(residual)))
        )
    return protected, revision, protected != original


def validate_embedding_privacy_dispatch(
    texts: Sequence[str],
    config,
    *,
    expected_revision: str,
) -> str:
    """Revalidate policy identity and exact outbound text before a cloud call."""
    revision = embedding_privacy_revision(config)
    if revision != str(expected_revision):
        raise EmbeddingPrivacyPolicyError(
            "cloud embedding privacy policy changed before provider dispatch"
        )
    if revision == "privacy:off":
        return revision
    _configured, active, _unknown = _configured_sensitive_pattern_names(config)
    for text in texts:
        current = str(text)
        if _canonicalize_embedding_privacy_placeholders(current) != current:
            raise EmbeddingPrivacyPolicyError(
                "cloud embedding dispatch contains a noncanonical privacy placeholder"
            )
        residual = _embedding_privacy_residual_patterns(current, active)
        if residual:
            raise EmbeddingPrivacyPolicyError(
                "cloud embedding privacy residual detector blocked pattern names: "
                + ", ".join(sorted(set(residual)))
            )
    return revision


def _active_sensitive_pattern_names(config: Any) -> list[str]:
    if not bool(getattr(config, "sensitive_patterns_enabled", False)):
        return []
    _configured, active, _unknown = _configured_sensitive_pattern_names(config)
    return active


def _sensitive_placeholder(pattern_name: str, secret: str) -> str:
    parts = [
        f"{_SENSITIVE_PLACEHOLDER_PREFIX} "
        f"name={_safe_placeholder_metadata(pattern_name)}; "
        f"chars={len(secret)}; bytes={len(secret.encode('utf-8', errors='surrogatepass'))}"
    ]
    if pattern_name != "password_assignment":
        digest = hashlib.sha256(secret.encode("utf-8", errors="surrogatepass")).hexdigest()[:16]
        parts.append(f"sha256={digest}")
    return "; ".join(parts) + "]"


def _redact_match(pattern_name: str, match: re.Match[str]) -> str:
    group_names = match.re.groupindex
    secret_group = None
    for candidate in ("secret", "secret_quoted", "secret_unquoted"):
        if candidate in group_names and match.groupdict().get(candidate) is not None:
            secret_group = candidate
            break
    if secret_group is None:
        return _sensitive_placeholder(pattern_name, match.group(0))
    secret = match.group(secret_group)
    relative_start = match.start(secret_group) - match.start(0)
    relative_end = match.end(secret_group) - match.start(0)
    full = match.group(0)
    return full[:relative_start] + _sensitive_placeholder(pattern_name, secret) + full[relative_end:]


def _sensitive_pattern_for_key(key: Any, active_names: list[str]) -> str | None:
    if not isinstance(key, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    compact = normalized.replace("_", "")
    if "api_key" in active_names and (
        compact in {"apikey", "apitoken", "accesstoken", "secretkey", "clientsecret"}
        or ("api" in normalized and "key" in normalized)
        or ("access" in normalized and "token" in normalized)
        or ("secret" in normalized and "key" in normalized)
    ):
        return "api_key"
    if "bearer_token" in active_names and compact in {"authorization", "authtoken", "bearertoken", "token"}:
        return "bearer_token"
    if "password_assignment" in active_names and compact in {"password", "passwd", "pwd", "passphrase"}:
        return "password_assignment"
    return None


def redact_sensitive_text(text: str, config) -> str:
    """Replace configured sensitive spans with deterministic placeholders."""
    if not isinstance(text, str) or not text:
        return text
    active_names = _active_sensitive_pattern_names(config)
    if not active_names:
        return text
    protected = text
    # private_key first — see #365; the durable default order lists
    # password_assignment before private_key, which leaks the key body to storage.
    for name in _private_key_first(active_names):
        protected = _apply_sensitive_pattern(
            name,
            lambda match, pattern_name=name: _redact_match(pattern_name, match),
            protected,
        )
    return protected


def redact_catalog_sensitive_text(text: str) -> str:
    """#1016: redact every catalog pattern, whatever the config, in derived text (the raw_payload
    preview) whose head/tail slicing could otherwise cut a credential's label off its value."""
    protected = text
    for name in _private_key_first(list(_SENSITIVE_PATTERN_CATALOG)):
        protected = _apply_sensitive_pattern(
            name,
            lambda match, pattern_name=name: _redact_match(pattern_name, match),
            protected,
        )
    return protected


def _redact_entire_sensitive_string(text: str, pattern_name: str) -> str:
    if not text or _SENSITIVE_PLACEHOLDER_PREFIX in text:
        return text
    return _sensitive_placeholder(pattern_name, text)


def redact_sensitive_value(value: Any, config, *, parse_json_strings: bool = False) -> Any:
    """Recursively redact configured sensitive values without externalizing data."""
    active_names = _active_sensitive_pattern_names(config)
    if not active_names:
        return value
    if isinstance(value, dict):
        protected: dict[Any, Any] = {}
        for key, val in value.items():
            protected_key = redact_sensitive_text(key, config) if isinstance(key, str) else key
            key_pattern = _sensitive_pattern_for_key(key, active_names)
            if key_pattern and isinstance(val, str):
                text_redacted = redact_sensitive_text(val, config)
                if text_redacted == val:
                    text_redacted = _redact_entire_sensitive_string(val, key_pattern)
                protected[protected_key] = text_redacted
            else:
                protected[protected_key] = redact_sensitive_value(
                    val,
                    config,
                    parse_json_strings=parse_json_strings,
                )
        return protected
    if isinstance(value, list):
        return [redact_sensitive_value(item, config, parse_json_strings=parse_json_strings) for item in value]
    if not isinstance(value, str):
        return value
    if parse_json_strings:
        parsed = _maybe_parse_json_string(value)
        if parsed is not None and not _json_has_duplicate_object_keys(value):
            protected = redact_sensitive_value(parsed, config, parse_json_strings=True)
            if protected != parsed:
                return json.dumps(protected, ensure_ascii=False, separators=(",", ":"))
    return redact_sensitive_text(value, config)


def _externalize_quarantined_assistant_output(
    content: str,
    *,
    role: str,
    session_id: str,
    config,
    hermes_home: str,
    reason: str,
) -> str | None:
    existing = find_externalized_payload_for_message(
        content,
        session_id=session_id,
        kind=_QUARANTINED_ASSISTANT_KIND,
        role=role,
        config=config,
        hermes_home=hermes_home,
    )
    if existing is not None:
        return _quarantined_assistant_placeholder(existing, reason=reason)

    result = externalize_ingest_payload(
        content,
        role=role,
        session_id=session_id,
        field_path="content",
        config=config,
        hermes_home=hermes_home,
        kind=_QUARANTINED_ASSISTANT_KIND,
    )
    if result is None:
        logger.warning(
            "LCM ingest protection could not quarantine repetitive assistant output; preserving inline content for lossless recovery"
        )
        return None

    payload = result.get("payload") or {}
    path = result.get("path")
    summary = {
        "ref": getattr(path, "name", ""),
        "kind": payload.get("kind", _QUARANTINED_ASSISTANT_KIND),
        "role": payload.get("role", role),
        "field_path": payload.get("field_path", "content"),
        "content_chars": payload.get("content_chars", len(content)),
        "content_bytes": payload.get("content_bytes", len(content.encode("utf-8"))),
    }
    return _quarantined_assistant_placeholder(summary, reason=reason)


def restore_ingest_payload_placeholders(
    text: str,
    *,
    config: Any,
    hermes_home: str = "",
    session_id: str = "",
) -> str:
    """Restore ingest placeholders in a stored identity string for matching only.

    Missing or mismatched payload files leave the placeholder untouched so callers
    never fabricate content or hide a recovery problem.
    """
    if not isinstance(text, str) or _EXTERNALIZED_PLACEHOLDER_PREFIX not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        payload = load_externalized_payload(ref, config=config, hermes_home=hermes_home)
        if payload is None or payload.get("kind") != "ingest_payload":
            return match.group(0)
        payload_session_id = payload.get("session_id") or ""
        if session_id and payload_session_id and payload_session_id != session_id:
            return match.group(0)
        content = payload.get("content")
        return content if isinstance(content, str) else match.group(0)

    return _INGEST_PLACEHOLDER_RE.sub(replace, text)


def _placeholder_for_payload(
    payload: str,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
) -> str | None:
    result = externalize_ingest_payload(
        payload,
        role=role,
        session_id=session_id,
        field_path=field_path,
        config=config,
        hermes_home=hermes_home,
    )
    if result is None:
        logger.warning(
            "LCM ingest protection could not externalize payload at %s; preserving inline content for lossless recovery",
            field_path,
        )
        return None
    return result["placeholder"]


def _protect_payload_substrings(
    text: str,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
) -> str:
    if not text or is_externalized_ingest_placeholder(text):
        return text

    def replace_data_uri(match: re.Match[str]) -> str:
        payload = match.group(0)
        return _placeholder_for_payload(
            payload,
            role=role,
            session_id=session_id,
            field_path=field_path,
            config=config,
            hermes_home=hermes_home,
        ) or payload

    protected = _DATA_URI_BASE64_RE.sub(replace_data_uri, text)

    def replace_base64_run(match: re.Match[str]) -> str:
        payload = match.group(1)
        if not looks_like_long_base64(payload):
            return payload
        return _placeholder_for_payload(
            payload,
            role=role,
            session_id=session_id,
            field_path=field_path,
            config=config,
            hermes_home=hermes_home,
        ) or payload

    protected = _BASE64_RUN_RE.sub(replace_base64_run, protected)

    def replace_wrapped_base64(payload: str) -> str:
        return _placeholder_for_payload(
            payload,
            role=role,
            session_id=session_id,
            field_path=field_path,
            config=config,
            hermes_home=hermes_home,
        ) or payload

    # Line-wrapped base64 (MIME/PEM) is not a single contiguous run; externalize
    # it here too so it does not land inline in SQLite/FTS/WAL/backups.
    return _replace_wrapped_base64_blocks(protected, replace_wrapped_base64)


def _protect_value(
    value: Any,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
    parse_json_strings: bool = False,
) -> Any:
    value = redact_sensitive_value(
        value,
        config,
        parse_json_strings=parse_json_strings,
    )
    if isinstance(value, dict):
        protected: dict[Any, Any] = {}
        for key, val in value.items():
            protected_key = (
                _protect_payload_substrings(
                    key,
                    role=role,
                    session_id=session_id,
                    field_path=_payload_key_field_path(field_path),
                    config=config,
                    hermes_home=hermes_home,
                )
                if isinstance(key, str)
                else key
            )
            child_path_key = "<key>" if protected_key != key else protected_key
            protected[protected_key] = _protect_value(
                val,
                role=role,
                session_id=session_id,
                field_path=_dict_field_path(field_path, child_path_key),
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=parse_json_strings,
            )
        return protected
    if isinstance(value, list):
        return [
            _protect_value(
                item,
                role=role,
                session_id=session_id,
                field_path=f"{field_path}[{idx}]",
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=parse_json_strings,
            )
            for idx, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    if parse_json_strings:
        if _json_has_duplicate_object_keys(value):
            return _protect_payload_substrings(
                value,
                role=role,
                session_id=session_id,
                field_path=field_path,
                config=config,
                hermes_home=hermes_home,
            )
        parsed = _maybe_parse_json_string(value)
        if parsed is not None:
            canonical = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            if canonical != value:
                raw_protected = _protect_payload_substrings(
                    value,
                    role=role,
                    session_id=session_id,
                    field_path=field_path,
                    config=config,
                    hermes_home=hermes_home,
                )
                if raw_protected != value:
                    return raw_protected
            protected = _protect_value(
                parsed,
                role=role,
                session_id=session_id,
                field_path=field_path,
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=True,
            )
            if protected != parsed:
                return json.dumps(protected, ensure_ascii=False, separators=(",", ":"))
            return value

    return _protect_payload_substrings(
        value,
        role=role,
        session_id=session_id,
        field_path=field_path,
        config=config,
        hermes_home=hermes_home,
    )


def protect_inline_payloads_in_text(
    text: str,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
) -> str:
    """Externalize inline media/base64 payloads inside a text scaffold.

    This is used for non-SQLite active-context scaffolds that still must not
    duplicate media-ish payloads into summaries or preserved objective text.
    """
    if not isinstance(text, str):
        return text
    text = redact_sensitive_text(text, config)
    return _protect_payload_substrings(
        text,
        role=role,
        session_id=session_id,
        field_path=field_path,
        config=config,
        hermes_home=hermes_home,
    )


def _protect_tool_calls(tool_calls: Any, *, role: str, session_id: str, config, hermes_home: str) -> Any:
    return _protect_value(
        tool_calls,
        role=role,
        session_id=session_id,
        field_path="tool_calls",
        config=config,
        hermes_home=hermes_home,
        parse_json_strings=True,
    )


def protect_message_for_ingest(
    message: Dict[str, Any],
    config,
    hermes_home: str = "",
    session_id: str = "",
    tool_name_hint: str = "",
    *,
    context_window_tokens: int | None = 0,
) -> Dict[str, Any]:
    """Return a copy of ``message`` safe to persist in SQLite.

    Payloads are externalized losslessly when they are inline media/base64-like
    strings before they hit ``messages.content`` or ``messages.tool_calls``.
    When the opt-in generic large-output externalization setting is enabled,
    whole-message content still follows the existing threshold-based behavior.
    """
    msg = dict(message or {})
    role = str(msg.get("role") or "unknown")
    # #680: the tool name the externalized-tool-output stub shows.
    tool_name = str(msg.get("tool_name") or msg.get("name") or tool_name_hint or "") if role == "tool" else ""
    raw_content = msg.get("content")
    raw_normalized_content = normalize_content_value(raw_content)
    original_content = redact_sensitive_value(
        raw_content,
        config,
        parse_json_strings=False,
    )
    normalized_content = normalize_content_value(original_content)
    recovered_with_stat = recover_hermes_persisted_output_with_file_stat(raw_normalized_content) if role == "tool" else None
    recovered_file_stat = None
    recovered_externalized = None
    if recovered_with_stat is not None:
        recovered_persisted_output, recovered_file_stat = recovered_with_stat
        recovered_content = redact_sensitive_value(
            recovered_persisted_output,
            config,
            parse_json_strings=False,
        )
        normalized_recovered_content = normalize_content_value(recovered_content)
        if normalized_recovered_content:
            persisted_output_source_path = _persisted_output_saved_path(raw_normalized_content)
            persisted_output_preview_sha256 = _persisted_output_preview_prefix_digest(raw_normalized_content)
            if _has_lossy_sensitive_redaction(normalized_content):
                persisted_output_preview_sha256 = None
            persisted_output_metadata = {
                "persisted_output_source_path": persisted_output_source_path,
                "persisted_output_expected_chars": _expected_persisted_output_chars(raw_normalized_content),
                "persisted_output_redacted_preview_sha256": _persisted_output_preview_prefix_digest(normalized_content),
                "persisted_output_file_size": recovered_file_stat["size"],
                "persisted_output_file_mtime_ns": recovered_file_stat["mtime_ns"],
                "persisted_output_file_ctime_ns": recovered_file_stat["ctime_ns"],
            }
            if persisted_output_preview_sha256:
                persisted_output_metadata["persisted_output_preview_sha256"] = persisted_output_preview_sha256
            recovered_externalized = maybe_externalize_payload(
                normalized_recovered_content,
                kind="tool_result",
                tool_call_id=str(msg.get("tool_call_id") or ""),
                session_id=session_id,
                role=role,
                config=config,
                hermes_home=hermes_home,
                force=True,
                metadata=persisted_output_metadata,
                tool_name=tool_name,
            )

    # A host-side truncation marker without durable recovered storage is not
    # lossless. Keep the marker/preview visible inline instead of hiding it
    # behind an LCM externalized-payload ref that would look recoverable.
    preserve_truncation_marker_inline = (
        role == "tool"
        and recovered_externalized is None
        and isinstance(normalized_content, str)
        and (
            _is_hermes_persisted_output_marker(normalized_content)
            or _is_unrecoverable_tool_truncation_marker(normalized_content)
        )
    )

    # Preserve the pre-existing opt-in large-output behavior on message content.
    # The always-on storage-boundary sanitizer below is a narrower safety net for
    # inline media/base64 substrings, including cases below the generic threshold
    # or when generic externalization is disabled.
    if normalized_content:
        if recovered_externalized:
            msg["content"] = recovered_externalized["placeholder"]
        elif (
            is_externalized_ingest_placeholder(normalized_content)
            or is_externalized_placeholder(normalized_content)
        ):
            msg["content"] = original_content
        elif preserve_truncation_marker_inline:
            protected_content = _protect_value(
                original_content,
                role=role,
                session_id=session_id,
                field_path="content",
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=False,
            )
            if (
                role == "tool"
                and not bool(getattr(config, "large_output_externalization_enabled", True))
                and _is_hermes_persisted_output_marker(raw_normalized_content)
            ):
                protected_content = _add_inline_persisted_output_identity_metadata(
                    normalize_content_value(protected_content) or "",
                    _persisted_output_marker_identity_digest(raw_normalized_content),
                )
            if recovered_with_stat is not None and _is_hermes_persisted_output_marker(normalized_content):
                protected_content = _add_inline_persisted_output_generation_metadata(
                    normalize_content_value(protected_content) or "",
                    recovered_file_stat,
                )
            msg["content"] = protected_content
        else:
            reason = (
                assistant_output_quarantine_reason(normalized_content)
                if role == "assistant"
                else None
            )
            externalized = None
            if reason:
                placeholder = _externalize_quarantined_assistant_output(
                    normalized_content,
                    role=role,
                    session_id=session_id,
                    config=config,
                    hermes_home=hermes_home,
                    reason=reason,
                )
                if placeholder:
                    externalized = {"placeholder": placeholder}
            kind = _externalization_kind_for_message(msg)
            if externalized is None and (
                kind != "raw_payload" or _non_tool_text_over_floor(
                    normalized_content, config, context_window_tokens=context_window_tokens,
                )
            ):
                externalized = maybe_externalize_payload(
                    normalized_content,
                    kind=kind,
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    session_id=session_id,
                    role=role,
                    config=config,
                    hermes_home=hermes_home,
                    tool_name=tool_name,
                )
            if externalized:
                msg["content"] = externalized["placeholder"]
            else:
                msg["content"] = _protect_value(
                    original_content,
                    role=role,
                    session_id=session_id,
                    field_path="content",
                    config=config,
                    hermes_home=hermes_home,
                    parse_json_strings=False,
                )
    else:
        msg["content"] = original_content

    if msg.get("tool_calls"):
        msg["tool_calls"] = _protect_tool_calls(
            msg.get("tool_calls"),
            role=role,
            session_id=session_id,
            config=config,
            hermes_home=hermes_home,
        )

    return msg


def quarantine_suspicious_assistant_message(
    message: Dict[str, Any],
    config,
    hermes_home: str = "",
    session_id: str = "",
    *,
    externalize: bool = True,
    prefer_existing_externalized: bool = False,
) -> Dict[str, Any]:
    """Return ``message`` with obviously broken assistant output quarantined.

    Unlike full ingest protection, this only touches suspicious assistant text.
    It is safe for active-context replay because it does not externalize user
    media, tool results, or ordinary long content.
    """
    msg = dict(message or {})
    role = str(msg.get("role") or "unknown")
    if role != "assistant":
        return msg
    normalized_content = normalize_content_value(msg.get("content"))
    reason = assistant_output_quarantine_reason(normalized_content)
    if not reason:
        return msg
    if externalize:
        placeholder = _externalize_quarantined_assistant_output(
            normalized_content,
            role=role,
            session_id=session_id,
            config=config,
            hermes_home=hermes_home,
            reason=reason,
        )
    else:
        placeholder = None
        if prefer_existing_externalized:
            placeholder = _existing_quarantined_assistant_placeholder(
                normalized_content,
                role=role,
                session_id=session_id,
                config=config,
                hermes_home=hermes_home,
                reason=reason,
            )
        if placeholder is None:
            placeholder = _volatile_quarantined_assistant_placeholder(
                normalized_content,
                reason=reason,
            )
    if not placeholder:
        return msg
    msg["content"] = placeholder
    return msg


def quarantine_suspicious_assistant_messages(
    messages: List[Dict[str, Any]],
    config,
    hermes_home: str = "",
    session_id: str = "",
    externalize: List[bool] | None = None,
    prefer_existing_externalized: List[bool] | None = None,
) -> List[Dict[str, Any]]:
    return [
        quarantine_suspicious_assistant_message(
            message,
            config=config,
            hermes_home=hermes_home,
            session_id=session_id,
            externalize=True if externalize is None else externalize[idx],
            prefer_existing_externalized=False
            if prefer_existing_externalized is None
            else prefer_existing_externalized[idx],
        )
        for idx, message in enumerate(messages)
    ]


def protect_messages_for_ingest(
    messages: List[Dict[str, Any]],
    config,
    hermes_home: str = "",
    session_id: str = "",
    tool_name_hints: List[str] | None = None,
    *,
    context_window_tokens: int | None = 0,
) -> List[Dict[str, Any]]:
    return [
        protect_message_for_ingest(
            message,
            config=config,
            hermes_home=hermes_home,
            session_id=session_id,
            tool_name_hint=(tool_name_hints or [])[index] if index < len(tool_name_hints or []) else "",
            context_window_tokens=context_window_tokens,
        )
        for index, message in enumerate(messages)
    ]


def _append_unique_refs(target: list[str], refs: list[str]) -> None:
    for ref in refs:
        if ref not in target:
            target.append(ref)


def _walk_string_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        parsed = _maybe_parse_json_string(value)
        if parsed is None:
            yield value
        else:
            yield from _walk_string_values(parsed)
    elif isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                yield key
            yield from _walk_string_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_string_values(item)


def _walk_tool_call_argument_values(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "arguments":
                yield nested
            yield from _walk_tool_call_argument_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_tool_call_argument_values(item)


def _is_inside_token_quote_span(text: str, start: int, token: str) -> bool:
    in_span = False
    i = 0
    while i < start:
        if text.startswith(token, i):
            in_span = not in_span
            i += len(token)
        else:
            i += 1
    return in_span


def _looks_like_example_quote_context(context: str) -> bool:
    return re.search(r"(?:pytest\s+output|log|example|traceback|failure)\s*:\s*$", context.lower()) is not None


def _has_local_escaped_quote_before(text: str, start: int) -> bool:
    boundary = max(text.rfind(delimiter, 0, start) for delimiter in (",", "{", "["))
    segment = text[boundary + 1:start]
    matches = list(re.finditer(r"\\+[\"']", segment))
    if not matches:
        return False
    quote = matches[-1]
    context = segment[max(0, quote.start() - 80):quote.start()]
    return _looks_like_example_quote_context(context)


def _is_escaped_placeholder_example(text: str, start: int) -> bool:
    prefix = text[max(0, start - 8):start]
    return prefix.endswith("\\") or _has_local_escaped_quote_before(text, start)


def _is_quoted_placeholder_example(text: str, start: int) -> bool:
    for quote_token in ('"', "'"):
        if not _is_inside_token_quote_span(text, start, quote_token):
            continue
        quote = text.rfind(quote_token, 0, start)
        if quote < 0:
            continue
        context = text[max(0, quote - 80):quote]
        if _looks_like_example_quote_context(context):
            return True
    return False


def _looks_like_json_container_string(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _looks_like_example_payload_ref(ref: str) -> bool:
    name = Path(ref).name.lower()
    return name.startswith(("example-", "example_", "fake-", "fake_", "dummy-", "dummy_", "placeholder-", "placeholder_"))


def _placeholder_line_prefix(text: str, start: int) -> str:
    separators = (("\n", 1), ("\r", 1), ("\\n", 2), ("\\r", 2))
    separator_start, separator_length = max(
        ((text.rfind(token, 0, start), length) for token, length in separators),
        key=lambda item: item[0],
    )
    prefix = text[separator_start + separator_length:start]
    return re.sub(r"\\+([\"'])", r"\1", prefix)


def _is_incidental_source_placeholder(text: str, start: int) -> bool:
    """Return true when a placeholder match *looks like* quoted source/docs.

    This is a text-shape heuristic only. It cannot distinguish an ingest-generated
    placeholder that happens to sit inside a code fence, a backticked span, or a
    line-numbered/diff excerpt from an incidental quotation of one, because both
    look identical in isolation. Callers must treat a true result as "suspect",
    not "drop"; see ``_refs_for_externalized_integrity_scan`` and
    ``scan_externalized_payload_integrity``, which settle the question with the
    robust discriminators (does the referenced payload file exist, or does the ref
    name carry externalize.py's minting shape?).
    """
    prefix = _placeholder_line_prefix(text, start)
    if (
        _SOURCE_LITERAL_ASSIGNMENT_RE.search(prefix)
        or _SOURCE_LITERAL_LINE_RE.fullmatch(prefix)
        or _SOURCE_DIFF_LITERAL_RE.fullmatch(prefix)
    ):
        return True
    before = text[:start]
    if before.count("```") % 2:
        return True
    return prefix.count("`") % 2 == 1


def _record_source_context_ref(target: list[str] | None, ref: str) -> None:
    if target is not None and ref not in target:
        target.append(ref)


def _extract_unescaped_externalized_payload_refs(
    text: str,
    *,
    ignore_quoted_spans: bool = False,
    source_context_refs: list[str] | None = None,
) -> list[str]:
    """Return confident refs; park source-context-shaped ones in ``source_context_refs``.

    A ref whose match sits in quoted-source context is not dropped here: it is
    appended to ``source_context_refs`` (when the caller supplies a list) so the
    layer that owns ``hermes_home`` can promote it back when a payload file with
    that name exists or the name carries externalize.py's minting shape. With no
    list supplied the parked refs are simply not returned, preserving the
    conservative text-only behavior.
    """
    refs: list[str] = []
    for pattern in (_INGEST_PLACEHOLDER_RE, _EXTERNALIZED_PAYLOAD_PLACEHOLDER_RE):
        for match in pattern.finditer(text):
            ref = match.group(1).strip()
            if not _is_basename_ref(ref):
                continue
            if _looks_like_example_payload_ref(ref) and _is_escaped_placeholder_example(text, match.start()):
                continue
            if (
                ignore_quoted_spans
                and _looks_like_example_payload_ref(ref)
                and _is_quoted_placeholder_example(text, match.start())
            ):
                continue
            if _is_incidental_source_placeholder(text, match.start()):
                _record_source_context_ref(source_context_refs, ref)
                continue
            if ref not in refs:
                refs.append(ref)
    return refs


def _refs_for_externalized_integrity_scan(
    value: str,
    *,
    role: str,
    field: str,
    source_context_refs: list[str] | None = None,
) -> list[str]:
    """Return refs that plausibly came from LCM storage-boundary placeholders.

    Tool outputs and tool-call arguments often contain escaped code snippets,
    pytest failures, or docs that mention placeholder examples. Counting those
    as live payload references turns doctor into a false-positive machine. Exact
    placeholders are still counted everywhere; embedded unescaped placeholders
    are counted for message content, raw JSON-container tool-call argument
    strings, and raw free-form tool-call argument strings so ingestion-produced
    refs do not disappear while quoted examples stay ignored.

    Refs suppressed only by the source-context text heuristic are appended to
    ``source_context_refs`` instead of being discarded, so the caller can settle
    them on payload-file existence rather than on text shape alone.
    """
    if not isinstance(value, str) or not value:
        return []
    stripped = value.strip()
    if is_externalized_ingest_placeholder(stripped) or is_externalized_placeholder(stripped):
        return extract_all_externalized_payload_refs(stripped)
    if field == "tool_calls":
        parsed = _maybe_parse_json_string(value)
        if parsed is None:
            return _extract_unescaped_externalized_payload_refs(
                value,
                ignore_quoted_spans=True,
                source_context_refs=source_context_refs,
            )
        # Raw JSON text erases whether a match came from quoted source. Once the
        # envelope parses, scan decoded values only and keep raw scanning as the
        # malformed-envelope fallback above.
        refs: list[str] = []
        for argument in _walk_tool_call_argument_values(parsed):
            if isinstance(argument, str):
                parsed_argument = _maybe_parse_json_string(argument)
                if parsed_argument is None:
                    _append_unique_refs(
                        refs,
                        _extract_unescaped_externalized_payload_refs(
                            argument,
                            ignore_quoted_spans=True,
                            source_context_refs=source_context_refs,
                        ),
                    )
                else:
                    for nested in _walk_string_values(parsed_argument):
                        nested_stripped = nested.strip()
                        if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                            _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
                        else:
                            _append_unique_refs(
                                refs,
                                _extract_unescaped_externalized_payload_refs(
                                    nested,
                                    ignore_quoted_spans=True,
                                    source_context_refs=source_context_refs,
                                ),
                            )
            else:
                for nested in _walk_string_values(argument):
                    nested_stripped = nested.strip()
                    if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                        _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
                    else:
                        _append_unique_refs(
                            refs,
                            _extract_unescaped_externalized_payload_refs(
                                nested,
                                ignore_quoted_spans=True,
                                source_context_refs=source_context_refs,
                            ),
                        )
        for nested in _walk_string_values(parsed):
            nested_stripped = nested.strip()
            if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
            else:
                _append_unique_refs(
                    refs,
                    _extract_unescaped_externalized_payload_refs(
                        nested,
                        ignore_quoted_spans=True,
                        source_context_refs=source_context_refs,
                    ),
                )
        return refs
    if role == "tool":
        refs = _extract_unescaped_externalized_payload_refs(
            value,
            source_context_refs=source_context_refs,
        )
        parsed = _maybe_parse_json_string(value)
        if parsed is not None:
            for nested in _walk_string_values(parsed):
                nested_stripped = nested.strip()
                if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                    _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
                else:
                    _append_unique_refs(
                        refs,
                        _extract_unescaped_externalized_payload_refs(
                            nested,
                            ignore_quoted_spans=True,
                            source_context_refs=source_context_refs,
                        ),
                    )
        return refs
    return _extract_unescaped_externalized_payload_refs(
        value,
        source_context_refs=source_context_refs,
    )


def scan_externalized_payload_integrity(conn: Any, config: Any, *, hermes_home: str = "", limit: int = 5) -> dict[str, Any]:
    """Compare externalized payload refs stored in messages with JSON files.

    This is intentionally read-only and metadata-only. It does not open payload
    files except through directory metadata, and row samples never include raw
    message content or tool-call arguments.
    """

    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    existing_files: set[str] = set()
    if storage_dir.exists() and storage_dir.is_dir():
        existing_files = {path.name for path in storage_dir.glob("*.json") if path.is_file()}

    referenced_refs: set[str] = set()
    first_location_by_ref: dict[str, dict[str, Any]] = {}
    for store_id, session_id, source, role, content, tool_calls in conn.execute(
        """
        SELECT store_id, session_id, source, role, content, tool_calls
        FROM messages
        WHERE COALESCE(content, '') LIKE '%ref=%]%'
           OR COALESCE(tool_calls, '') LIKE '%ref=%]%'
        ORDER BY store_id ASC
        """
    ).fetchall():
        for field, value in (("content", content), ("tool_calls", tool_calls)):
            if not isinstance(value, str):
                continue
            source_context_refs: list[str] = []
            scanned_refs = _refs_for_externalized_integrity_scan(
                value,
                role=str(role or ""),
                field=field,
                source_context_refs=source_context_refs,
            )
            # The text heuristics cannot tell an ingest-GENERATED placeholder that
            # happens to sit inside a code fence, a backticked span, or a
            # line-numbered excerpt from an incidental quotation of one; assistant
            # turns routinely wrap real tool output in fences. Two independent
            # signals settle a parked ref, and either one is enough:
            #   1. A backing payload file exists, so the placeholder is live.
            #   2. The ref name carries externalize.py's minting shape, so LCM
            #      generated it. This one does not consult the filesystem, which is
            #      what keeps a generated ref diagnosable after its payload is
            #      deleted or GC'd -- existence alone would drop the ref precisely
            #      when recovery is broken and doctor most needs to report it.
            # A ref with neither signal is the quoted-source case the heuristics
            # target: hand-written docs examples have no file and no minted shape.
            #
            # Accepted residual, deliberate -- do NOT "fix" it by re-gating signal 2
            # on local state: a stored excerpt quoting a real placeholder minted on
            # a DIFFERENT installation carries the minted shape with no local file,
            # so it is reported missing. That is a visible, self-correcting false
            # alarm (the operator sees a ref name nothing local ever minted), and it
            # is exactly what main did before any of this filtering existed, so it
            # is not a behavior this filter introduced. The conservative direction
            # for an integrity diagnostic is to over-report a broken ref rather than
            # go silent on one; re-gating on existence would restore the silence
            # this promotion was added to remove. Separating a foreign minted ref
            # from a local GC'd one needs durable local provenance -- a persisted
            # minted-ref ledger or a per-install salt in the filename -- which
            # changes the payload storage contract and is future work.
            promoted_refs = [
                ref
                for ref in source_context_refs
                if ref not in scanned_refs
                and (ref in existing_files or is_generated_payload_ref_name(ref))
            ]
            for ref in (*scanned_refs, *promoted_refs):
                referenced_refs.add(ref)
                first_location_by_ref.setdefault(
                    ref,
                    {
                        "store_id": int(store_id),
                        "session_id": session_id,
                        "source": source,
                        "role": role,
                        "field": field,
                        "externalized_ref": ref,
                    },
                )

    missing_refs = sorted(ref for ref in referenced_refs if ref not in existing_files)
    existing_ref_count = sum(1 for ref in referenced_refs if ref in existing_files)
    unreferenced_files = sorted(ref for ref in existing_files if ref not in referenced_refs)

    return {
        "externalized_payload_refs_total": len(referenced_refs),
        "externalized_payload_refs_existing": existing_ref_count,
        "externalized_payload_refs_missing": len(missing_refs),
        "externalized_payload_files_unreferenced": len(unreferenced_files),
        "missing_externalized_payload_refs": [
            first_location_by_ref[ref] for ref in missing_refs[:limit] if ref in first_location_by_ref
        ],
        "unreferenced_externalized_payload_files": [
            {"externalized_ref": ref} for ref in unreferenced_files[:limit]
        ],
    }


def scan_sqlite_payload_risks(conn: Any, *, limit: int = 5) -> dict[str, Any]:
    """Return bounded diagnostics for suspicious inline payload storage.

    Diagnostics intentionally omit previews/raw payload text. Rows include only
    metadata needed for triage and a recoverability ref when a compact
    externalized placeholder is present.
    """

    def make_row(row: Any, *, field: str, length_key: str, category: str) -> dict[str, Any]:
        store_id, session_id, source, role, length, value = row
        value = value or ""
        result = {
            "store_id": int(store_id),
            "session_id": session_id,
            "source": source,
            "role": role,
            "field": field,
            "length": int(length or 0),
            length_key: int(length or 0),
            "suspicious_category": category,
        }
        refs = extract_ingest_externalized_refs(value) if isinstance(value, str) else []
        ref = refs[0] if refs else (extract_externalized_ref(value) if isinstance(value, str) else None)
        if ref:
            result["externalized_ref"] = ref
        return result

    largest_content = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        ORDER BY content_len DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    largest_tool_calls = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(tool_calls), 0) AS tool_calls_len, tool_calls
        FROM messages
        ORDER BY tool_calls_len DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    candidate_cap = max(limit * 20, limit)
    # Pre-filter broadly in SQL, then apply the same conservative Python regex
    # used by ingest externalization. This avoids false positives from code or
    # doctor text that quotes scaffolds such as "data:%;base64,%" or
    # `DATA_URI = "data:image/png;base64," + DATA_PAYLOAD`.
    data_uri_content_candidates = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        WHERE lower(content) GLOB '*data:*;base64,*'
        ORDER BY content_len DESC
        LIMIT ?
        """,
        (candidate_cap,),
    ).fetchall()
    data_uri_tool_call_candidates = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(tool_calls), 0) AS tool_calls_len, tool_calls
        FROM messages
        WHERE lower(tool_calls) GLOB '*data:*;base64,*'
        ORDER BY tool_calls_len DESC
        LIMIT ?
        """,
        (candidate_cap,),
    ).fetchall()
    data_uri_content = [
        row for row in data_uri_content_candidates if isinstance(row[-1], str) and contains_data_uri_base64(row[-1])
    ][:limit]
    data_uri_tool_calls = [
        row for row in data_uri_tool_call_candidates if isinstance(row[-1], str) and contains_data_uri_base64(row[-1])
    ][:limit]

    generic_rows = []
    for store_id, session_id, source, role, content, tool_calls in conn.execute(
        """
        SELECT store_id, session_id, source, role, content, tool_calls
        FROM messages
        WHERE COALESCE(length(content), 0) >= ? OR COALESCE(length(tool_calls), 0) >= ?
        ORDER BY MAX(COALESCE(length(content), 0), COALESCE(length(tool_calls), 0)) DESC
        LIMIT ?
        """,
        (_GENERIC_BASE64_MIN_CHARS, _GENERIC_BASE64_MIN_CHARS, candidate_cap),
    ).fetchall():
        for field, value in (("content", content), ("tool_calls", tool_calls)):
            if isinstance(value, str) and contains_long_base64_run(value):
                result = {
                    "store_id": int(store_id),
                    "session_id": session_id,
                    "source": source,
                    "role": role,
                    "field": field,
                    "length": len(value),
                    "suspicious_category": "base64_like",
                }
                refs = extract_ingest_externalized_refs(value)
                ref = refs[0] if refs else extract_externalized_ref(value)
                if ref:
                    result["externalized_ref"] = ref
                generic_rows.append(result)
                break
        if len(generic_rows) >= limit:
            break

    quarantined_assistant_rows = [
        make_row(row, field="content", length_key="content_len", category=_QUARANTINED_ASSISTANT_KIND)
        for row in conn.execute(
            """
            SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
            FROM messages
            WHERE role = 'assistant'
              AND content LIKE '%Externalized LCM ingest payload:%quarantined_assistant_output%'
            ORDER BY store_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    ]

    suspicious_repetitive_assistant_rows = []
    for row in conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        WHERE role = 'assistant'
          AND COALESCE(length(content), 0) >= ?
          AND content NOT LIKE '%Externalized LCM ingest payload:%quarantined_assistant_output%'
        ORDER BY content_len DESC
        LIMIT ?
        """,
        (_QUARANTINED_ASSISTANT_MIN_CHARS, candidate_cap),
    ).fetchall():
        value = row[-1]
        if isinstance(value, str):
            reason = assistant_output_quarantine_reason(value)
            if reason:
                suspicious_repetitive_assistant_rows.append(
                    make_row(row, field="content", length_key="content_len", category=reason)
                )
        if len(suspicious_repetitive_assistant_rows) >= limit:
            break

    heartbeat_noise_rows = []
    for row in conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        WHERE role IN ('assistant', 'tool', 'system')
          AND COALESCE(length(content), 0) BETWEEN 1 AND ?
          AND (
            lower(trim(content)) GLOB 'still working*'
            OR lower(trim(content)) GLOB 'working on it*'
            OR lower(trim(content)) GLOB 'processing*'
            OR lower(trim(content)) GLOB 'checking*'
            OR lower(trim(content)) GLOB 'one moment*'
            OR lower(trim(content)) GLOB 'ping*'
            OR lower(trim(content)) GLOB 'heartbeat*'
            OR lower(trim(content)) GLOB 'no update*'
          )
        ORDER BY store_id ASC
        LIMIT ?
        """,
        (_HEARTBEAT_NOISE_MAX_CHARS, candidate_cap),
    ).fetchall():
        _store_id, _session_id, _source, role, _length, value = row
        reason = heartbeat_noise_reason(str(role or ""), value if isinstance(value, str) else "")
        if reason:
            heartbeat_noise_rows.append(
                make_row(row, field="content", length_key="content_len", category=reason)
            )
        if len(heartbeat_noise_rows) >= limit:
            break

    return {
        "largest_content_rows": [
            make_row(row, field="content", length_key="content_len", category="largest_content")
            for row in largest_content
        ],
        "largest_tool_calls_rows": [
            make_row(row, field="tool_calls", length_key="tool_calls_len", category="largest_tool_calls")
            for row in largest_tool_calls
        ],
        "suspicious_data_uri_content_rows": [
            make_row(row, field="content", length_key="content_len", category="data_uri_base64")
            for row in data_uri_content
        ],
        "suspicious_data_uri_tool_calls_rows": [
            make_row(row, field="tool_calls", length_key="tool_calls_len", category="data_uri_base64")
            for row in data_uri_tool_calls
        ],
        "suspicious_base64_like_rows": generic_rows,
        "quarantined_assistant_rows": quarantined_assistant_rows,
        "suspicious_repetitive_assistant_rows": suspicious_repetitive_assistant_rows,
        "heartbeat_noise_rows": heartbeat_noise_rows,
    }

def externalized_payload_stats(config: Any, hermes_home: str = "") -> dict[str, Any]:
    from .externalize import get_large_output_storage_dir

    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    count = 0
    total_bytes = 0
    total_chars = 0
    latest_path = ""
    latest_mtime = 0.0
    if storage_dir.exists() and storage_dir.is_dir():
        for path in storage_dir.glob("*.json"):
            if not path.is_file():
                continue
            count += 1
            try:
                stat = path.stat()
                total_bytes += int(stat.st_size)
                if stat.st_mtime > latest_mtime:
                    latest_mtime = stat.st_mtime
                    latest_path = str(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                total_chars += int(payload.get("content_chars") or len(payload.get("content", "") or ""))
            except Exception:
                continue
    return {
        "externalized_payload_dir": str(storage_dir),
        "externalized_payload_count": count,
        "externalized_payload_bytes": total_bytes,
        "externalized_payload_chars": total_chars,
        "latest_externalized_payload_path": latest_path,
        "latest_externalized_payload_mtime": latest_mtime,
    }
