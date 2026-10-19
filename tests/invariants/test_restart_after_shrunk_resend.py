"""I1: pin the current two-row ledger after a shrunk resend and restart."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests import harness


@pytest.mark.parametrize("metadata", ["bare", "message_uid"])
def test_restart_after_shrunk_resend(tmp_path: Path, metadata: str) -> None:
    rows: list[dict[str, Any]] = [
        {"role": "user", "content": "request B0"},
        {"role": "assistant", "content": "reply B1"},
    ]
    if metadata == "message_uid":
        for index, row in enumerate(rows):
            row["message_uid"] = f"old-{index}"
    make_engine: Any = harness.make_engine
    engine = make_engine(tmp_path, window=128000, conversation_id="conv")
    try:
        for view in (rows, rows[:1], [rows[0], rows[0]]):
            engine.ingest(deepcopy(view))
        engine.shutdown()
        engine = make_engine(tmp_path, window=128000, conversation_id="conv")
        engine.ingest(deepcopy(rows))
        stored = engine._store.get_session_messages("S")
        assert [(row["role"], row["content"]) for row in stored] == [
            (row["role"], row["content"]) for row in rows]
    finally:
        engine.shutdown()
