"""#985: an error reason keeps both ends of the host's stderr, so the command it asks for survives."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.instruments.reliability import process_cell as PC  # noqa: E402


def _cell(tmp_path: Path, text: str | None) -> PC.ProcessCell:
    cell = object.__new__(PC.ProcessCell)
    cell.d, cell.phase = tmp_path, 1
    if text is not None:
        (tmp_path / "host-1.stderr").write_text(text)
    return cell


def test_long_stderr_keeps_the_first_and_last_thousand_characters(tmp_path):
    text = "Installing dependencies. If that fails, run manually: pip install x" + "." * 5000 + "TAIL"
    out = _cell(tmp_path, text).stderr_tail()
    assert out.startswith("Installing dependencies. If that fails, run manually: pip install x")
    assert out.endswith("TAIL")
    assert " … " in out and len(out) == 2003


def test_short_and_missing_stderr_are_returned_whole(tmp_path):
    assert _cell(tmp_path, "short").stderr_tail() == "short"
    assert _cell(tmp_path / "missing", None).stderr_tail() == ""
