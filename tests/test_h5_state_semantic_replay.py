from __future__ import annotations

import sys

from benchmarking import h5_state_semantic_replay


def test_output_parent_exists_before_provider_work(tmp_path, monkeypatch):
    output = tmp_path / "new" / "nested" / "sweep.json"
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["h5_state_semantic_replay.py", "--out", str(output)],
    )

    assert h5_state_semantic_replay.main() == 3
    assert output.parent.is_dir()
    assert not output.exists()
