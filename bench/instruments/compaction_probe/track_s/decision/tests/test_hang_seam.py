"""Short real sleeps and the fake lane only; no product or credential access."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "s2"))
from s2lib import seam  # noqa: E402


@pytest.fixture
def fake(tmp_path, monkeypatch):
    seam.set_lane("fake", tmp_path)
    seam.set_fault()
    seam.take()
    monkeypatch.setenv("S2_FAULT_SLEEP_S", "0.01")
    yield seam
    seam.set_fault()
    seam.take()


def call():
    return seam.call_llm(task="compression", timeout=120, messages=[dict(role="user", content="synthetic")])


@pytest.mark.parametrize("mode", ["T", "P"])
def test_fault_window_and_call_records(fake, mode):
    fake.set_fault(mode)
    faults = []
    for index, turn in enumerate((3, 4, 7, 7, 8, 9), 1):
        fake.set_event(index, turn)
        for attempt in range(2):
            fault = (index == 3 and attempt == 0) if mode == "T" else turn in (7, 8)
            if fault:
                with pytest.raises(TimeoutError, match="host compression deadline"):
                    call()
                faults.append(turn)
            else:
                assert "model-free" in call().choices[0].message.content
            record = fake.take()[0][0]
            assert record.get("fault") == (mode if fault else None)
    receipt = fake.fault_receipt()
    assert receipt["event_index"] == 3
    assert receipt["faulted_calls"] == len(faults)
    assert receipt["faulted_turns"] == sorted(set(faults))
    assert receipt["effective_sleep_s"] == [0.01] * len(faults)


def test_absent_mode_leaves_no_fault_keys(fake):
    for index in range(1, 5):
        fake.set_event(index, index)
        call()
    assert all("fault" not in record for record in fake.take()[0])


def test_default_sleep_uses_timeout(fake, monkeypatch):
    monkeypatch.delenv("S2_FAULT_SLEEP_S")
    sleeps = []
    monkeypatch.setattr(fake.time, "sleep", sleeps.append)
    fake.set_fault("T")
    fake.set_event(3, 7)
    with pytest.raises(TimeoutError):
        call()
    assert sleeps == [120.0]
