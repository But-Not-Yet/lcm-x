"""Engine-construction subset of the shared behavioral harness (#1230)."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def make_engine(
    tmp_path: Path, *, window: int | None = None, session_id: str = "S",
    conversation_id: str | None = None, platform: str = "cli", **overrides: Any,
) -> LCMEngine:
    """Use shipped defaults, isolated storage, and explicit config overrides only."""
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm.db")
    names = {field.name for field in fields(config)}
    for name, value in overrides.items():
        if name not in names:
            raise AttributeError(f"Unknown LCMConfig attribute: {name}")
        setattr(config, name, value)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    if window is not None:
        kwargs: dict[str, Any] = {"platform": platform, "context_length": window}
        if conversation_id is not None:
            kwargs["conversation_id"] = conversation_id
        try:
            engine.on_session_start(session_id, **kwargs)
        except BaseException:
            engine.shutdown()
            raise
    return engine


@contextmanager
def engine_context(
    tmp_path: Path, *, window: int | None = None, session_id: str = "S",
    conversation_id: str | None = None, platform: str = "cli", **overrides: Any,
) -> Iterator[LCMEngine]:
    """Construct an engine and shut it down even when the test raises."""
    engine = make_engine(tmp_path, window=window, session_id=session_id,
                         conversation_id=conversation_id, platform=platform, **overrides)
    try:
        yield engine
    finally:
        engine.shutdown()
