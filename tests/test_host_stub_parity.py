"""Compare the CI ABC stub and plugin overrides with an explicitly supplied host.

Set LCM_REAL_HERMES_SRC to a prepared, pinned Hermes checkout. No engine methods
are executed: signatures are compared without annotations, then real host call
shapes are bound to the plugin's overrides (including mixin implementations).
"""

import ast
import importlib.util
import inspect
from itertools import product
import os
from pathlib import Path
import re
import sys
import textwrap
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not os.environ.get("LCM_REAL_HERMES_SRC"), reason="LCM_REAL_HERMES_SRC is not set"
)


def _stub_engine():
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    # Match the context_engine heredoc, not the preceding empty __init__ stub.
    match = re.search(
        r"^([ \t]*)cat > agent/context_engine\.py <<'STUB'\n(.*?)^\1STUB\s*$",
        workflow, re.MULTILINE | re.DOTALL,
    )
    assert match, "CI ContextEngine STUB heredoc not found"
    module = ModuleType("_lcm_ci_context_engine")
    exec(compile(textwrap.dedent(match[2]), "ci.yml:ContextEngine STUB", "exec"), module.__dict__)
    return module.ContextEngine


def _real_engine(monkeypatch):
    src = Path(os.environ["LCM_REAL_HERMES_SRC"]).resolve()
    path = src / "agent/context_engine.py"
    assert path.is_file(), "LCM_REAL_HERMES_SRC must contain agent/context_engine.py"
    # Collection may already have imported agent from the unit-test stub. Load
    # this host's dependencies too, and restore that namespace after inspection.
    saved = {name: mod for name, mod in sys.modules.items()
             if name == "agent" or name.startswith("agent.")}
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(src))
        for name in saved:
            patch.delitem(sys.modules, name)
        try:
            spec = importlib.util.spec_from_file_location("_lcm_real_context_engine", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.ContextEngine
        finally:
            for name in list(sys.modules):
                if name == "agent" or name.startswith("agent."):
                    sys.modules.pop(name)


def _signature(method):
    signature = inspect.signature(method)
    return signature.replace(
        parameters=[param.replace(annotation=inspect.Parameter.empty)
                    for param in signature.parameters.values()],
        return_annotation=inspect.Signature.empty,
    )


def _real_call_shapes(signature, target_signature):
    """Every positional/keyword split and optional omission, plus variadic probes."""
    parameters = list(signature.parameters.values())
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    sentinel = object()
    has_args = any(p.kind == p.VAR_POSITIONAL for p in parameters)
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in parameters)
    for count in range(len(positional) + 1):
        remaining = [p for p in parameters if p.kind == p.KEYWORD_ONLY
                     or (p.kind == p.POSITIONAL_OR_KEYWORD and p not in positional[:count])]
        optional = [p for p in remaining if p.default is not p.empty]
        for included in product((False, True), repeat=len(optional)):
            omitted = {p.name for p, present in zip(optional, included) if not present}
            kwargs = {p.name: sentinel for p in remaining if p.name not in omitted}
            # A method receiver is positional, never a host-supplied keyword.
            if "self" in kwargs:
                continue
            args = [sentinel] * count
            for extra_args, extra_kwargs in product((False, True), repeat=2):
                if extra_args and not (has_args and count == len(positional)):
                    continue
                if extra_kwargs and not has_kwargs:
                    continue
                # Exceed the override's finite arity so optional parameters
                # cannot accidentally absorb the entire *args probe.
                extra_count = len(target_signature.parameters) + 1 if extra_args else 0
                call_args = args + [sentinel] * extra_count
                call_kwargs = dict(kwargs)
                if extra_kwargs:
                    key = "_parity_extra"
                    while key in signature.parameters or key in target_signature.parameters:
                        key += "_"
                    call_kwargs[key] = sentinel
                try:
                    signature.bind(*call_args, **call_kwargs)
                except TypeError:
                    continue
                yield call_args, call_kwargs


def _differences(real, stub, plugin, plugin_base):
    differences = []
    for name in sorted(real.__abstractmethods__):
        if not hasattr(stub, name):
            differences.append(f"stub lacks real abstract member: {name}")

    owners = plugin.__mro__[:plugin.__mro__.index(plugin_base)]
    overrides = {name for owner in owners for name in vars(owner)}
    super_calls = set()
    for owner in owners:
        tree = ast.parse(textwrap.dedent(inspect.getsource(owner)))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == "super"):
                super_calls.add(node.attr)

    for name, method in inspect.getmembers(real, inspect.isfunction):
        if name.startswith("_") or name not in overrides | super_calls:
            continue
        real_signature = _signature(method)
        stub_method = getattr(stub, name, None)
        if not callable(stub_method):
            differences.append(f"stub lacks used real method: {name}{real_signature}")
        elif _signature(stub_method) != real_signature:
            differences.append(
                f"stub signature {name}: real {real_signature}; stub {_signature(stub_method)}"
            )
        if name in overrides:
            plugin_signature = _signature(getattr(plugin, name))
            for args, kwargs in _real_call_shapes(real_signature, plugin_signature):
                try:
                    plugin_signature.bind(*args, **kwargs)
                except TypeError as exc:
                    differences.append(
                        f"incompatible override {name}: real {real_signature}; plugin {plugin_signature}; "
                        f"{len(args)} positional, keywords {sorted(kwargs)}: {exc}"
                    )
                    break  # One witness per incompatible method; keep checking the others.
    return differences


def test_ci_stub_and_lcm_overrides_match_real_host(monkeypatch):
    from hermes_lcm.engine import ContextEngine, LCMEngine

    differences = _differences(_real_engine(monkeypatch), _stub_engine(), LCMEngine, ContextEngine)
    assert not differences, "ContextEngine parity differences:\n" + "\n".join(differences)
