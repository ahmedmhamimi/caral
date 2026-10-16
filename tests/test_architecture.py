"""
Architecture fitness tests: the layering rules are executable, so they cannot silently rot.

  errors, config            -> may import only each other / stdlib
  domain                    -> may import only domain + errors
  engine                    -> domain, engine, ports, errors, config   (never adapters/app/cli)
  adapters/<x>              -> domain, errors, config, ports, own subpackage (siblings never import each other)
  every module              -> starts with a docstring that lists what it exports
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "caral"


def modules():
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT.parent).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts.pop()
        yield p, ".".join(parts)


def imports_of(path: Path, module: str):
    tree = ast.parse(path.read_text())
    is_pkg = path.name == "__init__.py"
    pkg = module if is_pkg else module.rpartition(".")[0]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names if a.name.startswith("caral")]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target.startswith("caral"):
                out.append(target)
    return out


def layer(module: str) -> str:
    parts = module.split(".")
    if len(parts) == 1:
        return "root"
    if parts[1] in ("domain", "engine"):
        return parts[1]
    if parts[1] == "adapters":
        return "adapters." + parts[2] if len(parts) > 2 else "adapters"
    return "top:" + parts[1]


ALLOWED = {
    "domain": {"caral.domain", "caral.errors"},
    "engine": {"caral.domain", "caral.engine", "caral.ports", "caral.errors", "caral.config"},
}
TOP_LEVEL_FREE = {"top:app", "top:cli", "top:commands", "top:__main__"}


def starts_with_any(target, prefixes):
    return any(target == p or target.startswith(p + ".") for p in prefixes)


@pytest.mark.parametrize("path,module", list(modules()), ids=lambda v: str(v)[-40:])
def test_import_boundaries(path, module):
    lyr = layer(module)
    for target in imports_of(path, module):
        if lyr in ALLOWED:
            assert starts_with_any(target, ALLOWED[lyr]), f"{module} ({lyr}) must not import {target}"
        elif lyr.startswith("adapters."):
            own = "caral.adapters." + lyr.split(".")[1]
            ok = {"caral.domain", "caral.errors", "caral.config", "caral.ports", own}
            assert starts_with_any(target, ok), f"{module} must not import {target} (adapters may not depend on each other or the engine)"
        elif lyr == "root" or module in ("caral.errors", "caral.config", "caral.ports"):
            if module == "caral.config":
                assert starts_with_any(target, {"caral.errors"}), f"{module} must stay a leaf; imports {target}"
            elif module == "caral.errors":
                assert False, "errors must not import anything from caral"
            elif module == "caral.ports":
                assert starts_with_any(target, {"caral.domain"}), f"{module} imports {target}"


def test_lower_layers_never_import_the_app_shell():
    for path, module in modules():
        if layer(module) in ("domain", "engine") or layer(module).startswith("adapters"):
            for target in imports_of(path, module):
                assert not starts_with_any(target, {"caral.app", "caral.cli", "caral.commands"}), f"{module} imports {target}"


def test_every_module_has_a_docstring_with_exports():
    for path, module in modules():
        doc = ast.get_docstring(ast.parse(path.read_text()))
        assert doc and len(doc) > 30, f"{module} needs a module docstring"
        if path.name != "__init__.py" and any(isinstance(n, (ast.ClassDef, ast.FunctionDef)) for n in ast.parse(path.read_text()).body):
            assert "- " in doc, f"{module} docstring should list its public callables"


def test_no_print_in_library_layers():
    """Only the UI adapter, CLI, and doctor may write to the terminal; the engine speaks via events."""
    for path, module in modules():
        if layer(module) in ("domain", "engine") or layer(module) in ("adapters.tools", "adapters.providers", "adapters.store"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
                    pytest.fail(f"{module}:{node.lineno} uses print(); emit an event instead")


def test_domain_is_dependency_free():
    third_party = {"httpx", "requests", "rich"}
    for path, module in modules():
        if layer(module) in ("domain", "engine"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else \
                        [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
                assert not (set(names) & third_party), f"{module} imports third-party {names}"
