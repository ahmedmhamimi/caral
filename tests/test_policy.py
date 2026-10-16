"""
Tests for the permission rule chain: every mode, rule ordering, extension via add_rule, and the
credential/sudo protections that hold even in yolo mode.
"""

import pytest

from caral.adapters.tools import default_registry
from caral.config import Config
from caral.domain.tools import ToolContext
from caral.engine.permissions import Check, Decision, PermissionManager

REG = default_registry()


def setup(tmp_path, mode, **cfg):
    config = Config(**cfg)
    return PermissionManager(mode, tmp_path, config), ToolContext(cwd=tmp_path, config=config)


def chk(pm, ctx, tool, **args):
    return pm.check(REG.get(tool), args, ctx).decision


def test_reads_always_allowed_except_credentials(tmp_path):
    pm, c = setup(tmp_path, "ask")
    assert chk(pm, c, "read_file", path="a.py") == Decision.ALLOW
    assert chk(pm, c, "grep", pattern="x") == Decision.ALLOW
    assert chk(pm, c, "read_file", path="~/.ssh/id_rsa") == Decision.ASK
    assert chk(pm, c, "read_file", path=".env") == Decision.ASK
    assert chk(pm, c, "read_file", path=".env.example") == Decision.ALLOW
    pm, c = setup(tmp_path, "yolo")
    assert chk(pm, c, "read_file", path=".env") == Decision.ALLOW


def test_symlink_into_credentials_is_caught(tmp_path):
    secret = tmp_path / ".aws"
    secret.mkdir()
    (secret / "credentials").write_text("x")
    (tmp_path / "link").symlink_to(secret / "credentials")
    pm, c = setup(tmp_path, "ask")
    assert chk(pm, c, "read_file", path="link") == Decision.ASK


def test_edit_modes(tmp_path):
    inside, outside = {"path": "a.py"}, {"path": "/tmp/elsewhere.py"}
    pm, c = setup(tmp_path, "ask")
    assert chk(pm, c, "edit_file", **inside) == Decision.ASK
    pm.remember(REG.get("edit_file"), inside, c)
    assert chk(pm, c, "edit_file", **inside) == Decision.ALLOW
    assert chk(pm, c, "edit_file", **outside) == Decision.ASK      # outside root still asks after "always"
    pm, c = setup(tmp_path, "auto-edit")
    assert chk(pm, c, "write_file", **inside, content="x") == Decision.ALLOW
    assert chk(pm, c, "write_file", **outside, content="x") == Decision.ASK
    pm, c = setup(tmp_path, "yolo")
    assert chk(pm, c, "edit_file", **outside) == Decision.ALLOW
    pm, c = setup(tmp_path, "plan")
    d = pm.check(REG.get("edit_file"), inside, c)
    assert d.decision == Decision.DENY and "read-only" in d.reason


def test_symlink_escape_from_project_is_outside(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (proj / "esc").symlink_to(out)
    pm, c = setup(proj, "auto-edit")
    assert chk(pm, c, "write_file", path="esc/x.txt", content="x") == Decision.ASK


@pytest.mark.parametrize("mode", ["ask", "auto-edit", "yolo", "plan"])
def test_hard_denials_hold_in_every_mode(tmp_path, mode):
    pm, c = setup(tmp_path, mode)
    assert chk(pm, c, "write_file", path="~/.ssh/authorized_keys", content="x") == Decision.DENY
    assert chk(pm, c, "edit_file", path=".git/hooks/pre-commit", old_string="a", new_string="b") == Decision.DENY
    assert chk(pm, c, "write_file", path=".env", content="x") == Decision.DENY
    assert chk(pm, c, "bash", command="rm -rf /") == Decision.DENY
    assert chk(pm, c, "bash", command="ls & rm -rf /") == Decision.DENY


def test_bash_modes(tmp_path):
    pm, c = setup(tmp_path, "ask", allow_commands=["npm test"], deny_commands=["git push"])
    assert chk(pm, c, "bash", command="ls") == Decision.ALLOW
    assert chk(pm, c, "bash", command="make build") == Decision.ASK
    assert chk(pm, c, "bash", command="npm test") == Decision.ALLOW
    assert chk(pm, c, "bash", command="npm test && rm x") == Decision.ASK
    assert chk(pm, c, "bash", command="git push origin main") == Decision.DENY
    pm.remember(REG.get("bash"), {"command": "make build"}, c)
    assert chk(pm, c, "bash", command="make build") == Decision.ALLOW
    pm, c = setup(tmp_path, "yolo")
    assert chk(pm, c, "bash", command="make") == Decision.ALLOW
    pm, c = setup(tmp_path, "plan")
    assert chk(pm, c, "bash", command="make") == Decision.DENY
    assert chk(pm, c, "bash", command="git log") == Decision.ALLOW


def test_sudo_and_credential_commands_ask_even_in_yolo(tmp_path):
    pm, c = setup(tmp_path, "yolo")
    assert chk(pm, c, "bash", command="sudo apt install x") == Decision.ASK
    assert chk(pm, c, "bash", command="echo k >> ~/.ssh/authorized_keys") == Decision.ASK
    assert chk(pm, c, "bash", command="cat ~/.aws/credentials") == Decision.ALLOW  # safe read in yolo is the user's call


def test_interpreter_allow_rules_are_exact(tmp_path):
    pm, c = setup(tmp_path, "ask")
    bash = REG.get("bash")
    pm.remember(bash, {"command": "python build.py"}, c)
    assert chk(pm, c, "bash", command="python build.py") == Decision.ALLOW
    assert chk(pm, c, "bash", command="python -c 'import os; os.system(\"x\")'") == Decision.ASK


def test_net_hosts(tmp_path):
    pm, c = setup(tmp_path, "ask")
    web = REG.get("web_fetch")
    a = {"url": "https://example.com/page"}
    assert pm.check(web, a, c).decision == Decision.ASK
    pm.remember(web, a, c)
    assert chk(pm, c, "web_fetch", url="https://example.com/other") == Decision.ALLOW
    assert chk(pm, c, "web_fetch", url="https://evil.com/") == Decision.ASK
    pm, c = setup(tmp_path, "plan")
    assert chk(pm, c, "web_fetch", url="https://example.com") == Decision.DENY


def test_custom_rule_can_be_inserted_without_touching_core(tmp_path):
    pm, c = setup(tmp_path, "yolo")
    pm.add_rule(lambda r: Check(Decision.DENY, "no fetching in CI") if r.category == "net" else None, index=0)
    assert chk(pm, c, "web_fetch", url="https://example.com") == Decision.DENY
    assert chk(pm, c, "bash", command="ls") == Decision.ALLOW


def test_set_mode_validates(tmp_path):
    pm, _ = setup(tmp_path, "ask")
    pm.set_mode("plan")
    assert pm.mode == "plan"
    with pytest.raises(ValueError):
        pm.set_mode("wild")
