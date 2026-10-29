"""The repository is what the README says it is: every path it names exists, every reproduce.py
target points at a real script, and the example and the figure script run."""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def test_readme_paths_exist():
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    paths = set(re.findall(r"`((?:experiments|results|spanda_self_model|scripts|tests|docs|examples|paper)/[^`\s]*)`", text))
    assert paths, "README names no repository paths"
    missing = sorted(p for p in paths if "*" not in p and not os.path.exists(os.path.join(ROOT, p)))
    assert not missing, f"README names paths that do not exist: {missing}"


def test_reproduce_targets_point_at_scripts():
    import reproduce
    for name, (_, _, cmds) in reproduce.TARGETS.items():
        for cmd in cmds:
            assert os.path.exists(os.path.join(ROOT, cmd[0])), f"{name}: {cmd[0]} missing"
    out = subprocess.run([sys.executable, "reproduce.py", "list"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert "toy" in out and "committed" in out


def test_quickstart_runs():
    out = subprocess.run([sys.executable, os.path.join("examples", "quickstart.py")], cwd=ROOT,
                         capture_output=True, text=True, check=True, timeout=60).stdout
    assert "largest state difference 0.0e+00, anchors equal: True" in out
    assert "own run: copy = anchor (True);  injected run: copy = anchor (False)" in out


def test_figures_redraw(tmp_path):
    pytest.importorskip("matplotlib")
    env = dict(os.environ, FIG_STYLE="default")          # no LaTeX needed
    subprocess.run([sys.executable, os.path.join("paper", "figures", "make_figures.py"), "--out", str(tmp_path)],
                   cwd=ROOT, env=env, capture_output=True, check=True, timeout=300)
    made = sorted(os.listdir(tmp_path))
    assert made == sorted(["fig3_glance.pdf", "figA1_v4a.pdf", "figA2_v4b.pdf", "fig3_v4c.pdf", "fig4a_b2.pdf",
                           "fig4b_b2gap.pdf", "fig_j1.pdf", "figB1_run.pdf"]), made
