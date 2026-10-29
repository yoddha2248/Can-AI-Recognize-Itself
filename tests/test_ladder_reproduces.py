"""Rerun the full toy ladder and require byte-identical output to results/toy_ladder/ (slow)."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.slow
@pytest.mark.parametrize("name", ["vimarsa_tests", "v4a_tests", "v4b_tests", "v4c_tests"])
def test_toy_script_reproduces_committed_output(name, tmp_path):
    out = tmp_path / f"{name}.txt"
    subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "toy_ladder", f"{name}.py"), "--out", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    committed = open(os.path.join(ROOT, "results", "toy_ladder", f"{name}.txt"), encoding="utf-8").read()
    assert out.read_text(encoding="utf-8") == committed


@pytest.mark.slow
@pytest.mark.parametrize("robust,committed", [(False, "v4c_learned_rule.txt"),
                                              (True, "v4c_learned_rule_posthoc_robust.txt")])
def test_learned_rule_reproduces_committed_output(robust, committed, tmp_path):
    out = tmp_path / committed
    cmd = [sys.executable, os.path.join(ROOT, "experiments", "toy_ladder", "v4c_learned_rule.py"), "--out", str(out)]
    subprocess.run(cmd + (["--robust"] if robust else []), check=True, stdout=subprocess.DEVNULL)
    ref = open(os.path.join(ROOT, "results", "toy_ladder", committed), encoding="utf-8").read()
    assert out.read_text(encoding="utf-8") == ref


@pytest.mark.parametrize("name", ["vimarsa_tests", "v4b_tests", "v4c_tests"])
def test_toy_smoke_runs(name, tmp_path):
    subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "toy_ladder", f"{name}.py"), "--smoke",
                    "--out", str(tmp_path / "o.txt")], check=True, stdout=subprocess.DEVNULL)
    assert (tmp_path / "o.txt").stat().st_size > 0
