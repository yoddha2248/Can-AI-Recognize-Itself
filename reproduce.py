"""Reproduce the paper's results on any platform (Linux, macOS, Windows): python reproduce.py TARGET

Every target writes into runs/<target>/ (git-ignored), never over the committed results/, then
compares the run with results/ and prints the headline numbers next to the paper's.
The Makefile calls this script, so `make toy` and `python reproduce.py toy` are the same.

    python reproduce.py list         show the targets with hardware and time
    python reproduce.py toy          V1-V4c and the learned rule (NumPy only, ~4 min)
    python reproduce.py committed    print the headline numbers stored in results/
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RUNS = os.environ.get("RUNS", os.path.join(ROOT, "runs"))
TOY = os.path.join("experiments", "toy_ladder")
TRF = os.path.join("experiments", "transformer")
CTL = os.path.join("experiments", "controls")
CHECK = [os.path.join("scripts", "check_results.py")]


def out(*p):
    return os.path.join(RUNS, *p)


# target -> (description, hardware and time, list of commands)
TARGETS = {
    "toy": ("V1-V4c + learned rule (toy ladder)", "CPU, NumPy only, ~4 min; byte-identical check", [
        [os.path.join(TOY, "vimarsa_tests.py"), "--out", out("toy", "vimarsa_tests.txt")],
        [os.path.join(TOY, "v4a_tests.py"), "--out", out("toy", "v4a_tests.txt")],
        [os.path.join(TOY, "v4b_tests.py"), "--out", out("toy", "v4b_tests.txt")],
        [os.path.join(TOY, "v4c_tests.py"), "--out", out("toy", "v4c_tests.txt")],
        [os.path.join(TOY, "v4c_learned_rule.py"), "--out", out("toy", "v4c_learned_rule.txt")],
        [os.path.join(TOY, "v4c_learned_rule.py"), "--robust", "--out", out("toy", "v4c_learned_rule_posthoc_robust.txt")],
        CHECK + ["toy", out("toy")]]),
    "truncation": ("B2 truncation check (review follow-up)", "CPU, GPT-2 Medium, ~10-20 min", [
        [os.path.join(TRF, "b2_truncation_check.py"), "--out", out("truncation", "b2_truncation_check.json")],
        CHECK + ["truncation", out("truncation")]]),
    "b1-smoke": ("B1 part 1: is there a record of authorship?", "CPU, ~5 min, 1.5 GB download", [
        [os.path.join(TRF, "b1_pilot.py"), "--smoke", "--out", out("b1-smoke")],
        CHECK + ["b1", out("b1-smoke")]]),
    "b1": ("B1 full", "CPU, ~55 min, 4 models (~4.5 GB)", [
        [os.path.join(TRF, "b1_pilot.py"), "--out", out("b1")],
        CHECK + ["b1", out("b1")]]),
    "controls-smoke": ("Negative controls, smoke mode (pipeline check)", "CPU, ~10 min", [
        [os.path.join(CTL, "negative_controls.py"), "--smoke", "--out", out("controls-smoke")],
        CHECK + ["controls", out("controls-smoke")]]),
    "controls": ("Negative controls, full", "CPU ~52 min, or a GPU machine", [
        [os.path.join(CTL, "negative_controls.py"), "--out", out("controls")],
        CHECK + ["controls", out("controls")]]),
    "b2": ("B2 outsiders, paired tests", "GPU (4 GB), ~20 min, 11 GB download", [
        [os.path.join(TRF, "b2_gpu.py"), "--out", out("b2")],
        CHECK + ["b2", out("b2")]]),
    "j1": ("J1 J-space probes (reads the shipped B2 items)", "GPU, ~25 min", [
        [os.path.join(TRF, "j1_jspace_self.py"), "--out", out("j1")],
        CHECK + ["j1", out("j1")]]),
    "figures": ("Redraw the paper's data figures from results/", "CPU, seconds; matplotlib", [
        [os.path.join("paper", "figures", "make_figures.py"), "--out", out("figures")]]),
    "committed": ("Print the headline numbers stored in results/", "instant", [CHECK + ["committed"]]),
}
GROUPS = {"gpu": ["b2", "j1", "controls"]}


def run(target):
    if target in GROUPS:
        for t in GROUPS[target]:
            run(t)
        return
    desc, _, cmds = TARGETS[target]
    print(f"== {target}: {desc}", flush=True)
    os.makedirs(out(target), exist_ok=True)
    for cmd in cmds:
        quiet = target == "toy" and cmd[0].startswith(TOY)
        r = subprocess.run([PY] + cmd, cwd=ROOT, stdout=subprocess.DEVNULL if quiet else None)
        if r.returncode:
            sys.exit(f"failed ({r.returncode}): {' '.join(cmd)}")


def main(argv):
    if not argv or argv[0] in ("list", "-h", "--help", "help"):
        print(__doc__)
        w = max(map(len, TARGETS))
        for k, (d, hw, _) in TARGETS.items():
            print(f"  {k:<{w}}  {d}  [{hw}]")
        print(f"  {'gpu':<{w}}  b2 + j1 + controls")
        return
    for t in argv:
        if t not in TARGETS and t not in GROUPS:
            sys.exit(f"unknown target {t!r}; try: python reproduce.py list")
        run(t)


if __name__ == "__main__":
    main(sys.argv[1:])
