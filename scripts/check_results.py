"""
Compare a fresh run with the committed results and print the headline numbers the paper cites.

    python scripts/check_results.py toy      RUN_DIR   # RUN_DIR holds vimarsa_tests.txt, v4a..v4c_tests.txt
    python scripts/check_results.py b1       RUN_DIR   # b1_results.json (full) or b1_part1.json (--smoke)
    python scripts/check_results.py controls RUN_DIR   # negative_controls.json (full or smoke/)
    python scripts/check_results.py b2       RUN_DIR   # b2_results.json
    python scripts/check_results.py j1       RUN_DIR   # j1_results.json
    python scripts/check_results.py truncation RUN_DIR # b2_truncation_check.json
    python scripts/check_results.py committed          # headline numbers from results/ only

Exit status 1 if anything differs from the committed file beyond the stated tolerance.
Differences are reported, never hidden: a mismatch is a finding.
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
TOY = ["vimarsa_tests", "v4a_tests", "v4b_tests", "v4c_tests"]
TOY_EXTRA = ["v4c_learned_rule", "v4c_learned_rule_posthoc_robust"]   # compared when present
ok = True


def fail(msg):
    global ok
    ok = False
    print("  MISMATCH:", msg)


def nums(s):
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?", s)]


# ------------------------------------------------------------------ headline extraction
def toy_headlines(d):
    t = {f: open(os.path.join(d, f + ".txt"), encoding="utf-8").read() for f in TOY}
    v1 = re.search(r"full\s+V1a tracking = ([-+]\d\.\d+)", t["vimarsa_tests"]).group(1)
    v3a = re.search(r"max state difference self vs external = (\S+)", t["vimarsa_tests"]).group(1)
    v3b = re.search(r"prediction error self = (\S+), external = (\S+)", t["vimarsa_tests"])
    sh = re.search(r"^shamir\s+(.*)$", t["v4a_tests"], re.M).group(1).split()
    rp = re.search(r"^replicated\s+(.*)$", t["v4a_tests"], re.M).group(1).split()
    rows = {m.group(1): [float(x) for x in m.group(2).split()]
            for m in re.finditer(r"^([BCD]) .*?\)\s+([\d. ]+)$", t["v4b_tests"], re.M)}
    frac = [r[4] / r[0] for r in rows.values()]
    agree = re.search(r"per-trial agreement.*", t["v4c_tests"]).group(0)
    tot = re.search(r"^total\s+(.*)$", t["v4c_tests"], re.M).group(1).split()
    return [("V1a tracking (full)", v1, "0.037"),
            ("V3 state diff, no record", v3a, "0.00e+00"),
            ("V3 error self / injected, with efference copy", f"{v3b.group(1)} / {v3b.group(2)}", "0.00e+00 / 1.60e-03"),
            ("V4a replicated / Shamir @ 86% ablation", f"{rp[5]} / {sh[5]}", "1.00 / 1.00"),
            ("V4b outsider / null, new types", f"{min(frac):.1%}-{max(frac):.1%}", "0.5%-4.6%"),
            ("V4c agreement outsider+rule vs self", " / ".join(f"{x:.3f}" for x in nums(agree.split(":", 1)[1])),
             "1.000 / 1.000 / 1.000 (600/600)"),
            ("V4c self-efcopy, anchor reset", tot[1], "0.47")]


def controls_headlines(R):
    pa, pb = R["part_a"], R["part_b"]
    return [("Target B2 fluent AUROC (rerun)", pa["target"]["fluent"]["b2_auroc"][0], 0.637),
            ("S-all B2 AUROC max", max(pa["S-all"][k]["b2_auroc"][0] for k in ("fluent", "offctx")), 0.493),
            ("S-all probe AUROC max", max(pa["S-all"][k][p] for k in ("fluent", "offctx") for p in ("probe_full", "probe_random128")), 0.501),
            ("Scrambled GRU V1a R2", pb["scrambled GRU"]["v1a_r2_prev_output"], 0.963),
            ("Scrambled GRU V3 with copy / without", f'{pb["scrambled GRU"]["v3_acc_with_copy"]} / {pb["scrambled GRU"]["v3_acc_no_copy"]}', "1.0 / 0.5"),
            ("Trained GRU V1a R2", pb["trained GRU"]["v1a_r2_prev_output"], -53.416)]


def show(rows, title, label="paper"):
    print(f"\n{title}")
    for name, got, exp in rows:
        print(f"  {name:48s} {str(got):>28s}   ({label}: {exp})")


# ------------------------------------------------------------------ comparisons
def cmp_json_numbers(a, b, tol, path=""):
    """Recursively compare two JSON structures; numbers within tol, strings by their numbers."""
    if isinstance(a, dict):
        for k in b:
            if k == "meta":
                continue
            if k not in a:
                fail(f"{path}/{k} missing in run")
            else:
                cmp_json_numbers(a[k], b[k], tol, f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            fail(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            cmp_json_numbers(x, y, tol, f"{path}[{i}]")
    elif isinstance(b, bool) or b is None:
        if a != b:
            fail(f"{path}: {a} vs committed {b}")
    elif isinstance(b, (int, float)):
        if not isinstance(a, (int, float)) or abs(a - b) > tol:
            fail(f"{path}: {a} vs committed {b}")
    elif isinstance(b, str):
        na, nb = nums(a), nums(b)
        if len(na) != len(nb) or any(abs(x - y) > tol for x, y in zip(na, nb)):
            fail(f"{path}: '{a}' vs committed '{b}'")


def main():
    kind = sys.argv[1]
    d = sys.argv[2] if len(sys.argv) > 2 else None
    if kind == "committed":
        show(toy_headlines(os.path.join(RES, "toy_ladder")), "Toy ladder (committed)")
        show(controls_headlines(json.load(open(os.path.join(RES, "controls", "negative_controls.json")))),
             "Negative controls (committed)")
        return
    if kind == "toy":
        for f in TOY + [x for x in TOY_EXTRA if os.path.exists(os.path.join(d, x + ".txt"))]:
            a = open(os.path.join(d, f + ".txt"), encoding="utf-8").read()
            b = open(os.path.join(RES, "toy_ladder", f + ".txt"), encoding="utf-8").read()
            print(f"  {f}: {'byte-identical to committed' if a == b else 'DIFFERS'}")
            if a != b:
                fail(f"{f}.txt differs from results/toy_ladder/{f}.txt")
        show(toy_headlines(d), "Toy ladder (this run)")
        lr = os.path.join(d, "v4c_learned_rule.txt")
        if os.path.exists(lr):
            t = open(lr, encoding="utf-8").read()
            print("  " + "V4c learned rule (pre-registered)".ljust(48),
                  re.search(r"total agreeing trials: (\S+)", t).group(1).rjust(28), "  (paper: 566/600, N10 kill fires: True)")
    elif kind == "b1":
        ref = json.load(open(os.path.join(RES, "transformer", "b1_results.json")))
        full = os.path.join(d, "b1_results.json")
        run = json.load(open(full if os.path.exists(full) else os.path.join(d, "b1_part1.json")))
        p1 = run["part1"]
        print(f"\nB1 part 1: max |self - fed| = {p1['fed']} (paper: 0.0 exactly); "
              f"max |self - pasted| = {p1['paste']:.2e} (committed {ref['part1']['paste']:.2e})")
        if p1["fed"] != 0.0:
            fail("B1 max |self - fed| is not exactly 0")
        if "table" in run:
            cmp_json_numbers(run["table"], ref["table"], 0.0, "table")
            cmp_json_numbers(run["counts"], ref["counts"], 0, "counts")
            for r in run["table"]:
                print(f"  {r['scorer']:22s} fluent {r['auroc_fluent']:22s} off-context {r['auroc_offctx']}")
    elif kind == "controls":
        smoke = os.path.exists(os.path.join(d, "smoke", "negative_controls.json")) and \
            not os.path.exists(os.path.join(d, "negative_controls.json"))
        sub = "smoke" if smoke else ""
        run = json.load(open(os.path.join(d, sub, "negative_controls.json")))
        ref = json.load(open(os.path.join(RES, "controls", sub, "negative_controls.json")))
        print(f"  comparing with results/controls/{sub + '/' if sub else ''}negative_controls.json (tolerance 0.001)")
        cmp_json_numbers(run, ref, 0.001)
        show(controls_headlines(run), "Negative controls (this run" + (", SMOKE: not paper numbers)" if smoke else ")"),
             "paper, full run" if smoke else "paper")
    elif kind == "truncation":
        run = json.load(open(os.path.join(d, "b2_truncation_check.json")))
        ref = json.load(open(os.path.join(RES, "transformer", "b2_truncation_check.json")))
        cmp_json_numbers(run, ref, 0.001)
        print(f"  fluent insertions inside top 50: {run['n_fluent_in_top50']}/{run['n_fluent_total']} (paper: 411/434); "
              f"kill fires: {run['kill_fires']} (paper: False)")
        for r in run["rows"]:
            print(f"  {r['scorer']:24s} in-top50 AUROC {r['auroc_fluent_in_top50']}  gap {r.get('target_minus_this_in_top50', '')}")
    elif kind in ("b2", "j1"):
        f = f"{kind}_results.json"
        run, ref = json.load(open(os.path.join(d, f))), json.load(open(os.path.join(RES, "transformer", f)))
        cmp_json_numbers(run.get("table"), ref["table"], 0.0, "table")
        if kind == "b2":
            for r in run["table"]:
                print(f"  {r['scorer']:24s} fluent {r['auroc_fluent']:22s} self minus this {r['self_minus_this_fluent']}")
    else:
        raise SystemExit(__doc__)
    print("\nRESULT:", "all checks match the committed results" if ok else "MISMATCHES FOUND (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
