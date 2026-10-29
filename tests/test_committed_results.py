"""The committed result files carry the headline numbers the paper cites (guards against drift)."""
import json
import os

import numpy as np

from b2_items import load_b2_state

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TP = os.path.join(ROOT, "results", "transformer")


def _j(*p):
    return json.load(open(os.path.join(ROOT, "results", *p)))


def test_b1():
    r = _j("transformer", "b1_results.json")
    assert r["part1"]["fed"] == 0.0


def test_b2():
    t = {row["scorer"]: row for row in _j("transformer", "b2_results.json")["table"]}
    assert t["target"]["auroc_fluent"].startswith("0.637") and t["copy"]["auroc_fluent"].startswith("0.637")
    assert t["copy"]["self_minus_this_fluent"].startswith("+0.000")
    outs = [k for k in t if k not in ("target", "copy")]
    auc = [float(t[k]["auroc_fluent"].split()[0]) for k in outs]
    gaps = [float(t[k]["self_minus_this_fluent"].split()[0]) for k in outs]
    lows = [float(t[k]["self_minus_this_fluent"].split()[1].strip("[")) for k in outs]
    assert (min(auc), max(auc)) == (0.577, 0.606)
    assert (min(gaps), max(gaps)) == (0.031, 0.059)
    assert min(lows) > 0                                 # every paired interval excludes zero


def test_b2_items_match_scores():
    st = load_b2_state(os.path.join(TP, "b2_items.json.gz"))
    z = np.load(os.path.join(TP, "b2_scores.npz"))
    kinds = np.array([i["kind"] for i in st["items"]])
    assert len(st["items"]) == 1629 and len(st["train_txt"]) == 4000
    assert np.array_equal(kinds, z["kinds"])
    assert np.array_equal(np.array([i["k"] for i in st["items"]]), z["groups"])
    assert {k: int((kinds == k).sum()) for k in ("self", "fluent", "offctx")} == {"self": 600, "fluent": 434, "offctx": 595}


def test_j1():
    r = _j("transformer", "j1_results.json")
    rows = {(x["kind"], x["model"], x["subspace"]): x for x in r["table"]}
    shuf = [x["auroc"] for x in r["table"] if x["subspace"] == "shuffled"]
    assert 0.46 <= min(shuf) and max(shuf) <= 0.54                       # K3 leakage control
    for m in ("target gpt2-medium", "gpt2-small", "neo-125m"):           # K1: bottom beats top, off-context
        assert rows[("offctx", m, "J-bottom32")]["auroc"] > rows[("offctx", m, "J-top32")]["auroc"]
    assert rows[("offctx", "target gpt2-medium", "J-top128")]["minus_random"].startswith("-0.001")
    fl = [x["auroc"] for x in r["table"] if x["kind"] == "fluent" and x["subspace"] != "shuffled"]
    assert 0.46 <= min(fl) and max(fl) <= 0.56


def test_negative_controls():
    R = _j("controls", "negative_controls.json")
    pa, pb = R["part_a"], R["part_b"]
    assert pa["target"]["fluent"]["b2_auroc"][0] == 0.637
    assert max(pa["S-all"][k]["b2_auroc"][0] for k in ("fluent", "offctx")) <= 0.493
    assert max(pa["S-all"][k][p] for k in ("fluent", "offctx") for p in ("probe_full", "probe_random128")) <= 0.501
    s, t = pb["scrambled GRU"], pb["trained GRU"]
    assert s["v1a_r2_prev_output"] == 0.963 and s["v3_acc_with_copy"] == 1.0 and s["v3_acc_no_copy"] == 0.5
    assert t["v1a_r2_prev_output"] == -53.416


def test_v4c_learned_rule_n10():
    pre = open(os.path.join(ROOT, "results", "toy_ladder", "v4c_learned_rule.txt"), encoding="utf-8").read()
    rob = open(os.path.join(ROOT, "results", "toy_ladder", "v4c_learned_rule_posthoc_robust.txt"), encoding="utf-8").read()
    assert "total agreeing trials: 566/600" in pre and "kill fires (outsider failed to learn the rule): True" in pre
    assert "eta 0.01966" in pre and "period 50" in pre
    assert rob.startswith("POST-HOC") and "600/600" in rob


def test_b2_truncation_check():
    r = _j("transformer", "b2_truncation_check.json")
    assert (r["n_fluent_in_top50"], r["n_fluent_total"], r["own_in_top50_frac"]) == (411, 434, 1.0)
    rows = {x["scorer"]: x for x in r["rows"]}
    assert rows["target"]["auroc_fluent_in_top50"] == rows["copy"]["auroc_fluent_in_top50"] == 0.617
    outs = [x for k, x in rows.items() if k not in ("target", "copy")]
    assert min(x["target_minus_this_in_top50"][1] for x in outs) > 0 and r["kill_fires"] is False
