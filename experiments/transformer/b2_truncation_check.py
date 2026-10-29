"""
B2 truncation check: is B2's inside edge a top-k truncation cue?

The target's own tokens were sampled from its top 50, so they always lie inside its top 50;
inserted tokens (written by GPT-2 Large) may not. This script ranks every B2 token under the
target (GPT-2 Medium, CPU), then recomputes fluent-insertion AUROC for every scorer using only
inserted tokens inside the target's top 50, with paired bootstrap gaps over prompts (2,000).
Kill (pre-registered 2026-10-28, docs/PREREGISTRATION.md): every outsider's gap interval includes zero on the restricted set.

Inputs:  B2 items (results/transformer/b2_items.json.gz, shipped; or b2_checkpoint.pkl from
         your own b2_gpu.py run), results/transformer/b2_scores.npz (all scores)
Output:  results/transformer/b2_truncation_check.json

Paper: section 4.6 (B2) and objection 7.8 (a follow-up to an independent review).
Negatives ledger: none (kill did not fire; N6 stands).
Expected: own tokens 100% inside the target's top 50, fluent insertions 411/434 (94.7%); restricted
to those, target 0.617, copy 0.617, outsiders 0.561-0.587, gaps +0.030 [+0.015, +0.045] to
+0.056 [+0.037, +0.076]; kill fires: False.
Run (CPU, GPT-2 Medium, roughly 10-20 min): python experiments/transformer/b2_truncation_check.py [--out FILE]
"""
import argparse, os, json, sys
import numpy as np, torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
sys.path.insert(0, HERE)
from b2_items import load_b2_state
_ap = argparse.ArgumentParser(description="B2 truncation check: is B2's inside edge a top-k truncation cue?")
_ap.add_argument("--out", default=None, help="output JSON (default results/transformer/b2_truncation_check.json)")
_ap.add_argument("--b2-items", default=None, help="B2 items (.json.gz or b2_checkpoint.pkl); default: shipped")
_args = _ap.parse_args()
from transformers import AutoModelForCausalLM

RES = os.path.join(ROOT, "results", "transformer")
items = load_b2_state(_args.b2_items)["items"]
z = np.load(os.path.join(RES, "b2_scores.npz"), allow_pickle=True)
kinds, groups = z["kinds"], z["groups"]
SCORERS = ["target", "copy", "gpt2_small", "neo_125m", "neo_1_3b",
           "neo_125m_distilled_1ep", "neo_125m_distilled_3ep"]

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium").eval()
ranks = np.zeros(len(items), dtype=int)
cache = {}
with torch.no_grad():
    for n, it in enumerate(items):
        key = tuple(it["prefix"])
        if key not in cache:
            logits = model(torch.tensor([it["prefix"]])).logits[0, -1]
            cache[key] = torch.argsort(logits, descending=True)
        order = cache[key]
        ranks[n] = int((order == it["tok"]).nonzero()[0, 0]) + 1
        if n % 200 == 0:
            print(n, flush=True)

intop = ranks <= 50
fluent, self_ = kinds == "fluent", kinds == "self"
print("own tokens inside top 50:", intop[self_].mean())
print("fluent insertions inside top 50:", intop[fluent].mean(), "of", fluent.sum())


def auroc(score, mask):
    m = mask & (self_ | fluent)
    return roc_auc_score(self_[m].astype(int), score[m])


def boot_gap(a, b, mask, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    prompts = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in prompts}
    out = []
    for _ in range(B):
        pick = rng.choice(prompts, len(prompts), replace=True)
        ix = np.concatenate([idx_by[g] for g in pick])
        m = np.zeros(len(items), bool); w = np.zeros(len(items), int)
        np.add.at(w, ix, 1)
        sel = np.repeat(np.arange(len(items)), w)
        mm = (mask & (self_ | fluent))[sel]
        sel = sel[mm]
        y = self_[sel].astype(int)
        if y.min() == y.max():
            continue
        out.append(roc_auc_score(y, z[a][sel]) - roc_auc_score(y, z[b][sel]))
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


res = {"n_fluent_total": int(fluent.sum()), "n_fluent_in_top50": int((fluent & intop).sum()),
       "own_in_top50_frac": float(intop[self_].mean()), "rows": []}
for s in SCORERS:
    row = {"scorer": s,
           "auroc_fluent_all": round(auroc(z[s], np.ones(len(items), bool)), 3),
           "auroc_fluent_in_top50": round(auroc(z[s], intop | self_), 3)}
    if s != "target":
        g = boot_gap("target", s, intop | self_)
        row["target_minus_this_in_top50"] = [round(v, 3) for v in g]
    res["rows"].append(row)
    print(row, flush=True)

outsiders = [r for r in res["rows"] if r["scorer"] not in ("target", "copy")]
res["kill_fires"] = all(r["target_minus_this_in_top50"][1] <= 0 for r in outsiders)
json.dump(res, open(_args.out or os.path.join(RES, "b2_truncation_check.json"), "w"), indent=1)
print("kill fires (edge is a truncation artefact):", res["kill_fires"])
