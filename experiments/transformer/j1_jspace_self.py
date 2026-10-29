"""
J1: is the self/other signal in the model's J-space, and is it privileged there?

J-space (Gurnee et al. 2026) is read off the averaged Jacobian of the final-layer state with
respect to a middle-layer state, J = E[ d h_L[t'] / d h_l[t] ] (summed over t' >= t, averaged over
t and prompts). Its top right-singular directions are the layer-l directions with the most
influence on what the model outputs: the candidate "workspace".

Test (pre-registered 2026-10-23, docs/PREREGISTRATION.md)
  Items: B2's self/other set (own token vs fluent or off-context inserted token, 150 prompts).
  For each model, the layer-l state at the token's position is projected onto
    (a) its own top-k J-space, (b) a random k-dim subspace, (c) its bottom-k Jacobian directions,
    (d) the full space.
  A logistic probe (5-fold, grouped by prompt) reads own vs inserted from each projection.
  Models: target gpt2-medium (the self), outsiders gpt2 small and GPT-Neo 125M.
Kill-checks
  K1 J-space is not special: AUROC(top-k) not above AUROC(random-k) (paired bootstrap CI).
  K2 Workspace is not privileged: target's J-space AUROC minus an outsider's has a CI covering 0.
  K3 Probe leakage: shuffled-label AUROC must sit at 0.5.
Also reports each Jacobian's spectrum (dimensions for 90% of squared singular values, as a
fraction of width), as a side measure; the paper does not use it.

Paper: section 4.7; Figure 8; Tables A1/A2 (row J1). Negatives ledger: N7.
Expected (results/transformer/j1_results.json): K3 shuffled labels 0.46-0.54; off-context
bottom-k beats top-k in every model (0.894-0.930 vs 0.790-0.873); target top128 minus random128
off-context -0.001 [-0.022, +0.019]; fluent 0.49-0.55 in every subspace.
Run (GPU, ~25 min on a laptop GPU): python experiments/transformer/j1_jspace_self.py [--out DIR] [--smoke]
  Inputs: the B2 items, read from results/transformer/b2_items.json.gz (shipped), or from
  b2_checkpoint.pkl if you ran b2_gpu.py yourself, or from the path in J1_B2_CKPT.
  --smoke (or J1_SMOKE=1): GPT-2 Small only, 20 prompts, 4 fitting sequences.
"""
import argparse, os, json, time, pickle, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
sys.path.insert(0, HERE)
from b2_items import load_b2_state
_ap = argparse.ArgumentParser(description="J1: self/other signal in J-space (GPU)")
_ap.add_argument("--out", default=None, help="output directory (default results/transformer[/smoke])")
_ap.add_argument("--smoke", action="store_true", help="tiny pipeline check (same as J1_SMOKE=1)")
_args = _ap.parse_args()
if _args.smoke:
    os.environ["J1_SMOKE"] = "1"
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

SMOKE = os.environ.get("J1_SMOKE") == "1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0); random.seed(0); np.random.seed(0)
OUT = os.path.join(ROOT, "results", "transformer")
B2 = os.environ.get("J1_B2_CKPT")                     # None -> pkl if present, else shipped json.gz
OUT = os.path.join(OUT, "smoke") if SMOKE else OUT
OUT = _args.out or OUT
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

MODELS = [("target gpt2-medium", "openai-community/gpt2-medium"),
          ("gpt2-small", "openai-community/gpt2"),
          ("neo-125m", "EleutherAI/gpt-neo-125m")]
if SMOKE: MODELS = MODELS[1:2]
N_FIT, FIT_LEN, KS, ROWS = (4, 24, [16], 64) if SMOKE else (64, 48, [32, 128], 64)

state = load_b2_state(B2)
items = state["items"]; fit_txt = state["train_txt"]
if SMOKE: items = [i for i in items if i["k"] < 20]
kinds = np.array([i["kind"] for i in items]); labels = (kinds == "self").astype(int)
groups = np.array([i["k"] for i in items])
fit_seqs = [(p + c)[:FIT_LEN] for p, c in fit_txt[:N_FIT]]
log("device", DEV, "| items", len(items), "| fit sequences", len(fit_seqs))
CK = os.path.join(OUT, "j1_checkpoint.pkl")
done = pickle.load(open(CK, "rb")) if os.path.exists(CK) else {}


def load(repo):
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, attn_implementation="eager")
    m = m.to(DEV).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


class Tap:
    """Forward hook on block l-1: its output is h_l (= hidden_states[l]). In 'leaf' mode the
    output is replaced by a detached copy that requires grad, so we can differentiate to it."""
    def __init__(self, block): self.leaf, self.h = False, None; block.register_forward_hook(self)
    def __call__(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.leaf: h = h.detach().requires_grad_(True)
        self.h = h
        if self.leaf: return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def jacobian(model, tap):
    d = model.config.hidden_size; J = torch.zeros(d, d, device=DEV); tap.leaf = True
    for s in fit_seqs:
        x = torch.tensor([s], device=DEV)
        with torch.enable_grad():
            out = model(x, output_hidden_states=True).hidden_states[-1][0]      # (T, d), final layer
            h = tap.h; T = out.shape[0]
            for r0 in range(0, d, ROWS):
                r = torch.arange(r0, min(r0 + ROWS, d), device=DEV)
                G = torch.zeros(len(r), T, d, device=DEV); G[torch.arange(len(r)), :, r] = 1.0
                try:
                    g = torch.autograd.grad(out, h, G, retain_graph=True, is_grads_batched=True)[0]
                except Exception:                                   # fall back to one row at a time
                    g = torch.stack([torch.autograd.grad(out, h, G[i], retain_graph=True)[0]
                                     for i in range(len(r))])
                J[r] += g[:, 0].mean(1)                              # average over source positions t
    tap.leaf = False
    return (J / len(fit_seqs)).cpu().double().numpy()


@torch.no_grad()
def features(model, tap):
    F = []
    for it in items:
        model(torch.tensor([it["prefix"] + [it["tok"]]], device=DEV))
        F.append(tap.h[0, -1].float().cpu().numpy())
    return np.array(F)


def oof_scores(X, y, mask):
    """Out-of-fold probe scores, grouped by prompt so no prompt is in train and test."""
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups[mask]):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(X[tr], y[tr]); s[te] = clf.decision_function(X[te])
    return s


spectra, preds = done.get("spectra", {}), done.get("preds", {})
for name, repo in MODELS:
    if name in spectra: log("skip", name, "(checkpoint)"); continue
    log("loading", name); model = load(repo)
    L = model.config.num_hidden_layers; l = L // 2
    tap = Tap(model.transformer.h[l - 1])
    log(f"  Jacobian at layer {l}/{L}"); J = jacobian(model, tap)
    U, S, Vt = np.linalg.svd(J)
    cum = np.cumsum(S ** 2) / np.sum(S ** 2); d = len(S)
    n90 = int(np.searchsorted(cum, 0.9) + 1)
    spectra[name] = dict(layer=l, width=d, dims90=n90, frac90=round(n90 / d, 3),
                         top_sv=[round(float(v), 3) for v in S[:5]])
    log("  spectrum", spectra[name])
    log("  features"); X = features(model, tap)
    del model, tap; import gc; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    rng = np.random.default_rng(1)
    for kind in ("fluent", "offctx"):
        mask = (kinds == "self") | (kinds == kind); y = labels[mask]
        for k in KS:
            Q = np.linalg.qr(rng.normal(size=(d, k)))[0]
            subs = {f"J-top{k}": Vt[:k].T, f"random{k}": Q, f"J-bottom{k}": Vt[-k:].T}
            for sname, B in subs.items():
                preds[(name, kind, sname)] = oof_scores(X[mask] @ B, y, mask)
        preds[(name, kind, "full")] = oof_scores(X[mask], y, mask)
        Xs = X[mask] @ Vt[:KS[-1]].T
        preds[(name, kind, "shuffled")] = oof_scores(Xs, np.random.default_rng(2).permutation(y), mask)
    done.update(spectra=spectra, preds=preds); pickle.dump(done, open(CK, "wb"))

# ---------------- statistics: paired bootstrap over prompts ----------------
rng = np.random.default_rng(0); rows = []
for kind in ("fluent", "offctx"):
    mask = (kinds == "self") | (kinds == kind); y = labels[mask]; g = groups[mask]; ks = np.unique(g)
    shuf_y = np.random.default_rng(2).permutation(y)        # same permutation as the probe saw
    boots = [np.concatenate([np.where(g == q)[0] for q in rng.choice(ks, len(ks))]) for _ in range(1000)]
    def ci(fa, fb=None):
        v = []
        for idx in boots:
            if len(set(y[idx])) < 2: continue
            a = roc_auc_score(y[idx], fa[idx]); v.append(a - (roc_auc_score(y[idx], fb[idx]) if fb is not None else 0))
        return np.percentile(v, 2.5), np.percentile(v, 97.5)
    names = [n for n, _ in MODELS]
    for name in names:
        for sname in [s for (n, kd, s) in preds if n == name and kd == kind]:
            p = preds[(name, kind, sname)]
            yy = shuf_y if sname == "shuffled" else y
            row = dict(kind=kind, model=name, subspace=sname, auroc=round(roc_auc_score(yy, p), 3))
            if sname.startswith("J-top"):
                k = sname[5:]; lo, hi = ci(p, preds[(name, kind, f"random{k}")])
                row["minus_random"] = f"{roc_auc_score(y, p) - roc_auc_score(y, preds[(name, kind, f'random{k}')]):+.3f} [{lo:+.3f}, {hi:+.3f}]"
                if name != names[0]:
                    ref = preds[(names[0], kind, sname)]; lo, hi = ci(ref, p)
                    row["target_minus_this"] = f"{roc_auc_score(y, ref) - roc_auc_score(y, p):+.3f} [{lo:+.3f}, {hi:+.3f}]"
            rows.append(row)

import pandas as pd
rep = pd.DataFrame(rows).fillna("")
print("\nSpectra:"); [print(" ", n, s) for n, s in spectra.items()]
print("\n" + rep.to_string(index=False), flush=True)
json.dump(dict(spectra=spectra, table=rows, n_items=len(items), n_fit=len(fit_seqs), ks=KS),
          open(os.path.join(OUT, "j1_results.json"), "w"), indent=1)
log("done")
