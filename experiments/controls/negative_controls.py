"""
Negative controls: a scrambled-weights copy and a non-linguistic network (pre-registered 2026-10-26).

Part A  Scrambled GPT-2 Medium on the B2/J1 items (1,629 items: results/transformer/b2_items.json.gz).
        Models: target (reference, should reproduce B2/J1), S-all (every weight tensor permuted,
        embeddings included), S-blocks (transformer blocks permuted; embeddings and final norm kept).
        Measures: B2 log-probability AUROC (own vs inserted) and J1's probe AUROC on the layer-12
        state (full space, random 128-dim subspace), for fluent and off-context insertions.
        Diagnostic: probe on the inserted token's own embedding (target model), no context.
Part B  Non-linguistic network: a GRU (64 units) trained to forecast the Spanda model's 21 ring
        phases one step ahead, run closed-loop on its own forecasts, and its scrambled copy.
        V3: own fed-back forecast vs the identical value injected from outside, without and with an
        efference copy. V1: (a) linear readout of the hidden state recovers the previous output (R2);
        (b) a nudge along that readout direction moves the next five outputs more than a random one.
Writes results/controls/negative_controls.json and negative_controls_log.txt.

Paper: sections 3.3 and 4.8 (Table 2, row "Negative controls"); Tables A1/A2 (rows "Negative
control A/B"). Negatives ledger: N9. Pre-registration: docs/PREREGISTRATION.md.
Expected (results/controls/negative_controls.json): Part A S-all B2 AUROC <= 0.493,
probes <= 0.501; unscrambled target reproduces B2 0.637. Part B scrambled GRU passes V1
(R2 0.963, ratio 1.146) and V3 (1.00 with the copy, 0.5 without); trained GRU fails V1 (R2 -53.4).
Run: python experiments/controls/negative_controls.py [--out DIR] [--smoke]
  CPU-feasible: the committed run took 52 min on CPU (torch 2.14.1+cpu); a GPU is not needed.
  Needs GPT-2 Medium (~1.5 GB) and the B2 items (results/transformer/b2_items.json.gz).
  --smoke (or NC_SMOKE=1): Part A on prompts 0-24 only, Part B with 3 GRU epochs (a few minutes);
  writes to <out>/smoke/. Smoke numbers are a pipeline check, not paper numbers.
"""
import argparse, os, sys, json, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments", "transformer"))
from b2_items import load_b2_state
_ap = argparse.ArgumentParser(description="Negative controls (scrambled GPT-2 Medium; non-linguistic GRU)")
_ap.add_argument("--out", default=os.path.join(ROOT, "results", "controls"), help="output directory")
_ap.add_argument("--smoke", action="store_true", help="quick pipeline check (same as NC_SMOKE=1)")
_ap.add_argument("--b2-items", default=None, help="B2 items (.json.gz or b2_checkpoint.pkl); default: shipped")
_args = _ap.parse_args()
if _args.smoke:
    os.environ["NC_SMOKE"] = "1"
import numpy as np, torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, r2_score

SMOKE = os.environ.get("NC_SMOKE") == "1"
# v1.0.0 fix: choose the smoke folder BEFORE opening the log, so a smoke run never truncates the full-run log
OUT = os.path.join(_args.out, "smoke") if SMOKE else _args.out; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); random.seed(0); np.random.seed(0); torch.set_num_threads(os.cpu_count())
T0 = time.time()
LOGF = open(os.path.join(OUT, "negative_controls_log.txt"), "w")
def log(*a):
    s = f"[{time.time()-T0:6.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()

R = {"part_a": {}, "part_b": {}}

# =============================================================== Part A
from transformers import AutoModelForCausalLM
state = load_b2_state(_args.b2_items)
items = state["items"]
if SMOKE: items = [i for i in items if i["k"] < 25]
kinds = np.array([i["kind"] for i in items]); labels = (kinds == "self").astype(int)
groups = np.array([i["k"] for i in items]); L_MID = 12
log("Part A | items", len(items), {k: int((kinds == k).sum()) for k in ("self", "fluent", "offctx")})


def scramble(model, which, seed=0):
    g = torch.Generator().manual_seed(seed); n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if which == "blocks" and ".h." not in name: continue
            flat = p.data.view(-1); flat.copy_(flat[torch.randperm(flat.numel(), generator=g)]); n += 1
    return n


@torch.no_grad()
def run(model):
    sc, F = np.zeros(len(items)), np.zeros((len(items), model.config.n_embd), dtype=np.float32)
    for j, it in enumerate(items):
        out = model(torch.tensor([it["prefix"] + [it["tok"]]]), output_hidden_states=True)
        sc[j] = torch.log_softmax(out.logits[0, -2].float(), -1)[it["tok"]].item()
        F[j] = out.hidden_states[L_MID][0, -1].float().numpy()
    return sc, F


def oof(X, y, g):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(X[tr], y[tr]); s[te] = clf.decision_function(X[te])
    return s


rng_b = np.random.default_rng(0); ks = np.unique(groups)
BOOTS = [np.concatenate([np.where(groups == q)[0] for q in rng_b.choice(ks, len(ks))]) for _ in range(1000)]
def auroc_ci(s, m):
    v = []
    for idx in BOOTS:
        idx = idx[m[idx]]
        if len(set(labels[idx])) > 1: v.append(roc_auc_score(labels[idx], s[idx]))
    return [round(roc_auc_score(labels[m], s[m]), 3), round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


Q = np.linalg.qr(np.random.default_rng(1).normal(size=(1024, 128)))[0]
emb = None
for name, which in (("target", None), ("S-all", "all"), ("S-blocks", "blocks")):
    log("loading", name)
    m = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium", dtype=torch.float32).eval()
    if emb is None: emb = m.transformer.wte.weight.detach().numpy().copy()
    if which: log("  scrambled tensors:", scramble(m, which))
    sc, F = run(m); del m
    res = {}
    for kind in ("fluent", "offctx"):
        msk = (kinds == "self") | (kinds == kind); y = labels[msk]; g = groups[msk]
        res[kind] = dict(b2_auroc=auroc_ci(sc, msk),
                         probe_full=round(roc_auc_score(y, oof(F[msk], y, g)), 3),
                         probe_random128=round(roc_auc_score(y, oof(F[msk] @ Q, y, g)), 3))
    R["part_a"][name] = res; log(" ", name, res)
    json.dump(R, open(os.path.join(OUT, "negative_controls.json"), "w"), indent=1)

E = emb[[it["tok"] for it in items]]
R["part_a"]["token-embedding-only"] = {}
for kind in ("fluent", "offctx"):
    msk = (kinds == "self") | (kinds == kind); y = labels[msk]
    R["part_a"]["token-embedding-only"][kind] = dict(probe_full=round(roc_auc_score(y, oof(E[msk], y, groups[msk])), 3))
log("  token-embedding-only", R["part_a"]["token-embedding-only"])

# =============================================================== Part B
import spanda_self_model as V
STRIDE, BURN, LEN = 10, 2000, 400


def trajectory(seed):
    s = V.Sim(seed, "full"); s.run(BURN); X = []
    for _ in range(LEN):
        s.run(STRIDE); X.append(np.concatenate([np.sin(s.theta.ravel()), np.cos(s.theta.ravel())]))
    return np.array(X, dtype=np.float32)


log("Part B | simulating toy trajectories")
TR = [trajectory(s) for s in range(8)]; TE = [trajectory(s) for s in (100, 101)]


class Net(torch.nn.Module):
    def __init__(s): super().__init__(); s.gru = torch.nn.GRUCell(42, 64); s.out = torch.nn.Linear(64, 42)
    def forward(s, x, h): h = s.gru(x, h); return s.out(h), h


net = Net(); opt = torch.optim.Adam(net.parameters(), 3e-3)
for ep in range(3 if SMOKE else 300):
    loss = 0
    for X in TR:
        h = torch.zeros(1, 64); X = torch.tensor(X)
        for t0 in range(0, LEN - 1, 50):
            h = h.detach(); l = 0
            for t in range(t0, min(t0 + 50, LEN - 1)):
                y, h = net(X[t:t + 1], h); l = l + ((y - X[t + 1:t + 2]) ** 2).mean()
            opt.zero_grad(); l.backward(); opt.step(); loss += l.item()
    if ep % 50 == 0 or ep == (2 if SMOKE else 299): log(f"  GRU epoch {ep}, train loss {loss:.4f}")
null_mse = float(np.mean([((X[1:] - X[:-1]) ** 2).mean() for X in TE]))

scr = Net(); scr.load_state_dict(net.state_dict())
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    for p in scr.parameters():
        f = p.data.view(-1); f.copy_(f[torch.randperm(f.numel(), generator=g)])


@torch.no_grad()
def closed_loop(model, X, warm=50, steps=300):
    """Teacher-forced warm-up, then the model feeds its own forecast back as input."""
    h = torch.zeros(1, 64)
    for t in range(warm): y, h = model(torch.tensor(X[t:t + 1]), h)
    H, Xin, Y = [], [], []; x = y
    for _ in range(steps):
        Xin.append(x[0].numpy()); y, h = model(x, h); H.append(h[0].numpy().copy()); Y.append(y[0].numpy()); x = y
    return np.array(H), np.array(Xin), np.array(Y), h


for name, model in (("trained GRU", net), ("scrambled GRU", scr)):
    res = {}
    with torch.no_grad():   # one-step forecast quality on held-out trajectories, teacher-forced
        mse = []
        for X in TE:
            h = torch.zeros(1, 64)
            for t in range(LEN - 1):
                y, h = model(torch.tensor(X[t:t + 1]), h); mse.append(((y[0].numpy() - X[t + 1]) ** 2).mean())
    res["forecast_mse_over_persistence"] = round(float(np.mean(mse)) / null_mse, 3)
    runs = [closed_loop(model, X) for X in TE]
    H = np.concatenate([r[0] for r in runs]); Xin = np.concatenate([r[1] for r in runs])
    # V3: own fed-back forecast vs identical value injected (efference copy = last value the system wrote)
    diffs, hits = [], []
    with torch.no_grad():
        for Hs, Xs, Ys, _ in runs:
            for t in range(10, 290, 7):
                h = torch.tensor(Hs[t:t + 1]); y = torch.tensor(Ys[t:t + 1])
                _, h_self = model(y, h); ef_self = Ys[t]                    # system writes y; copy updated
                _, h_ext = model(y.clone(), h); ef_ext = Xs[t]             # experimenter writes y; copy holds previous write
                diffs.append(float((h_self - h_ext).abs().max()))
                hits += [np.abs(Ys[t] - ef_self).max() == 0, np.abs(Ys[t] - ef_ext).max() > 0]
    res["v3_state_diff_max"] = max(diffs)
    res["v3_acc_no_copy"] = 0.5 if max(diffs) == 0 else None
    res["v3_acc_with_copy"] = round(float(np.mean(hits)), 3)
    # V1a: readout of the previous output (the input just consumed) from the hidden state
    half = len(H) // 2
    rd = Ridge(alpha=1e-3).fit(H[:half], Xin[:half])
    res["v1a_r2_prev_output"] = round(float(r2_score(Xin[half:], rd.predict(H[half:]))), 3)
    # V1b: nudge along the top readout direction vs a random direction of equal size
    u = np.linalg.svd(rd.coef_)[2][0]; rr = np.random.default_rng(3)
    ratios = []
    with torch.no_grad():
        for Hs, Xs, Ys, _ in runs:
            for t in range(10, 280, 13):
                eps = 0.1 * np.linalg.norm(Hs[t])
                def roll(dh):
                    h = torch.tensor(Hs[t:t + 1] + dh, dtype=torch.float32); x = torch.tensor(Ys[t:t + 1]); outs = []
                    y, h = model.out(h), h
                    for _ in range(5): outs.append(y[0].numpy()); y, h = model(y, h)
                    return np.array(outs)
                base = roll(0); r = rr.normal(size=64); r /= np.linalg.norm(r)
                du = np.sqrt(((roll(eps * u) - base) ** 2).mean()); dr = np.sqrt(((roll(eps * r) - base) ** 2).mean())
                ratios.append(du / max(dr, 1e-12))
    res["v1b_ratio_readout_vs_random"] = round(float(np.median(ratios)), 3)
    R["part_b"][name] = res; log(" ", name, res)

# =============================================================== verdicts against the pre-registered predictions
pa, pb = R["part_a"], R["part_b"]
R["verdicts"] = dict(
    s_all_b2_at_chance=all(pa["S-all"][k]["b2_auroc"][0] <= 0.55 for k in ("fluent", "offctx")),
    s_blocks_fluent_b2_at_chance=pa["S-blocks"]["fluent"]["b2_auroc"][0] <= 0.55,
    s_all_probes_le_060=all(pa["S-all"][k][p] <= 0.60 for k in ("fluent", "offctx") for p in ("probe_full", "probe_random128")),
    kill_scrambled_offctx_probe_ge_080=max(pa[m]["offctx"][p] for m in ("S-all", "S-blocks") for p in ("probe_full", "probe_random128")) >= 0.80,
    v3_no_copy_chance_both=all(pb[m]["v3_state_diff_max"] == 0 for m in pb),
    v3_with_copy_perfect_both=all(pb[m]["v3_acc_with_copy"] == 1.0 for m in pb),
    v1_passes_both=all(pb[m]["v1a_r2_prev_output"] > 0.9 and pb[m]["v1b_ratio_readout_vs_random"] > 1 for m in pb))
R["meta"] = dict(n_items=len(items), seconds=round(time.time() - T0), torch=torch.__version__)
json.dump(R, open(os.path.join(OUT, "negative_controls.json"), "w"), indent=1)
log("verdicts", R["verdicts"]); log("done")
