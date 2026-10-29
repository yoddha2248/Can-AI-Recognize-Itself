"""
V4c with an outsider that LEARNS the reflexive update rule from ring phases.

In v4c_tests.py the "outsider + rule" recogniser was given the rule (rate, period, timing).
Here the outsider sees only ring phases from three training runs on seeds not used in V4c
(500-502, anchor started at a random value so that its jumps are visible). From phases alone it:
  1. reads the anchor at every step (one-step force fit with the coupling constants V4b recovers),
  2. finds the steps where the anchor jumps, and from them the update period and phase,
  3. fits the rate eta by least squares of each jump on |z| wrap(arg z - a_before), choosing
     which phase snapshot (lag -1, 0, +1) the rule reads by the smallest residual.
It then reruns V4c (same seeds, same disruptions, same 600 trials) with its fitted rule.
Default: 3 training runs (seeds 500-502) = the pre-registered test. TRAIN_SEEDS=... runs the
post-hoc, exploratory scaling check (more observation), and ROBUST=1 the post-hoc robust fit
(median of per-jump ratios), both labelled as such in their output names.
Kill (pre-registered 2026-10-28, docs/PREREGISTRATION.md): agreement with the self (log-undo) recogniser < 0.99 at any level,
or fitted eta off by more than 10%.
Output: results/toy_ladder/v4c_learned_rule.txt (pre-registered) or v4c_learned_rule_posthoc_robust.txt (--robust)
Note: the first run (2026-10-28) applied the fitted rule one snapshot late (k = i + 1 + lag); its
outputs are kept in results/superseded/ as *_indexbug.txt. Fixed here: k = i + lag, the same
indexing as the fit.

Paper: section 4.4 (V4c, learned-rule outsider; a follow-up to an independent review).
Negatives ledger: N10 (kill fired).
Expected:
  default (pre-registered)   results/toy_ladder/v4c_learned_rule.txt: period 50, eta 0.01966
                             (1.69% low), agreement 566/600 (rings 0.840), kill fires: True
  --robust (post-hoc)        results/toy_ladder/v4c_learned_rule_posthoc_robust.txt: eta 0.02000,
                             agreement 600/600, kill fires: False. No claim in the paper rests on it.
Run: python experiments/toy_ladder/v4c_learned_rule.py [--robust] [--train-seeds 500,501,...] [--out FILE | --out -]
     (CPU, ~1-2 min)
"""
import argparse, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_ap = argparse.ArgumentParser(description="V4c with an outsider that learns the update rule (N10)")
_ap.add_argument("--robust", action="store_true", help="post-hoc robust fit (same as ROBUST=1)")
_ap.add_argument("--train-seeds", default=None, help="post-hoc scaling check, e.g. 500,501,...,519 (same as TRAIN_SEEDS)")
_ap.add_argument("--out", default=None, help="output file ('-' = print only); default results/toy_ladder/<name>.txt")
_args = _ap.parse_args()
if _args.robust:
    os.environ["ROBUST"] = "1"
if _args.train_seeds:
    os.environ["TRAIN_SEEDS"] = _args.train_seeds
import sys
sys.path.insert(0, ROOT)
import spanda_self_model as vt                                                      # noqa: E402
from spanda_self_model.observers import (run, known_force, infer_anchor,           # noqa: E402
                                         infer_anchor_with_rule, closer, V4C_WINDOW)
wrap, K, N, C, dt = vt.wrap, vt.K, vt.N, vt.CENTER, vt.dt
W_HIST = W_RELAX = V4C_WINDOW
_default_name = os.environ.get("OUT_NAME") or ("v4c_learned_rule_posthoc_robust.txt" if os.environ.get("ROBUST") == "1"
                 else "v4c_learned_rule.txt" if "TRAIN_SEEDS" not in os.environ else "v4c_learned_rule_posthoc_seeds.txt")
OUT = _args.out or os.path.join(ROOT, "results", "toy_ladder", _default_name)


def anchor_at(a, b, c):
    """Anchor read from three consecutive phase snapshots (force at b)."""
    force = -(wrap(c - b) - wrap(b - a)) / dt ** 2
    resid = force - known_force(b)
    A = np.array([[np.sin(b[k, C]), np.cos(b[k, C])] for k in range(K)])
    gc, gs = np.linalg.lstsq(A, resid[:, C], rcond=None)[0]
    return np.arctan2(-gs, gc)


# ---------------- 1-3: learn the rule from phases of training runs ----------------
ROBUST = os.environ.get("ROBUST") == "1"
TRAIN_SEEDS = [int(x) for x in os.environ.get("TRAIN_SEEDS", "500,501,502").split(",")]
rng = np.random.default_rng(11)
jumps, period_votes, phase_votes = [], [], []
for seed in TRAIN_SEEDS:
    th, p, _ = vt.init(seed)
    a0 = rng.uniform(-np.pi, np.pi)
    th, p, a, t, ph, an, _ = run(th, p, a0, 0, 1500)          # ground truth runs; outsider sees ph only
    est = np.array([anchor_at(ph[i], ph[i + 1], ph[i + 2]) for i in range(len(ph) - 2)])  # est[i] = anchor at step i+1
    d = np.abs(wrap(np.diff(est)))
    thr = 10 * np.median(d) + 1e-9
    # a jump in the anchor contaminates the two windows that straddle it; collapse adjacent flags
    flagged = np.where(d > thr)[0]
    events = [f for j, f in enumerate(flagged) if j == 0 or f - flagged[j - 1] > 2]
    gaps = np.diff(events)
    period_votes += list(gaps)
    P = int(np.median(gaps))
    for f in events:
        lo, hi = f - 2, f + 4                                   # clean readings either side
        if lo < 0 or hi >= len(est):
            continue
        jumps.append(dict(seed=seed, f=f, a_before=est[lo], a_after=est[hi], ph=ph))
P_hat = int(np.median(period_votes))

# the update happens between est[f] and est[f+2]; candidate snapshots the rule might read
best = None
for lag in (-1, 0, 1, 2):
    X, Y = [], []
    for j in jumps:
        k = j["f"] + 1 + lag
        if not 0 <= k < len(j["ph"]):
            continue
        z = np.mean(np.exp(1j * j["ph"][k]))
        X.append(np.abs(z) * wrap(np.angle(z) - j["a_before"]))
        Y.append(wrap(j["a_after"] - j["a_before"]))
    X, Y = np.array(X), np.array(Y)
    if ROBUST:   # post-hoc: median of per-jump ratios; lag chosen by their spread (MAD)
        r = Y / X
        eta = float(np.median(r))
        rss = float(np.median(np.abs(r - eta)))
    else:        # pre-registered: least squares over all detected jumps
        eta = float(X @ Y / (X @ X))
        rss = float(np.sum((Y - eta * X) ** 2))
    if best is None or rss < best[2]:
        best = (lag, eta, rss)
lag_hat, eta_hat, rss_hat = best
# phase of the update schedule, from the training runs' event steps (step index of the update)
offs = [((j["f"] + 1) % P_hat) for j in jumps]
off_hat = int(np.bincount(offs).argmax())
learned = dict(period=P_hat, offset=off_hat, eta=eta_hat, lag=lag_hat, rss=rss_hat, n_jumps=len(jumps))


def infer_anchor_learned(phases, t_start):
    """As v4c_tests.infer_anchor_with_rule, but with the outsider's fitted period, phase, lag and eta."""
    times = [t_start + i for i in range(len(phases) - 1)]
    upd = [i for i, t in enumerate(times) if (t % P_hat) == off_hat]
    last = upd[-1] + 2
    a = infer_anchor(phases[last:])
    for i in reversed(upd):
        k = min(max(i + lag_hat, 0), len(phases) - 1)          # same indexing as the fit: k = update step + lag
        z = np.mean(np.exp(1j * phases[k]))
        a_new = a
        for _ in range(50):
            a = a_new - eta_hat * np.abs(z) * wrap(np.angle(z) - a)
    return a


# ---------------- rerun V4c exactly as in v4c_tests.py, adding the learned-rule recogniser ----------------
rng = np.random.default_rng(7)
levels = ["rings", "partial", "total"]
RECS = ["self-efcopy", "outsider+given-rule", "outsider+learned-rule"]
acc = {lv: {r: [] for r in RECS} for lv in levels}
for seed in range(10):
    sims = {}
    for who, s in (("self", seed), ("twin", seed + 100)):
        th, p, a = vt.init(s)
        th, p, a, t, *_ = run(th, p, a, 0, 2000)
        th, p, a, t, ph, an, _ = run(th, p, a, t, W_HIST)
        t_hist = t - W_HIST
        upd = [i for i in range(W_HIST) if (t_hist + i) % vt.vimarsa_every == 0]
        sims[who] = dict(th=th, p=p, a=a, t=t, ph=ph, a_end=an[-1], a_hat_end=infer_anchor(ph[upd[-1] + 1:]))
    S, T = sims["self"], sims["twin"]
    for lv in levels:
        for _ in range(20):
            th = np.mod(S["th"] + rng.normal(0, 1.5, (K, N)), 2 * np.pi)
            a = S["a"]
            if lv == "partial":
                a = a + rng.normal(0, 0.05)
            elif lv == "total":
                a = rng.uniform(-np.pi, np.pi)
            th, p, a_now, t, ph, an, ups = run(th, S["p"], a, S["t"], W_RELAX)
            acc[lv]["self-efcopy"].append(closer(a_now - sum(ups), S["a_end"], T["a_end"]))
            acc[lv]["outsider+given-rule"].append(
                closer(infer_anchor_with_rule(ph, S["t"]), S["a_hat_end"], T["a_hat_end"]))
            acc[lv]["outsider+learned-rule"].append(
                closer(infer_anchor_learned(ph, S["t"]), S["a_hat_end"], T["a_hat_end"]))

lines = [("POST-HOC robust fit. " if ROBUST else "") + f"Learned rule (from phases of {len(TRAIN_SEEDS)} training runs): period {P_hat} (true {vt.vimarsa_every}), "
         f"offset {off_hat}, snapshot lag {lag_hat}, eta {eta_hat:.5f} (true {vt.vimarsa_rate}; "
         f"error {100 * abs(eta_hat / vt.vimarsa_rate - 1):.2f}%), from {len(jumps)} jumps",
         "", "Re-recognition accuracy (200 trials per level; 0.50 = chance)",
         f"{'':10s}" + "".join(f"{r:>24s}" for r in RECS)]
for lv in levels:
    lines.append(f"{lv:10s}" + "".join(f"{np.mean(acc[lv][r]):24.2f}" for r in RECS))
agree = {lv: float(np.mean(np.array(acc[lv]["self-efcopy"]) == np.array(acc[lv]["outsider+learned-rule"])))
         for lv in levels}
lines += ["", "per-trial agreement, self-efcopy vs outsider+learned-rule: " +
          ", ".join(f"{lv} {v:.3f}" for lv, v in agree.items()),
          f"total agreeing trials: {sum(int(round(v * 200)) for v in agree.values())}/600"]
kill = min(agree.values()) < 0.99 or abs(eta_hat / vt.vimarsa_rate - 1) > 0.10
lines.append(f"kill fires (outsider failed to learn the rule): {kill}")
if OUT != "-":
    open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
