"""
V4c (re-recognition / pratyabhijna) on the Spanda Self-Model.

A system and its twin (identical laws, different seed) each leave a 500-step history.
The system is then disrupted, relaxes for 500 steps, and must say which history was its own.

Disruption levels
  rings     - all ring phases scrambled (sd 1.5 rad); self-anchor untouched
  partial   - rings scrambled and anchor nudged (sd 0.05 rad)
  total     - rings scrambled and anchor reset to a random value

Recognisers
  self-raw     - compares its current anchor with each history's final anchor
  self-efcopy  - first undoes its own logged anchor updates (efference copy), then compares
  outsider     - sees ring phases only; infers each anchor from the phase dynamics
                 (V4b-style fit of the anchor force), then compares
  outsider+rule - outsider that has also learned the vimarsa update rule and runs it backwards

Paper: section 4.4; Figure 6; Tables A1/A2 (row V4c). Negatives ledger: N4.
Expected (results/toy_ladder/v4c_tests.txt): outsider+rule agrees with self-efcopy on 600/600
trials; self-efcopy 1.00 with the anchor intact, 0.47 when it is reset.
Run: python experiments/toy_ladder/v4c_tests.py [--out FILE | --out -] [--smoke]     (CPU, ~45 s)
"""
import numpy as np

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import spanda_self_model as vt
from spanda_self_model.runio import RESULTS, parse
args = parse('V4c re-recognition against a twin', os.path.join(RESULTS, "toy_ladder", "v4c_tests.txt"), smoke=True)
K, N, dt, C, wrap = vt.K, vt.N, vt.dt, vt.CENTER, vt.wrap
W_HIST = W_RELAX = 500
rng = np.random.default_rng(7)
JEFF = vt.JEFF


from spanda_self_model.observers import known_force, run, infer_anchor, infer_anchor_with_rule, closer  # noqa: E402


levels = ["rings", "partial", "total"]
RECS = ["self-raw", "self-efcopy", "outsider", "outsider+rule"]
acc = {lv: {r: [] for r in RECS} for lv in levels}
twin_gap = []

for seed in range(2 if args.smoke else 10):
    # self and twin: identical laws, different seeds; burn in, then record a history window
    sims = {}
    for who, s in (("self", seed), ("twin", seed + 100)):
        th, p, a = vt.init(s)
        th, p, a, t, *_ = run(th, p, a, 0, 2000)
        th, p, a, t, ph, an, _ = run(th, p, a, t, W_HIST)
        t_hist = t - W_HIST
        upd = [i for i in range(W_HIST) if (t_hist + i) % vt.vimarsa_every == 0]
        sims[who] = dict(th=th, p=p, a=a, t=t, ph=ph, a_end=an[-1],
                         a_hat=infer_anchor(ph),
                         a_hat_end=infer_anchor(ph[upd[-1] + 1:]))   # final constant segment
    S, T = sims["self"], sims["twin"]
    twin_gap.append(abs(wrap(S["a_end"] - T["a_end"])))

    for lv in levels:
        for _ in range(20):
            th = np.mod(S["th"] + rng.normal(0, 1.5, (K, N)), 2 * np.pi)
            a = S["a"]
            if lv == "partial":
                a = a + rng.normal(0, 0.05)
            elif lv == "total":
                a = rng.uniform(-np.pi, np.pi)
            th, p, a_now, t, ph, an, ups = run(th, S["p"], a, S["t"], W_RELAX)

            acc[lv]["self-raw"].append(closer(a_now, S["a_end"], T["a_end"]))
            a_back = a_now - sum(ups)                       # undo its own updates
            acc[lv]["self-efcopy"].append(closer(a_back, S["a_end"], T["a_end"]))
            acc[lv]["outsider"].append(closer(infer_anchor(ph), S["a_hat"], T["a_hat"]))
            acc[lv]["outsider+rule"].append(
                closer(infer_anchor_with_rule(ph, S["t"]), S["a_hat_end"], T["a_hat_end"]))

print(f"self-twin anchor gap after burn-in: median {np.median(twin_gap):.4f} rad "
      f"(range {min(twin_gap):.4f}-{max(twin_gap):.4f})")
print("\nRe-recognition accuracy (200 trials per level; 0.50 = chance)")
print(f"{'':10s}" + "".join(f"{r:>15s}" for r in RECS))
for lv in levels:
    print(f"{lv:10s}" + "".join(f"{np.mean(acc[lv][r]):15.2f}" for r in RECS))

# outsider calibration: how well does it read an anchor from phases alone?
errs = []
for seed in range(2 if args.smoke else 10):
    th, p, a = vt.init(seed)
    th, p, a, t, *_ = run(th, p, a, 0, 2000)
    th, p, a, t, ph, an, _ = run(th, p, a, t, W_HIST)
    errs.append(abs(wrap(infer_anchor(ph) - np.mean(an))))
print(f"\noutsider anchor-reading error at rest: median {np.median(errs):.2e} rad")

# per-trial agreement between self-efcopy and outsider+rule
agree = [np.mean(np.array(acc[lv]["self-efcopy"]) == np.array(acc[lv]["outsider+rule"])) for lv in levels]
print("per-trial agreement, self-efcopy vs outsider+rule:", dict(zip(levels, np.round(agree, 3))))
