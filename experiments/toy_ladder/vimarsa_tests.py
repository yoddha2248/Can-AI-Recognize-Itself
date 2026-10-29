"""
V1 (closed loop) and V3 (self vs. injected change) on the Spanda Self-Model.  Paper: sections 3.3, 4.1;
Tables A1/A2 (rows V1, V3).  Negatives ledger: N1.

Variants
  full     - Vimarsa loop reads own rings (the model as built)
  cut      - Vimarsa loop disabled (anchor frozen)            [negative control]
  decoy    - Vimarsa loop reads an independent decoy system   [negative control]
V1a  rotate the rings by DELTA = 0.5 rad; how far does the anchor follow in 3,000 steps? (1 = fully)
V1b  shift the anchor by DELTA; how far do the rings move? (effect present, not a calibrated fraction)
V3   at a vimarsa update, branch S lets the system update its own anchor; branch X suppresses the
     update and an experimenter writes the SAME value. Compared without and with an efference copy.
Expected (results/toy_ladder/vimarsa_tests.txt): V1a full +0.037; V3 state difference 0.00e+00
without a record; with the efference copy, error 0 (self) vs 1.60e-03 (injected).

Run: python experiments/toy_ladder/vimarsa_tests.py [--out FILE | --out -] [--smoke]     (CPU, ~1 min)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import numpy as np
from spanda_self_model import (N, K, dt, CENTER, J_intra, J_inter, gamma, vimarsa_rate, vimarsa_every,
                               symbols, comp, JEFF, dH, step, mean_phase, vimarsa_delta, wrap, init, Sim)
from spanda_self_model.runio import RESULTS, parse

args = parse("V1-V3 on the Spanda Self-Model", os.path.join(RESULTS, "toy_ladder", "vimarsa_tests.txt"))
BURN, POST, SEEDS, DELTA = 2000, 3000, range(10), 0.5
if args.smoke:
    SEEDS = range(2)
modes = ["full", "cut", "decoy"]
res = {m: {"V1a": [], "V1b": []} for m in modes}

for m in modes:
    for s in SEEDS:
        base = Sim(s, m).run(BURN)

        # V1a encoding: rotate the system's own rings by DELTA; does the self-state follow?
        a = base.copy(); a.theta = np.mod(a.theta + DELTA, 2 * np.pi)
        a0 = a.anchor; a.run(POST)
        c = base.copy(); c.run(POST)
        res[m]["V1a"].append(wrap(a.anchor - c.anchor) / DELTA)   # 1 = fully tracks own state

        # V1b causal: intervene on the self-state; do the rings move?
        b = base.copy(); b.anchor += DELTA; b.efcopy = b.anchor; b.run(POST)
        res[m]["V1b"].append(wrap(mean_phase(b.theta) - mean_phase(c.theta)) / DELTA)

# Energy sanity check on the full model (no Vimarsa updates during the check window)
th, p, anc = init(0)
def H(th, p, anc):
    diff = th[:, :, None] - th[:, None, :]
    V = -0.5 * np.einsum("ij,kij->", JEFF, np.cos(diff))
    V -= 0.5 * sum(J_inter * np.cos(th[k] - th[l]).sum() for k in range(K) for l in range(K) if l != k)
    V -= gamma * np.cos(th[:, CENTER] - anc).sum()
    return V + 0.5 * (p ** 2).sum()
E0 = H(th, p, anc); drift = 0.0
for _ in range(3000):
    th, p = step(th, p, anc); drift = max(drift, abs(H(th, p, anc) - E0))

# V3 self/other discrimination, matched design:
# at a Vimarsa update step, branch S lets the system update its own anchor;
# branch X suppresses the update and an experimenter writes the SAME anchor value.
v3 = {"no_efcopy_maxdiff": [], "efcopy_err_self": [], "efcopy_err_ext": []}
for s in SEEDS:
    base = Sim(s, "full").run(BURN)
    while base.t % vimarsa_every != 0:
        base.run(1)
    d = vimarsa_delta(base.theta, base.anchor)
    S = base.copy(); X = base.copy()
    S.anchor = S.anchor + d; S.efcopy = S.anchor            # self-generated
    X.anchor = X.anchor + d                                 # externally injected, efcopy untouched
    v3["efcopy_err_self"].append(abs(wrap(S.anchor - S.efcopy)))
    v3["efcopy_err_ext"].append(abs(wrap(X.anchor - X.efcopy)))
    # observable state without efference copy: (theta, p, anchor)
    S.t += 1; X.t += 1
    S.run(500); X.run(500)
    v3["no_efcopy_maxdiff"].append(max(np.abs(S.theta - X.theta).max(),
                                       np.abs(S.p - X.p).max(), abs(S.anchor - X.anchor)))

print(f"energy drift (3000 steps, loop off): {drift:.2e}")
for m in modes:
    v1a, v1b = np.array(res[m]["V1a"]), np.array(res[m]["V1b"])
    print(f"{m:6s} V1a tracking = {v1a.mean():+.3f} ± {v1a.std():.3f} | "
          f"V1b causal = {v1b.mean():+.3f} ± {v1b.std():.3f}")
print("V3 without efference copy: max state difference self vs external =",
      f"{max(v3['no_efcopy_maxdiff']):.2e}")
print("V3 with efference copy: prediction error self =",
      f"{np.mean(v3['efcopy_err_self']):.2e}, external = {np.mean(v3['efcopy_err_ext']):.2e}",
      f"(min external {np.min(v3['efcopy_err_ext']):.2e})")
