"""A five-minute tour of the Spanda Self-Model and the paper's central move.

Runs in a few seconds with NumPy only:  python examples/quickstart.py

1. Build the model and run it: 3 rings x 7 oscillators, an anchor (the self-variable) that pulls
   on the rings every step and reads them every 50 steps.
2. Self versus injected change (the idea of V3): an anchor update the system made itself and an
   identical one injected from outside leave identical states, so the system cannot tell them
   apart from its state alone. One stored variable, the efference copy, is enough to tell them apart.
3. The outsider (the idea of V4b/V4c): an observer that sees only the ring phases reads the
   anchor back from them, to within a fraction of a milliradian.

The full tests, with their controls and pre-registered criteria, are in experiments/toy_ladder/.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import spanda_self_model as ssm                                         # noqa: E402
from spanda_self_model.observers import run, infer_anchor               # noqa: E402

# 1. the model ------------------------------------------------------------------------------
sim = ssm.Sim(seed=0, mode="full").run(ssm.BURN)
z = np.mean(np.exp(1j * sim.theta))
print("1. After a 2,000-step burn-in (seed 0)")
print(f"   ring coherence |z| = {abs(z):.3f}, mean ring phase = {np.angle(z):+.3f} rad, "
      f"anchor = {ssm.wrap(sim.anchor):+.3f} rad")

# 2. self versus injected change --------------------------------------------------------------
own = sim.copy()
while own.t % ssm.vimarsa_every != 0:           # advance to the step where the system updates itself
    own.run(1)
inj = own.copy()
before = own.anchor
own.run(1)                                      # the system makes its own update ...
inj.mode = "cut"                                # ... the other copy takes the same step with its loop off
inj.run(1)
inj.anchor = own.anchor                         # and receives the identical change from outside
inj.mode = "full"
diff = max(np.abs(own.theta - inj.theta).max(), np.abs(own.p - inj.p).max())
print("\n2. One own update vs. an identical injected one")
print(f"   anchor moved by {own.anchor - before:+.2e} rad in both runs; largest state difference {diff:.1e}, anchors equal: {own.anchor == inj.anchor}")
print(f"   from its state alone the system cannot tell which happened; its efference copy can:")
print(f"   own run: copy = anchor ({own.efcopy == own.anchor});  injected run: copy = anchor ({inj.efcopy == inj.anchor})")

# 3. the outsider -----------------------------------------------------------------------------
th, p, a = ssm.init(1)
th, p, a, t, *_ = run(th, p, a, 0, 2000)
th, p, a, t, phases, anchors, _ = run(th, p, a, t, 49)   # a stretch with no anchor update
a_hat = infer_anchor(phases[1:])
print("\n3. An outsider that sees only the ring phases (seed 1)")
print(f"   true anchor {ssm.wrap(anchors[-1]):+.5f} rad, read from phases {ssm.wrap(a_hat):+.5f} rad "
      f"(error {abs(ssm.wrap(a_hat - anchors[-1])):.1e} rad)")
print("\nWhat looked like the system's private self-variable is readable from outside.")
print("Next: python reproduce.py toy   (the full ladder, ~4 min, checked against results/)")
