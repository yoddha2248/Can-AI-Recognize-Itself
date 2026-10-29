"""What an observer can do with the Spanda Self-Model: the V4b/V4c outsider and the logged run.

Shared by experiments/toy_ladder/v4c_tests.py and v4c_learned_rule.py (formerly the second
script exec'd the first's source to reuse these functions). The code is unchanged, so both
scripts' outputs stay byte-identical to results/toy_ladder/.

    run(theta, p, anchor, t, n)            ground truth with the reflexive loop; logs phases,
                                           anchors and the system's own anchor updates
    known_force(theta)                     coupling force, known to the outsider from V4b
    infer_anchor(phases)                   outsider: the anchor read from ring phases alone
    infer_anchor_with_rule(phases, t0)     outsider that is also given the update rule and runs
                                           it backwards to the anchor at the start of the window
    closer(x, a, b)                        is x nearer to a than to b on the circle?
"""
import numpy as np

from . import model as vt

K, N, dt, C, wrap = vt.K, vt.N, vt.dt, vt.CENTER, vt.wrap
JEFF = vt.JEFF
V4C_WINDOW = 500          # history and relaxation windows of V4c, in steps


def known_force(theta):
    """Coupling part of the force (couplings known to the outsider from V4b, to 0.001)."""
    g = np.einsum("ij,kij->ki", JEFF, np.sin(theta[:, :, None] - theta[:, None, :]))
    for k in range(K):
        for l in range(K):
            if l != k:
                g[k] += vt.J_inter * np.sin(theta[k] - theta[l])
    return g


def run(theta, p, anchor, t, n):
    """Ground truth with Vimarsa loop; logs phases, anchors and own anchor updates."""
    phases, anchors, updates = [theta], [anchor], []
    for _ in range(n):
        theta, p = vt.step(theta, p, anchor)
        if t % vt.vimarsa_every == 0:
            d = vt.vimarsa_delta(theta, anchor)
            anchor = anchor + d
            updates.append(d)
        phases.append(theta); anchors.append(anchor)
        t += 1
    return theta, p, anchor, t, phases, anchors, updates


def infer_anchor(phases):
    """Outsider: fit gamma*sin(theta_Omega - a) from second differences of phases only."""
    rows, ys = [], []
    for i in range(0, len(phases) - 2, 2):
        a, b, c = phases[i], phases[i + 1], phases[i + 2]
        force = -(wrap(c - b) - wrap(b - a)) / dt ** 2          # = total force at b
        resid = force - known_force(b)
        for k in range(K):
            rows.append([np.sin(b[k, C]), np.cos(b[k, C])]); ys.append(resid[k, C])
    gc, gs = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)[0]
    return np.arctan2(-gs, gc)                                 # gc = g cos a, gs = -g sin a


def infer_anchor_with_rule(phases, t_start):
    """Outsider that has also learned the Vimarsa rule: read the anchor on the final
    constant-anchor segment, then run the rule backwards using the observed ring phases."""
    times = [t_start + i for i in range(len(phases) - 1)]
    upd = [i for i, t in enumerate(times) if t % vt.vimarsa_every == 0]   # step index of updates
    last = upd[-1] + 1                                                    # anchor constant after this
    a = infer_anchor(phases[last:])
    for i in reversed(upd):
        z = np.mean(np.exp(1j * phases[i + 1]))      # rule reads phases right after the step
        a_new = a
        for _ in range(50):                          # invert a_new = a + r|z|wrap(arg z - a)
            a = a_new - vt.vimarsa_rate * np.abs(z) * wrap(np.angle(z) - a)
    return a


def closer(x, a, b):
    """True if x is nearer to a than to b (on the circle)."""
    return abs(wrap(x - a)) < abs(wrap(x - b))
