# The Spanda Self-Model

A minimal dynamical system with an explicit self-variable, used for the toy test ladder (V1–V4c).
This page is the paper's Appendix B in Markdown. The code is `spanda_self_model/model.py`.
None of the parameters was tuned for any test.

## State

There are 21 phases θ<sub>k,i</sub> and momenta p<sub>k,i</sub>, for rings k = 1, 2, 3 and positions
i = 1, …, 7. Position i = 4 is the ring's centre (index `CENTER = 3` in the code). There is one
more variable, the anchor Ω. Each position carries a symbol: s<sub>i</sub> = A for i ≠ 4, and
s<sub>4</sub> = Ω.

## Hamiltonian

```
H = ½ Σ_{k,i} p_{k,i}²
    − Σ_k Σ_{i<j} J_ij cos(θ_{k,i} − θ_{k,j})              within a ring
    − J_inter Σ_{k<l} Σ_i cos(θ_{k,i} − θ_{l,i})           between rings
    − γ Σ_k cos(θ_{k,4} − Ω)                               the anchor's pull

J_ij = J_intra + c(s_i, s_j),   c(A, A) = 1.0,   c(A, Ω) = 0.2
```

## Parameters

| Quantity | Value | Role |
| --- | --- | --- |
| Rings × oscillators | 3 × 7 | 21 phase oscillators; the centre one of each ring carries Ω |
| J_intra + c(s_i, s_j) | 1.3 (A–A), 0.5 (A–Ω) | Coupling within a ring |
| J_inter | 0.4 | Coupling between corresponding oscillators of different rings |
| γ | 0.8 | Pull of the anchor on each ring's centre oscillator, at every step |
| η (`vimarsa_rate`) | 0.02 | Rate of the reflexive update, weighted by the rings' coherence \|z\| |
| Update period (`vimarsa_every`) | 50 steps (0.5 time units) | How often the anchor reads the rings |
| dt | 0.01 | Störmer–Verlet time step |
| Burn-in (`BURN`) | 2,000 steps | Run before any test begins |

## Dynamics

**Integration.** The model uses Störmer–Verlet with dt = 0.01. Each step is a half step in p,
then a full step in θ (taken mod 2π), then a half step in p. The anchor is held fixed within a
step, so H is conserved to O(dt²) while the loop is off (energy drift 2.2 × 10⁻⁴ over 3,000 steps).

**Reflexive update (the vimarśa loop).** Every 50 steps, after the step, the anchor reads the
rings:

```
z = (1/21) Σ_{k,i} exp(i θ_{k,i})
Ω ← Ω + η |z| wrap(arg z − Ω),      wrap maps an angle into (−π, π]
```

The anchor pulls on the rings at every step with strength 0.8. It reads them only every 50 steps,
at rate 0.02. So the model's self governs far more than it observes. After the rings are rotated
by 0.5 rad, the rings swing about the anchor, while the anchor moves by less than 0.1 rad. This is
Figure B1 of the paper, redrawn by `paper/figures/make_figures.py`.

**Efference copy.** `Sim.efcopy` is one stored variable, set to the new Ω after each update. The
dynamics never read it. It is the "one added variable" of test V3 (ledger N1).

## Initial state and modes

Initially θ<sub>k,i</sub> = 2π b<sub>i</sub> + ε, where b = (0.50, 0.52, 0.53, 0.56, 0.53, 0.52, 0.50)
and ε ~ N(0, 0.1²) is drawn per seed. The momenta start at p = 0 and the anchor at Ω = π.

| `Sim(seed, mode)` | Reflexive update reads | Used for |
| --- | --- | --- |
| `"full"` | the system's own rings | the model as built |
| `"cut"` | nothing (anchor frozen) | negative control |
| `"decoy"` | an independent set of rings (another seed, offset by 1.3 rad, fixed anchor) | negative control |

## Using it

```python
import spanda_self_model as ssm
from spanda_self_model.observers import run, infer_anchor

sim = ssm.Sim(seed=0, mode="full").run(ssm.BURN)   # advance 2,000 steps
print(sim.anchor, sim.efcopy, sim.theta.shape)     # (3, 7)

th, p, a = ssm.init(1)
th, p, a, t, phases, anchors, updates = run(th, p, a, 0, 2049)   # logged ground truth
print(ssm.wrap(infer_anchor(phases[-49:]) - anchors[-1]))   # outsider's error, ~1e-13 rad
```

`examples/quickstart.py` walks through the model, the self-versus-injected test and the outsider
in a few seconds.

The model is deterministic and noise-free, so every toy result is byte-identical across machines
with NumPy 2.x. That is why its "self" predicts itself with zero error; see Honest limits in the
README. A noisy version and a learning version are not yet built.
