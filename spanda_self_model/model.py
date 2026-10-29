"""
The Spanda Self-Model: a minimal dynamical system with an explicit self-variable.

Three rings (K = 3) of seven Kuramoto-XY oscillators each (N = 7), 21 phases in all,
with conjugate momenta. The centre oscillator of each ring (index CENTER = 3) carries the
symbol Omega; the other six carry A. A single scalar, the anchor, is the system's
self-variable.

Hamiltonian (energy() below):

    H = 1/2 sum p^2
        - 1/2 sum_k sum_{i,j} JEFF[i,j] cos(theta[k,i] - theta[k,j])      intra-ring
        - 1/2 J_inter sum_{k != l} sum_i cos(theta[k,i] - theta[l,i])    inter-ring
        - gamma sum_k cos(theta[k,CENTER] - anchor)                       anchor pull

    JEFF[i,j] = J_intra + comp[symbol_i, symbol_j]  (diagonal 0)
              = 1.3 between two A nodes, 0.5 between A and Omega.
    J_intra = 0.3, J_inter = 0.4, gamma = 0.8.

Integration: Stormer-Verlet (velocity Verlet, half-kick / drift / half-kick), dt = 0.01,
phases wrapped to [0, 2 pi). The anchor is held fixed during each step.

Vimarsa (self-reading) loop: every 50 steps (vimarsa_every), after the step, the anchor moves
towards the mean phase of the observed oscillators by

    delta = 0.02 * |z| * wrap(arg z - anchor),    z = mean(exp(i theta))

i.e. rate 0.02 times the coherence |z|. The anchor pulls on the rings every step (strength 0.8)
but reads them only every 50 steps (rate 0.02): the self governs far more than it observes.

Efference copy: `Sim.efcopy` stores the last anchor value the system itself set. It is the
"one added variable" of test V3 (negatives ledger N1). The model's dynamics never read it.

Variants (Sim mode):
    full    the vimarsa loop reads the system's own rings (the model as built)
    cut     the vimarsa loop is disabled (anchor frozen)               [negative control]
    decoy   the vimarsa loop reads an independent decoy system         [negative control]

The model is deterministic and noise-free. The numerics are exactly those of
spanda_testbed/vimarsa_tests_lib.py in the predecessor release (Why-AI-Might-be-Conscious- v0.1.0);
this module only adds docstrings and energy() (the energy check formerly inlined in V1-V3).
"""
import copy as _copy

import numpy as np

# ---------------------------------------------------------------- parameters
N, K, dt, CENTER = 7, 3, 0.01, 3          # oscillators per ring, rings, time step, Omega index
J_intra, J_inter, gamma = 0.3, 0.4, 0.8   # intra-ring base, inter-ring, anchor coupling
vimarsa_rate, vimarsa_every = 0.02, 50    # self-reading rate (x coherence), period in steps
symbols = ["A", "A", "A", "Ω", "A", "A", "A"]
comp = {("A", "A"): 1.0, ("A", "Ω"): 0.2, ("Ω", "A"): 0.2, ("Ω", "Ω"): 2.0}
JEFF = np.array([[J_intra + comp[(symbols[i], symbols[j])] for j in range(N)] for i in range(N)])
np.fill_diagonal(JEFF, 0.0)

# default protocol used by the test ladder
BURN, POST, SEEDS, DELTA = 2000, 3000, range(10), 0.5


def dH(theta, anchor):
    """Gradient dH/dtheta, shape (K, N). theta: (K, N) phases; anchor: scalar."""
    diff = theta[:, :, None] - theta[:, None, :]            # [k,i,j]
    g = np.einsum("ij,kij->ki", JEFF, np.sin(diff))          # +J sin (intra)
    for k in range(K):
        for l in range(K):
            if l != k:
                g[k] += J_inter * np.sin(theta[k] - theta[l])  # +J sin (inter)
    g[:, CENTER] += gamma * np.sin(theta[:, CENTER] - anchor)  # +gamma sin (anchor)
    return g


def step(theta, p, anchor):
    """One Stormer-Verlet step of size dt with the anchor held fixed. Returns (theta, p)."""
    p_h = p - dH(theta, anchor) * dt / 2
    theta = np.mod(theta + p_h * dt, 2 * np.pi)
    p = p_h - dH(theta, anchor) * dt / 2
    return theta, p


def energy(theta, p, anchor):
    """Hamiltonian H(theta, p; anchor). Conserved (to O(dt^2)) while the anchor is fixed."""
    diff = theta[:, :, None] - theta[:, None, :]
    V = -0.5 * np.einsum("ij,kij->", JEFF, np.cos(diff))
    V -= 0.5 * sum(J_inter * np.cos(theta[k] - theta[l]).sum() for k in range(K) for l in range(K) if l != k)
    V -= gamma * np.cos(theta[:, CENTER] - anchor).sum()
    return V + 0.5 * (p ** 2).sum()


def mean_phase(theta):
    """Circular mean of all phases."""
    return np.angle(np.mean(np.exp(1j * theta)))


def vimarsa_delta(observed_theta, anchor):
    """Anchor update of the vimarsa loop: 0.02 * coherence * wrapped phase error."""
    z = np.mean(np.exp(1j * observed_theta))
    return vimarsa_rate * np.abs(z) * np.angle(np.exp(1j * (np.angle(z) - anchor)))


def wrap(x):
    """Wrap an angle (or array) to (-pi, pi]."""
    return np.angle(np.exp(1j * x))


def init(seed):
    """Seeded initial state: phases near a common profile (sd 0.1), zero momenta, anchor pi."""
    rng = np.random.default_rng(seed)
    base = np.array([0.5, 0.52, 0.53, 0.56, 0.53, 0.52, 0.5]) * 2 * np.pi
    theta = np.array([base + rng.normal(0, 0.1, N) for _ in range(K)])
    return theta, np.zeros((K, N)), np.pi


class Sim:
    """A running Spanda Self-Model with its efference copy.

    Attributes: theta (K, N), p (K, N), anchor (float), efcopy (float), t (step count),
    mode ('full' | 'cut' | 'decoy'), and the decoy system's state (used only in 'decoy').
    """

    def __init__(self, seed, mode):
        self.mode = mode
        self.theta, self.p, self.anchor = init(seed)
        self.decoy_theta, self.decoy_p, _ = init(seed + 10_000)
        self.decoy_theta = np.mod(self.decoy_theta + 1.3, 2 * np.pi)  # decoy sits elsewhere
        self.efcopy = self.anchor   # predicted anchor = last value the system itself set
        self.t = 0

    def run(self, n):
        """Advance n steps: Verlet step, then (every vimarsa_every steps) the vimarsa update."""
        for _ in range(n):
            self.theta, self.p = step(self.theta, self.p, self.anchor)
            if self.mode == "decoy":
                self.decoy_theta, self.decoy_p = step(self.decoy_theta, self.decoy_p, np.pi)
            if self.t % vimarsa_every == 0 and self.mode != "cut":
                src = self.decoy_theta if self.mode == "decoy" else self.theta
                self.anchor = self.anchor + vimarsa_delta(src, self.anchor)
                self.efcopy = self.anchor
            self.t += 1
        return self

    def copy(self):
        """Deep copy (used to branch matched interventions)."""
        return _copy.deepcopy(self)
