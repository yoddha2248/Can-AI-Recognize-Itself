"""Spanda Self-Model: 3 rings x 7 Kuramoto-XY oscillators with an anchor self-variable.

    import spanda_self_model as ssm
    sim = ssm.Sim(seed=0, mode="full").run(ssm.BURN)

See spanda_self_model/model.py (or docs/MODEL.md) for the equations and parameters, and
spanda_self_model/observers.py for the outsider of tests V4b and V4c.
"""
from .model import (N, K, dt, CENTER, J_intra, J_inter, gamma, vimarsa_rate, vimarsa_every,
                    symbols, comp, JEFF, BURN, POST, SEEDS, DELTA,
                    dH, step, energy, mean_phase, vimarsa_delta, wrap, init, Sim)

__version__ = "1.0.0"
__all__ = ["N", "K", "dt", "CENTER", "J_intra", "J_inter", "gamma", "vimarsa_rate",
           "vimarsa_every", "symbols", "comp", "JEFF", "BURN", "POST", "SEEDS", "DELTA",
           "dH", "step", "energy", "mean_phase", "vimarsa_delta", "wrap", "init", "Sim"]
