"""Fast checks of the packaged Spanda Self-Model (seconds)."""
import numpy as np

import spanda_self_model as ssm


def test_parameters():
    assert (ssm.K, ssm.N, ssm.dt, ssm.CENTER) == (3, 7, 0.01, 3)
    assert (ssm.J_intra, ssm.J_inter, ssm.gamma) == (0.3, 0.4, 0.8)
    assert (ssm.vimarsa_rate, ssm.vimarsa_every) == (0.02, 50)
    A = [i for i in range(7) if i != ssm.CENTER]
    assert np.allclose(ssm.JEFF[np.ix_(A, A)][~np.eye(6, dtype=bool)], 1.3)   # A-A
    assert np.allclose(ssm.JEFF[ssm.CENTER, A], 0.5) and np.allclose(ssm.JEFF[A, ssm.CENTER], 0.5)  # A-Omega
    assert np.all(np.diag(ssm.JEFF) == 0)


def test_integrates_and_is_deterministic():
    a = ssm.Sim(0, "full").run(500)
    b = ssm.Sim(0, "full").run(500)
    assert np.all(np.isfinite(a.theta)) and np.all(np.isfinite(a.p))
    assert np.array_equal(a.theta, b.theta) and a.anchor == b.anchor
    assert a.efcopy == a.anchor                         # the efference copy tracks own updates


def test_energy_drift_matches_committed():
    # results/toy_ladder/vimarsa_tests.txt: "energy drift (3000 steps, loop off): 2.21e-04"
    th, p, anc = ssm.init(0)
    E0, drift = ssm.energy(th, p, anc), 0.0
    for _ in range(3000):
        th, p = ssm.step(th, p, anc)
        drift = max(drift, abs(ssm.energy(th, p, anc) - E0))
    assert f"{drift:.2e}" == "2.21e-04"


def test_cut_mode_freezes_anchor():
    s = ssm.Sim(3, "cut").run(1000)
    assert s.anchor == np.pi


def test_v3_no_record_state_difference_is_exactly_zero():
    """Self-made and injected anchor changes leave bit-identical states (N1, first half)."""
    for seed in range(3):
        base = ssm.Sim(seed, "full").run(ssm.BURN)
        while base.t % ssm.vimarsa_every != 0:
            base.run(1)
        d = ssm.vimarsa_delta(base.theta, base.anchor)
        S, X = base.copy(), base.copy()
        S.anchor = S.anchor + d; S.efcopy = S.anchor        # the system updates itself
        X.anchor = X.anchor + d                             # an experimenter writes the same value
        assert S.efcopy != X.efcopy                         # only the record differs (N1, second half)
        S.t += 1; X.t += 1
        S.run(500); X.run(500)
        assert np.array_equal(S.theta, X.theta) and np.array_equal(S.p, X.p) and S.anchor == X.anchor
