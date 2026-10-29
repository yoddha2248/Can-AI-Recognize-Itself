# Negatives ledger

Every negative result reported in the paper, numbered N1–N10, with the date it was recorded.
A negative is any result that goes against the paper's own claims or expectations. That includes
a test that fails to separate a self from a cheat or an outside observer, a control that does not
fail as predicted, and a pre-registered check of the paper's own claims that fails. Negatives are
results, and none is removed after the fact. When a later result changes how a row should be read,
a dated note is appended to that row; the row itself is not rewritten. N8 is reserved for a test
not reported in this paper.

| Row | Script | Result file | Paper |
| --- | --- | --- | --- |
| N1 | `experiments/toy_ladder/vimarsa_tests.py` | `results/toy_ladder/vimarsa_tests.txt` | 4.1 |
| N2 | `experiments/toy_ladder/v4a_tests.py` | `results/toy_ladder/v4a_tests.txt` | 4.2 |
| N3 | `experiments/toy_ladder/v4b_tests.py` | `results/toy_ladder/v4b_tests.txt` | 4.3 |
| N4 | `experiments/toy_ladder/v4c_tests.py` | `results/toy_ladder/v4c_tests.txt` | 4.4 |
| N5 | `experiments/transformer/b1_pilot.py` | `results/transformer/b1_results.json` | 4.5 |
| N6 | `experiments/transformer/b2_gpu.py` (and `b2_truncation_check.py`, which did not overturn it) | `results/transformer/b2_results.json`, `b2_truncation_check.json` | 4.6, 7.8 |
| N7 | `experiments/transformer/j1_jspace_self.py` | `results/transformer/j1_results.json` | 4.7 |
| N9 | `experiments/controls/negative_controls.py` | `results/controls/negative_controls.json` | 3.3, 4.8 |
| N10 | `experiments/toy_ladder/v4c_learned_rule.py` | `results/toy_ladder/v4c_learned_rule.txt` (pre-registered); `v4c_learned_rule_posthoc_robust.txt` (post-hoc); superseded runs in `results/superseded/` | 4.4 |

## Ledger

**N1 (2026-10-21).** V1–V3, as drafted, are passed by a toy oscillator with one added variable, so
they cannot distinguish vimarśa (reflexive self-awareness) from self-bookkeeping.

**N2 (2026-10-21).** V4a (distributed self) is passed by replicated copies and by 3-of-21 secret
sharing, so no storage-structure test can separate self-modelling from bookkeeping.

**N3 (2026-10-22).** V4b (novel self-prediction) is passed by an outsider fitting the same equations
from ring phases alone, to within 0.5–4.6% of null error. It detects generative models, not selfhood.
*Note 2026-10-28:* stated exactly, the outsider recovered the coupling constants from ring phases,
was told each disturbance (including the size of any anchor shift) and simulated the response
without the reflexive update.

**N4 (2026-10-22).** V4c (re-recognition against a twin) reduces to continuity of a surviving
variable. An outsider with the learned update rule matches the self on all 600 trials, and both fall
to chance when the self-variable is disrupted. *Note 2026-10-28:* the outsider in `v4c_tests.py` was
given the update rule, not shown to learn it; see N10.

**N5 (2026-10-22).** Distilling DistilGPT-2 on 400 target continuations for 3 epochs did not move it
toward the target (fluent AUROC 0.550 → 0.555; rank correlation 0.964 → 0.958). So the "however
well trained" arm of the B2 kill-check (see `PREREGISTRATION.md`) is untested, not passed.

**N6 (2026-10-23).** Outsiders that see only outputs trail the self by a small but reliable margin
(fluent AUROC 0.031–0.059 lower, with paired intervals excluding zero). A 4× larger outsider did not
close the gap, and distillation on 4,000 target continuations widened it. This goes against the
strong form of the symmetry thesis, that outsiders match; the full-access form stands.
*Note 2026-10-28:* the edge survives removing insertions outside the target's top 50 (target 0.617;
gaps +0.030 to +0.056, intervals above zero; pre-registered check, `b2_truncation_check.json`).

**N7 (2026-10-23).** The J-space workspace carries no self-specific signal. Probes read own and
inserted tokens apart no better from top-k J-space than from random or low-gain directions.
Fluent insertions cannot be read from the middle-layer state in any subspace. The target's edge
over outsiders is not located in J-space (J1).

**N9 (2026-10-26).** The negative controls do not all fail V1–V3. A scrambled-weights GRU with no
language passes V3 given one copied variable (1.00; exactly 0.5 without it). It also passes V1
(readout R² 0.963, causal ratio 1.146), while the trained GRU fails V1 (R² −53 across
trajectories, ratio 0.95). So V1–V3 need neither training nor language: an untrained recurrent
network plus a record passes them.

**N10 (2026-10-28).** An outsider that fits V4c's reflexive update rule from ring phases alone does
not reach the 0.99 agreement criterion. It used three training runs and the pre-registered
least-squares fit. The period was recovered exactly and the rate 1.7% low; agreement with the self
was 566/600 (0.84 when the anchor survives). A post-hoc robust fit on the same data recovers the
rate exactly and agrees on 600/600, so the shortfall lies in the estimator, not the information.
The paper rests no claim on the post-hoc fit.
