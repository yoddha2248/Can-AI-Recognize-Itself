# Pre-registration record

Each criterion and prediction below was written down **before** the run it governs, and is quoted
here word for word from the project's dated working log. Only the criteria and predictions are
quoted. Surrounding discussion and material unrelated to this paper are omitted, and the log's
internal labels and file paths are omitted or replaced in square brackets. Outcomes are added only
in the notes marked *Outcome*.

**What this is and is not.** These are entries in the author's own dated log, not a registration
with a third-party registry such as OSF, so their dates are self-recorded. They are published so
that readers can compare each criterion with the result reported in the paper and in
[`NEGATIVES.md`](NEGATIVES.md).

## What was fixed in advance, test by test

| Test | Recorded before the run | Date |
| --- | --- | --- |
| V1–V3 | Built-in kill-check; predictions for the positive control; both negative controls named | 2026-10-21 |
| V4a–V4c | The cheapest cheat for each test, and the prediction that the model fails all three as built | 2026-10-22 |
| V4a–V4c numerical thresholds | **Not** recorded in advance. The pass thresholds (e.g. agreement ≥ 0.99 in V4c) were written after the runs | — |
| B2 | Kill-check: an outsider that sees only outputs stays far behind however well trained | 2026-10-22 |
| J1 | Kill-checks K1, K2, K3 | 2026-10-23 |
| Negative controls | Predictions and kill for Parts A and B | 2026-10-26 |
| V4c with a learned rule; B2 truncation check | Kill criteria, written after an independent review and before either run | 2026-10-28 |

---

## V1–V3: controls and built-in kill-check (2026-10-21)

> **Negative controls:** an AlphaFold-style non-linguistic network and a scrambled-weights copy.
> Both should fail V1–V3.
>
> **Positive control:** the Spanda Self-Model (3-ring Kuramoto-XY with Vimarśa feedback), which has
> a self-loop by construction and a stable self-world gap of 0.078 rad. It should pass V1 and V3.
>
> **Built-in kill-check:** if the toy model passes, V1–V3 measure the architecture of
> self-reference, not consciousness. They can then be necessary-condition evidence at most, never
> sufficient. We state this before running anything.

*Outcome:* the toy passed V1 and V3, so the kill-check applies (N1). The AlphaFold-style network
was out of reach; a small recurrent network forecasting the ring phases stood in for it (see
"Negative controls" below).

## V4a–V4c: cheats and predictions (2026-10-22)

> | Test | Requirement | Why a single efference-copy variable fails | Cheapest cheater to red-team it |
> | --- | --- | --- | --- |
> | V4a Distributed self | Self/other discrimination survives random ablation of any k% of units, degrading gracefully | One variable is a single point of failure: ablate it and V3 drops to chance | Copy the efference variable onto every node. If that passes, V4a is still bookkeeping. |
> | V4b Novel self-prediction | The system predicts its own response to perturbation types it was never built for, better than an outside predictor with the same data | A record of past updates says nothing about responses to new kinds of disturbance | An outside model fitted to the system's logs. The self-model must beat it, not merely match it. |
> | V4c Re-recognition (pratyabhijñā) | After a large disruption, the system tells its own earlier state-history from a twin's (same parameters, different seed) | Disruption overwrites the stored value, and a twin carries an identical variable | A stored hash of past states. The test must require the history to be recognised from the dynamics, not looked up. |
>
> **Predictions for the Spanda Self-Model:** fails all three, as built. The efference-copy variant
> also fails V4a (one ablation kills it). Recorded now so the run can't move the goalposts.

*Outcome:* all three tests were passed by a cheat or an outsider (N2–N4).

## B2: kill-check for the transformer bridge (2026-10-22)

> - Run V3, V4b and V4c analogues against two outsiders: one with full access (the trivial case) and
>   one seeing outputs only (matching the toy's ring-only observer).
> - Prediction from N1–N4: the self-variable exists and passes, but the outputs-only outsider largely
>   matches it. Kill-check: if the outputs-only outsider stays far behind however well it is trained,
>   that is a genuinely new kind of evidence and goes in the ledger as a result against the symmetry.

*Outcome:* partly fired (N6). Outsiders that see only outputs trail by 0.031–0.059 AUROC, and an
exact copy matches. Training the outsider harder did not close the gap (N5).

## J1: kill-checks (2026-10-23)

> On B2's self/other items, a grouped-CV probe reads own vs. inserted tokens from each model's
> middle-layer state projected onto its top-k J-space, a random k-subspace, its bottom-k
> directions, and the full space. Target GPT-2 Medium; outsiders GPT-2 Small and GPT-Neo 125M;
> k = 32 and 128.
>
> - K1, J-space not special: top-k AUROC not above random-k (paired bootstrap CI covers 0).
> - K2, workspace not privileged: target minus outsider in J-space has a CI covering 0. If so,
>   third-person symmetry extends to the workspace indicator.
> - K3, leakage: shuffled labels must give AUROC near 0.5.

*Outcome:* N7. The script is `experiments/transformer/j1_jspace_self.py`.

## Negative controls (2026-10-26)

> **Part A: scrambled GPT-2 Medium on the B2/J1 items** (1,629 items, identical to B2 and J1).
> Predictions: S-all at chance on B2 (≤ 0.55, both kinds); S-blocks fluent ≤ 0.55; probes on S-all
> ≤ 0.60. Kill: a scrambled probe ≥ 0.80 on off-context items.
>
> **Part B: V1 and V3 on a non-linguistic network.** A GRU (64 units) forecasting the 21 ring
> phases, closed loop, and its scrambled copy. Predictions: V3 chance without a copy and 1.00 with
> it for both; V1 passed by both (R² > 0.9, ratio > 1). Kill against our own drafting if both hold.

*Outcome:* Part A held as predicted. Part B went against expectation: the trained GRU failed V1
while the scrambled copy passed (N9). The script is `experiments/controls/negative_controls.py`.

## V4c with a learned rule, and the B2 truncation check (2026-10-28)

These were written after an independent review of a manuscript draft and before either run. The
review had pointed out that the V4b and V4c outsiders were given the update rule and the anchor
shift, not shown to learn them. It had also pointed out that B2's own tokens are sampled from the
target's top 50, so part of the target's edge might be a truncation cue.

> **[V4c, learned rule.]** The outsider sees ring phases only, from three training runs on seeds
> not used in V4c (500–502). It infers the anchor on short windows, detects when the anchor jumps,
> estimates the update period and phase, and fits the rate η by least squares of each jump on
> |z|·wrap(arg z − a). It then reruns V4c with its fitted rule in place of the true one. **Kill:**
> if agreement with the self (log-undo) recognizer falls below 0.99 at any disruption level, or
> the fitted η is off by more than 10%, the paper must say the outsider was given the rule and
> learning it failed.
>
> **[B2, truncation.]** On the 1,629 B2 items, compute each inserted token's rank under the target.
> Recompute every scorer's fluent AUROC and the paired gaps (2,000 bootstrap resamples over
> prompts) using only inserted tokens inside the target's top 50. **Kill:** if every outsider's gap
> interval then includes zero, the B2 edge is a truncation artefact and N6 is revised. Whatever
> happens, the paper states that own tokens are samples from the target's own distribution and
> that the copy matches because it computes the same function.

*Outcome:* the learned-rule kill fired (N10); a post-hoc robust fit is reported as post-hoc. The
truncation kill did not fire.
