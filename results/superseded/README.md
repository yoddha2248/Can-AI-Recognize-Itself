# Superseded outputs

Kept for the record, not cited: these V4c learned-rule runs (2026-10-28) applied the fitted rule one
phase snapshot late (`k = i + 1 + lag`), fixed in `experiments/toy_ladder/v4c_learned_rule.py` (`k = i + lag`);
the current outputs are `results/toy_ladder/v4c_learned_rule.txt` (pre-registered, N10) and
`v4c_learned_rule_posthoc_robust.txt` (post-hoc).

| File | What it was |
| --- | --- |
| `v4c_learned_rule_run1_indexbug.txt` | First pre-registered run, with the indexing bug (568/600) |
| `v4c_learned_rule_posthoc_robust_indexbug.txt` | Post-hoc robust fit, with the indexing bug (591/600) |
| `v4c_learned_rule_posthoc_20runs_indexbug.txt` | Post-hoc scaling check, 20 training runs, with the indexing bug (534/600) |
