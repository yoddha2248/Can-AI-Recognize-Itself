<!-- REPO_NAME: the repository URL is defined once, in the [repo] link reference at the bottom
     of this file (and in CITATION.cff: repository-code). Change it there only. -->

# Can AI Recognize Itself? Code and results

**Code, results and negatives ledger for the paper "Can AI Recognize Itself? Why self-related
indicators cannot settle machine consciousness, and what a tenth-century Kashmiri debate
anticipated"** (Ajay Hagan, Spanda Research). Repository: [Can-AI-Recognize-Itself][repo].

Self-related indicators of consciousness (a model of one's own states, telling one's own
activity from external input, predicting oneself, recognizing one's own past) are proposed as
evidence about machine consciousness. This code tests whether such indicators can tell a self
from a system that merely carries information about itself. For every test it builds the test,
the cheapest mechanism that passes it, and an outside observer that tries to match the system's
own answers, then runs the ladder on a 21-oscillator self-model and on GPT-2 Medium, with
scrambled-weight and non-linguistic negative controls.

The paper responds to Seth, A. K., Block, N. & Tononi, G. (2026), "Three reasons why AI is not
conscious and perhaps never will be", *Nature* 658, 866–868.

**Predecessor.** This repository follows
[Why AI Might Be Conscious — and Why No Test Can Yet Tell][v010] (v0.1.0, essay plus the toy
ladder, B1 and B2; concept DOI [10.5281/zenodo.23589038](https://doi.org/10.5281/zenodo.23589038)).
Everything from v0.1.0 that the paper uses is included here. What is new is set out in
[What's new since v0.1.0](#whats-new-since-v010) below; file-level detail is in [CHANGELOG.md](CHANGELOG.md).

**Two registers.** The code and the numbers in `results/` are established results [E]: they
come from fixed seeds and are checked by `tests/`. Interpretation, including the reading through
Pratyabhijñā philosophy, lives in the paper and is marked there as interpretation [A]. Nothing in
the code depends on it. The model's component names (anchor, vimarśa loop) come from Kashmir
Śaiva philosophy, which informed its design.

## What's new since v0.1.0

v0.1.0 was an essay with the code behind it. v1.0.0 is the code and full record for a research
paper: it adds three experiments, two controls, a pre-registration record and a test suite, and
it corrects three statements in v0.1.0's README. The table gives the overview; each new item is described
below it.

| | v0.1.0 ([Why-AI-Might-be-Conscious-][v010]) | v1.0.0 (this repository) |
| --- | --- | --- |
| Toy self-model tests V1–V4c | Scripts | Scripts, every result file, the model as an installable module with its equations, and a quickstart example |
| Transformer tests | B1, B2 | B1, B2, **J1**, **B2 truncation check** |
| Negative controls | None | **Scrambled GPT-2 Medium; untrained non-linguistic network** |
| Follow-ups from independent review | None | **V4c with a learned rule** (pre-registered; failed, N10) |
| What was fixed before each run | Not published | **`docs/PREREGISTRATION.md`**: each criterion quoted word for word, with what was *not* fixed in advance |
| Negative results | Summarized in the README; no ledger file | **`docs/NEGATIVES.md`**, ledger N1–N10 with dates; superseded runs kept and marked |
| Reproduction | Run scripts by hand | One command per result on any OS (`python reproduce.py`, or `make`), `--smoke` modes, a pytest suite; toy ladder reruns byte for byte; paper figures redrawn from `results/` |
| Data | B1 and B2 summary JSON | Plus the B2 item set (no pickle) and per-token scores; data licence |
| Headline claim | "Gamed, or reduces to privileged access, which is real but graded" | **"Gameable or merely privileged"**; graded dropped; symmetry stated as interpretation |
| What the toy outsiders were given | "Learned the update rule"; "recovered the system's own equations" | **Stated exactly**: given the update rule (V4c); recovered the coupling constants and was told each disturbance (V4b) |
| Essay | Included | Stays in v0.1.0 |

### New experiments

**J1: is the self/other signal in the model's workspace?** (`experiments/transformer/j1_jspace_self.py`)
Recent interpretability work describes a verbalizable "workspace" in language models, J-space
(Gurnee et al. 2026). If a self were to be found by functional means, this is where one would
look. J1 probes own-versus-inserted tokens from GPT-2 Medium's middle layer, projected onto its
J-space, onto random directions and onto its lowest-gain directions. J-space carries some
self/other information but no more than other directions, and fluent insertions are not readable
there at all (ledger N7).

**B2 truncation check: is B2's inside edge a sampling artefact?** (`experiments/transformer/b2_truncation_check.py`)
In B2 the model's own tokens are drawn from its top 50, so an inserted token outside that set
would be an easy cue. Restricted to insertions inside the top 50 (94.7% of them), the target
still beats every outsider (AUROC 0.617; gaps +0.030 to +0.056, all intervals above zero). The
edge is not a truncation cue. Pre-registered; the kill did not fire.

**V4c with a learned rule: can an outsider learn the self's update rule, not just be given it?**
(`experiments/toy_ladder/v4c_learned_rule.py`)
In v0.1.0 the outsider that matched the system on 600/600 trials had been *given* the reflexive
update rule. Here it must fit the rule from ring phases alone. The pre-registered fit recovered
the period exactly and the rate 1.7% low, and agreed on 566/600 trials, below the 0.99
criterion: **the kill fired (N10)**. A post-hoc robust fit on the same data recovers the rate
exactly and agrees on 600/600; it is shipped and labelled post-hoc, and the paper rests no claim
on it. Three earlier runs with an indexing bug are kept in `results/superseded/`.

### New controls

**Negative control A: scrambled weights** (`experiments/controls/negative_controls.py`, Part A).
GPT-2 Medium with its weights permuted keeps the architecture and loses everything learned. B2
and the J1 probes fall to chance (AUROC 0.46–0.53), so the transformer signals come from what
the model learned.

**Negative control B: a network with no language and no training** (Part B).
A small recurrent network forecasting the ring phases, and a scrambled copy of it. The
scrambled copy passes V1 and, given one copied variable, V3. So V1 and V3 detect recurrence plus
a record, not learning or language. This went against our expectation (N9) and is why the toy
self-tests count as gameable rather than merely weak.

### New records

**`docs/PREREGISTRATION.md`.** Every kill criterion and prediction written before its run,
quoted word for word from the project's dated log, with a table of what was fixed in advance and
what was not (the numerical thresholds for V4a–V4c were set afterwards).

**`docs/NEGATIVES.md`.** The negatives ledger N1–N10 with dates. Negative results are results.

**`docs/DATA_LICENSE.md`.** Code MIT; result files CC BY 4.0; the B2 item file CC BY-SA 3.0,
because it encodes WikiText-103 prompts.

### New infrastructure

- **`spanda_self_model/`**: the Spanda Self-Model as an installable module (`pip install -e .`),
  with its Hamiltonian, integrator and reflexive update documented (`docs/MODEL.md`, Appendix B
  of the paper), and the outsider used by V4b and V4c (`spanda_self_model/observers.py`).
  Numerics unchanged.
- **`experiments/`**, one folder per kind of test, mirrored by `results/`: `toy_ladder/`,
  `transformer/`, `controls/`.
- **Reproduction on any OS**: `python reproduce.py <target>` (or `make <target>`) runs a test into
  `runs/`, never over `results/`, and `scripts/check_results.py` compares it with the committed
  files. Every script also takes `--out` and, where cheap, `--smoke`.
- **A pytest suite**, including a byte-for-byte rerun of the toy ladder and checks that every path
  named in this README exists.
- **`examples/quickstart.py`**: the model, the self-versus-injected test and the outsider, in a
  few seconds.
- **`paper/figures/`**: the script that redraws the paper's data figures from `results/`, and the
  TikZ sources of its diagrams.
- **B2 item set without pickle** (`results/transformer/b2_items.json.gz`): the 1,629 test
  items and 4,000 distillation texts that J1 and the controls read, in a safe format.

### Corrected statements

- **"An outsider that learned the update rule" (V4c).** v0.1.0 said this; in fact the outsider
  was *given* the reflexive update rule. v1.0.0 says so, and the learned-rule test above reports
  what happens when it must learn it (566/600 pre-registered; N10).
- **"Recovered the system's own equations" (V4b).** The outsider recovered the coupling
  constants from ring phases (1.300, 0.500, 0.400, |γ| 0.794), was told each disturbance,
  including the size of an anchor shift, and simulated without the reflexive update. v1.0.0
  states this exactly.
- **"Privileged access, which is real but graded."** v1.0.0 drops "graded": a larger outsider
  and a distilled one did not close B2's gap, and the outsiders were not ordered by how much
  access they had. What stands is that the edge is small, survives the truncation check, is
  partly built in (the model scores the distribution it sampled from), and is matched exactly
  by a copy, which computes the same function. The headline is now "gameable or merely
  privileged", and the step from there to third-person symmetry is presented in the paper as
  interpretation [A], not as a result.

## Quick start

```
pip install -r requirements-testbed.txt      # NumPy only
python examples/quickstart.py                # the model, the self-vs-injected test and the outsider, in seconds
python reproduce.py toy                      # V1-V4c + learned rule, ~4 min on one CPU, byte-identical check
python -m pytest -q -m "not slow"            # the test suite without the 2-minute rerun
```

Transformer experiments, controls and figures (install PyTorch for your hardware first):

```
pip install -r requirements.txt
python reproduce.py list             # every target, with hardware and time
python reproduce.py b1-smoke         # B1 part 1: is there a record of authorship?  CPU, minutes
python reproduce.py truncation       # B2 truncation check                          CPU, ~10-20 min
python reproduce.py controls-smoke   # negative controls, pipeline check            CPU, ~10 min
python reproduce.py b1               # B1 full                                      CPU, ~55 min
python reproduce.py controls         # negative controls, full                      CPU, ~52 min
python reproduce.py gpu              # B2 + J1 + controls                           GPU
python reproduce.py figures          # redraw the paper's data figures from results/
python reproduce.py committed        # print the headline numbers stored in results/
```

On Linux and macOS `make <target>` does the same. Every target writes into `runs/` (git-ignored),
never over `results/`, then runs `scripts/check_results.py`, which compares the run with the
committed files and prints the headline numbers next to the paper's. Every script also accepts
`--out` and, where cheap, `--smoke`, and by default writes to `results/` as the committed runs did.
To use the model in your own code, `pip install -e .` makes `spanda_self_model` importable
anywhere (NumPy is its only dependency).

## Tests, scripts, results and the paper

Paper sections: 3 (testbed and method: 3.1 model, 3.3 method and controls, 3.5 statistics),
4 (results 4.1–4.9, with Figure 3 summarizing every scored test; negative controls in 4.8), 7.8–7.10
(objections from review), Appendix A (Table A1 design, Table A2 outcome), Appendix B (model specification,
Table B1, Figure B1). Ledger rows are in
[docs/NEGATIVES.md](docs/NEGATIVES.md).

| Test | Script | Result file | Paper | Ledger | Headline number |
| --- | --- | --- | --- | --- | --- |
| V1a / V1b closed loop | `experiments/toy_ladder/vimarsa_tests.py` | `results/toy_ladder/vimarsa_tests.txt` | 3.3, 4.1; A1/A2 (V1) | N1 | V1a tracking 0.037 ± 0.001 (loop cut 0, decoy 0); V1b 1.49 ± 0.01 (cut 1.94, decoy 0.29) |
| V3 self vs. injected change | `experiments/toy_ladder/vimarsa_tests.py` | `results/toy_ladder/vimarsa_tests.txt` | 4.1; A1/A2 (V3) | N1 | State difference without a record exactly 0; with an efference copy, error 0 (self) vs 1.6e-3 (injected) |
| V4a distributed self | `experiments/toy_ladder/v4a_tests.py` | `results/toy_ladder/v4a_tests.txt` | 4.2; Fig. 4; A1/A2 (V4a) | N2 | Replicated copies and 3-of-21 Shamir shares both 1.00 with 86% of nodes ablated |
| V4b novel self-prediction | `experiments/toy_ladder/v4b_tests.py` | `results/toy_ladder/v4b_tests.txt` | 4.3; Fig. 5; A1/A2 (V4b) | N3 | Phases-only outsider within 0.5–4.6% of null error on new disturbance types |
| V4c re-recognition vs. a twin | `experiments/toy_ladder/v4c_tests.py` | `results/toy_ladder/v4c_tests.txt` | 4.4; Fig. 6; A1/A2 (V4c) | N4 | Outsider *given* the update rule agrees with the self on 600/600 trials; self 1.00 with the anchor intact, 0.47 when it is reset |
| V4c with a *learned* rule (review follow-up) | `experiments/toy_ladder/v4c_learned_rule.py` | `results/toy_ladder/v4c_learned_rule.txt` (pre-registered); `v4c_learned_rule_posthoc_robust.txt` (post-hoc, `--robust`) | 4.4 | N10 (kill fired) | Pre-registered least-squares fit: period 50 exact, rate 1.7% low, agreement 566/600 (0.84 with the anchor intact), below the 0.99 criterion. Post-hoc robust fit: rate exact, 600/600; no claim rests on it |
| B1 authorship record (GPT-2 Medium) | `experiments/transformer/b1_pilot.py` | `results/transformer/b1_results.json` | 4.5; A1/A2 (B1) | N5 | max \|self − fed\| = 0 exactly; fluent AUROC 0.598 |
| B2 outsiders, paired tests | `experiments/transformer/b2_gpu.py` | `results/transformer/b2_results.json` | 4.6; Figs. 7a, 7b; A1/A2 (B2) | N6 | Target AUROC 0.637, exact copy 0.637, outsiders 0.577–0.606; paired gaps +0.031 to +0.059, all excluding 0 |
| B2 truncation check (review follow-up) | `experiments/transformer/b2_truncation_check.py` | `results/transformer/b2_truncation_check.json` | 4.6; objection 7.8 | none: kill did not fire (N6 stands) | Own tokens 100% inside the target's top 50, fluent insertions 94.7%; restricted to those, target 0.617, copy 0.617, outsiders 0.561–0.587, gaps +0.030 to +0.056, all excluding 0 |
| J1 self/other signal in J-space | `experiments/transformer/j1_jspace_self.py` | `results/transformer/j1_results.json` | 4.7; Fig. 8; A1/A2 (J1) | N7 | Off-context: bottom-k 0.894–0.930 beats J-space top-k 0.790–0.873; top-128 − random-128 (target) −0.001 [−0.022, +0.019]; fluent 0.46–0.55 everywhere; shuffled labels 0.46–0.54 |
| NC-A scrambled GPT-2 Medium | `experiments/controls/negative_controls.py` (Part A) | `results/controls/negative_controls.json` | 3.3, 4.8; A1/A2 (control A) | none: held as predicted (see N9) | S-all B2 AUROC ≤ 0.493, probes ≤ 0.50; unscrambled target reproduces 0.637 |
| NC-B non-linguistic GRU | `experiments/controls/negative_controls.py` (Part B) | `results/controls/negative_controls.json` | 3.3, 4.8; A1/A2 (control B) | N9 | Scrambled GRU passes V1 (R² 0.963, nudge ratio 1.15) and V3 (1.00 with the copy, 0.5 without); trained GRU fails V1 (R² −53.4) |

V2 (language-independence) is defined in the paper but is untestable on the toy, which has no
verbal channel. The B2 per-token scores are in `results/transformer/b2_scores.npz`; the B2
item set that J1 and the controls read is `results/transformer/b2_items.json.gz`.

## Hardware and time

| Run | Hardware | Time | Download |
| --- | --- | --- | --- |
| Toy ladder V1–V4c | CPU, NumPy only | ~2 min in total (V4a 5 s, V4b 40 s, V4c 45 s, V1–V3 30 s) | none |
| V4c learned rule | CPU, NumPy only | ~1 min per mode (pre-registered, `--robust`) | none |
| B1 | CPU (2 threads) | ~55 min; `--smoke` (part 1 only) a few minutes | ~4.5 GB (GPT-2 Medium, Large, Small, DistilGPT-2), WikiText-103 |
| B2 | GPU, 4 GB is enough (run on an RTX 2050) | ~20 min of compute | ~11 GB (adds GPT-Neo 125M and 1.3B) |
| J1 | GPU (laptop) | ~25 min | GPT-2 Medium, GPT-2 Small, GPT-Neo 125M |
| B2 truncation check | CPU (GPT-2 Medium) | ~10–20 min | GPT-2 Medium |
| Negative controls | CPU is enough (committed run: CPU, torch 2.14.1+cpu) | ~52 min; `--smoke` ~10 min | GPT-2 Medium |

Weights are cached in `.hf_cache/` at the repository root (git-ignored) unless `HF_HOME` is
set. B2 saves `b2_checkpoint.pkl` after each stage, so an interrupted run resumes.

## What was reproduced for this release

Rerun from this repository in a clean container (Python 3.13, NumPy 2.5.3, torch 2.14.1+cpu,
transformers 5.20.2, 2 CPU cores) before release, 2026-10-28, and B2 and J1 on a laptop GPU. After the
reorganisation of 2026-10-29 the toy ladder, the B2 truncation check and the controls smoke run were
rerun from the new layout, and the test suite passed: the truncation check is byte-identical and the smoke controls match within tolerance (25 tests passed).

| Run | Outcome |
| --- | --- |
| Toy ladder V1–V4c (`python reproduce.py toy`) | All four outputs byte-identical to `results/toy_ladder/` |
| V4c learned rule, pre-registered and `--robust` | Both outputs byte-identical (566/600, kill fires; post-hoc 600/600) |
| B1 full (CPU, ~2 h on a shared 2-core machine) | `b1_results.json` identical to the committed file; max \|self − fed\| = 0.0, pasted 7.78e-4 |
| B2 truncation check (CPU) | `b2_truncation_check.json` identical, computed from the shipped `b2_items.json.gz` |
| Negative controls, full (CPU) | `negative_controls.json` identical apart from run metadata; log identical apart from timestamps |
| Negative controls, smoke | Identical to `results/controls/smoke/` |
| B2 and J1 (GPU: RTX 2050, 4 GB, Windows) | `b2_results.json` and `j1_results.json` byte-identical to the committed files (2026-10-28) |

## Honest limits

- The toy is one deterministic, noise-free 21-oscillator model; its "self" predicts itself with
  zero error because nothing is noisy. A noisy and a learning version are still to be built.
- The transformer is GPT-2 Medium (355M), with outsiders of 82M–1.3B parameters; no frontier
  model and no verbal introspection were tested. One insertion source; one target size.
- The negative controls use one small recurrent network, trained only modestly better than a
  persistence baseline.
- The tests, cheats and outsiders are ours. A different kind of test might resist them; finding
  one would be the most useful thing anyone could do with this code.
- Exact reproduction of the transformer runs depends on hardware and library versions (sampling
  and floating-point order); the toy ladder does not.

## Repository layout

```
spanda_self_model/      the model, as an installable package (NumPy only)
  model.py                equations, parameters, Sim (modes full / cut / decoy), efference copy
  observers.py            the outsider of V4b/V4c and the logged ground-truth run
  runio.py                --out / --smoke handling shared by the scripts
experiments/            one script per test; each writes to results/ by default, or --out
  toy_ladder/             V1-V3, V4a, V4b, V4c, V4c with a learned rule        (NumPy, CPU)
  transformer/            B1, B2, J1, the B2 truncation check, the B2 item loader (PyTorch)
  controls/               negative controls: scrambled GPT-2 Medium, non-linguistic GRU
results/                committed outputs of every run cited in the paper, mirroring experiments/
  toy_ladder/  transformer/  controls/
  superseded/             three V4c learned-rule runs with an indexing bug, kept, not cited
examples/quickstart.py  a few-second tour of the model and the outsider
paper/figures/          make_figures.py (redraws the data figures from results/), tikz/ (diagrams)
docs/                   MODEL.md, NEGATIVES.md (ledger N1-N10), PREREGISTRATION.md, DATA_LICENSE.md
scripts/check_results.py  compares a run with results/ and prints headline numbers
tests/                  pytest: the model, the committed results, the toy rerun, the layout
reproduce.py, Makefile  one command per result; runs go to runs/ (git-ignored)
```

```python
import spanda_self_model as ssm
sim = ssm.Sim(seed=0, mode="full").run(ssm.BURN)    # modes: full, cut, decoy
print(sim.anchor, sim.efcopy)
```

## References

- Seth, A. K., Block, N. & Tononi, G. (2026). Three reasons why AI is not conscious and perhaps never will be. *Nature* 658, 866–868.
- Butlin, P. et al. (2025). Identifying indicators of consciousness in AI systems. *Trends in Cognitive Sciences*.
- Birch, J. (2024). *The Edge of Sentience*, ch. "Large Language Models and the Gaming Problem". Oxford University Press.
- Binder, F. J. et al. (2025). Looking Inward: Language Models Can Learn About Themselves by Introspection. ICLR 2025. arXiv:2410.13787.
- Anthropic (2025). Emergent introspective awareness in large language models.
- Gurnee, W., Sofroniew, N., Pearce, A., et al. (2026). Verbalizable representations form a global workspace in language models. arXiv:2607.15495. The "J-space" probed in J1.

## Disclosure

Developed with Claude (Anthropic), which helped design the experiments, write the code and draft
this text. Anthropic's research is among the work discussed. The author is responsible for the
content.

## License and citation

Code: MIT, © 2026 Ajay Hagan / Spanda Research (see `LICENSE`). Result files: CC BY 4.0, except
`results/transformer/b2_items.json.gz`, which encodes WikiText-103 prompts and is CC BY-SA 3.0
(see `docs/DATA_LICENSE.md`). Kill criteria written before each run are quoted in
`docs/PREREGISTRATION.md`; negative results are in `docs/NEGATIVES.md`. To cite, see
`CITATION.cff`; a DOI will be added when this release is archived on Zenodo. The predecessor
release is cited as Hagan (2026), DOI [10.5281/zenodo.23589038](https://doi.org/10.5281/zenodo.23589038).

[repo]: https://github.com/yoddha2248/Can-AI-Recognize-Itself
[v010]: https://github.com/yoddha2248/Why-AI-Might-be-Conscious-
