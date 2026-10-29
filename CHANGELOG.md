# Changelog

This repository (`Can-AI-Recognize-Itself`) holds the code and results for the paper
"Can AI Recognize Itself?". It succeeds an earlier, separate repository and is linked to it as
its predecessor:
[Why-AI-Might-be-Conscious-](https://github.com/yoddha2248/Why-AI-Might-be-Conscious-)
(v0.1.0, concept DOI [10.5281/zenodo.23589038](https://doi.org/10.5281/zenodo.23589038)).

## v1.0.0 (unreleased)

First release of this repository. All code and results of v0.1.0 are carried over unchanged
in substance; everything the paper cites is now in one place.

### Added
- **J1** (`experiments/transformer/j1_jspace_self.py`, `results/transformer/j1_results.json`):
  is the self/other signal in GPT-2's J-space workspace? It is not (ledger N7).
- **Negative controls** (`experiments/controls/negative_controls.py`, `results/controls/`):
  a scrambled-weights GPT-2 Medium (signals fall to chance) and a non-linguistic GRU whose
  scrambled copy passes V1 and V3 (ledger N9). Full and smoke outputs are both committed.
- **V4c with a learned rule** (`experiments/toy_ladder/v4c_learned_rule.py`): a follow-up to an independent review,
  pre-registered. An outsider that must learn the update rule from ring phases falls short of
  the 0.99 criterion (566/600; kill fired, ledger N10). The post-hoc robust fit (600/600) is
  shipped and labelled as post-hoc; three superseded runs with an indexing bug are kept in
  `results/superseded/`, not cited.
- **B2 truncation check** (`experiments/transformer/b2_truncation_check.py`): a follow-up to an independent review,
  pre-registered. Restricted to inserted tokens inside the target's top 50, every outsider gap
  still excludes zero (kill did not fire; N6 stands).
- **Layout** (2026-10-29): scripts grouped by kind under `experiments/` (`toy_ladder/`,
  `transformer/`, `controls/`), mirrored by `results/` (formerly `tests_ladder/`, `transformer/`,
  `controls/` and `results/spanda_testbed/`, `results/transformer_pilot/`,
  `results/negative_controls/`). The V4c outsider moved into `spanda_self_model/observers.py`;
  `v4c_learned_rule.py` imports it instead of exec'ing `v4c_tests.py`. Outputs unchanged (the toy
  ladder reruns byte for byte).
- **`reproduce.py`**: one command per result on any OS; the Makefile now calls it.
- **`examples/quickstart.py`**, **`docs/MODEL.md`**, **`pyproject.toml`** (`pip install -e .`),
  **`paper/figures/`** (redraws the paper's data figures from `results/`; TikZ sources of the
  diagrams), and `tests/test_layout.py` (README paths exist, targets resolve, example and
  figures run).
- **Packaged model** `spanda_self_model/`: the Spanda Self-Model as an importable module with
  docstrings (equations, parameters, efference copy). Numerics unchanged: the toy ladder
  reproduces the committed outputs byte for byte.
- **B2 item set without pickle**: `results/transformer/b2_items.json.gz` (0.5 MB),
  a lossless export of the 1,629 items and 4,000 distillation continuations that J1 and the
  controls read. The 1 MB pickle checkpoint is no longer shipped; `experiments/transformer/b2_items.py`
  documents the export and verifies the round trip.
- **Tests** (`tests/`, pytest): model integration and energy drift, V3 state difference exactly
  0 without a record, the committed results carry the paper's headline numbers, and the full
  toy ladder reruns byte-identically (marked `slow`).
- **Reproduction**: `Makefile` targets (`toy`, `b1-smoke`, `b1`, `controls-smoke`, `controls`,
  `b2`, `j1`, `gpu`) that write into `runs/` and compare with `results/` via
  `scripts/check_results.py`.
- `--out` and `--smoke` flags on every script; module docstrings that name the paper section,
  ledger row and expected numbers.
- `docs/NEGATIVES.md`: the negatives ledger N1–N10 (N8 reserved), with dates; later readings are
  appended as dated notes, never rewritten into the rows.
- `docs/PREREGISTRATION.md`: every kill criterion and prediction written before its run, quoted
  word for word from the project's dated log, with a table of what was and was not fixed in advance.
- `docs/DATA_LICENSE.md`: result files CC BY 4.0; the B2 item file CC BY-SA 3.0 (WikiText-103).
- `.zenodo.json`: archive metadata for the Zenodo release.
- `LICENSE` (MIT), `CITATION.cff` (references the v0.1.0 release), pinned `requirements.txt`.

### Changed
- Layout (relative to v0.1.0): `spanda_testbed/` became `spanda_self_model/` (model) and
  `experiments/toy_ladder/` (V1–V4c); `transformer_pilot/` became `experiments/transformer/`, with
  results under `results/toy_ladder/` and `results/transformer/`.
- Paths: every script writes into `results/` relative to the repository root and caches model
  weights in `.hf_cache/` at the root (git-ignored).
- B1 now caches weights in `.hf_cache/` like B2 (v0.1.0 used the default Hugging Face cache).

### Fixed
- `experiments/controls/negative_controls.py`: a smoke run opened, and so emptied, the full-run log file
  before switching to `smoke/`. The smoke folder is now chosen first. Results are unaffected.

### Removed from the shipped files
- The essay (it stays in the v0.1.0 repository) and `b2_checkpoint.pkl` (replaced by the json.gz
  item set; a local B2 run regenerates the pickle).

## v0.1.0 (2026-10-23), predecessor repository

"Why AI Might Be Conscious — and Why No Test Can Yet Tell": code, results and essay, MIT.
Toy test ladder V1–V4c on the Spanda Self-Model, B1 (GPT-2 family, CPU) and B2 (adds GPT-Neo
outsiders and paired bootstrap tests, GPU), the outputs of every run, and the essay responding
to Seth, Block & Tononi (2026). Repository
[yoddha2248/Why-AI-Might-be-Conscious-](https://github.com/yoddha2248/Why-AI-Might-be-Conscious-);
Zenodo concept DOI [10.5281/zenodo.23589038](https://doi.org/10.5281/zenodo.23589038),
version DOI [10.5281/zenodo.23589685](https://doi.org/10.5281/zenodo.23589685).
