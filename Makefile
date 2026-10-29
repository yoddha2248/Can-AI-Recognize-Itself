# Reproduce the paper's results. Every target is defined once, in reproduce.py, and writes into
# runs/<target>/ (git-ignored), never over the committed results/. Without make (e.g. on Windows):
#     python reproduce.py <target>        (python reproduce.py list shows them all)

PY ?= python3

.PHONY: help list toy truncation b1-smoke b1 controls-smoke controls b2 j1 gpu figures committed example test test-fast

help list:
	@$(PY) reproduce.py list
	@echo "  example         run examples/quickstart.py (seconds)"
	@echo "  test            pytest, includes the 2-minute toy rerun; test-fast skips it"

toy truncation b1-smoke b1 controls-smoke controls b2 j1 gpu figures committed:
	$(PY) reproduce.py $@

example:
	$(PY) examples/quickstart.py

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"
