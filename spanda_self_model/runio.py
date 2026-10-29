"""Small helpers shared by the run scripts: repository root, --out/--smoke flags, stdout tee."""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "results")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def parse(description, default_out, smoke=True):
    """Parse --out PATH (default: the committed result file) and, if offered, --smoke.

    --out - prints to stdout only. Anything else: stdout is also written to that file.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", default=default_out,
                    help=f"file to write the printed results to ('-' = stdout only; default {os.path.relpath(default_out, ROOT)})")
    if smoke:
        ap.add_argument("--smoke", action="store_true",
                        help="quick pipeline check with fewer seeds; numbers will NOT match the paper")
    args = ap.parse_args()
    if not smoke:
        args.smoke = False
    if args.out != "-":
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        f = open(args.out, "w")
        sys.stdout = _Tee(sys.__stdout__, f)
    return args
