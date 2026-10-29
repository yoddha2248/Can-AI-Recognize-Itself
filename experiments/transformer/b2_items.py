"""
The B2 self/other item set, shared by B2, J1 and the negative controls.

b2_gpu.py builds the items (1,629 own / fluent / off-context tokens over 150 prompts) and the
4,000 distillation continuations, and keeps them in a pickle checkpoint
(results/transformer/b2_checkpoint.pkl, ~1 MB, git-ignored). The release ships the same
data as plain JSON instead (results/transformer/b2_items.json.gz): lossless, no pickle.

    load_b2_state()        -> {"items": [...], "train_txt": [(prompt_ids, continuation_ids), ...]}
    python experiments/transformer/b2_items.py export [PKL] [OUT]   # pickle -> json.gz, verified round trip

Regenerating the items needs a GPU run of b2_gpu.py; sampling on different hardware or library
versions may not give identical tokens, which is why the committed set is shipped.
"""
import gzip
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TP = os.path.join(ROOT, "results", "transformer")
PKL = os.path.join(TP, "b2_checkpoint.pkl")
JSON_GZ = os.path.join(TP, "b2_items.json.gz")


def _from_json(d):
    return {"items": [dict(k=i["k"], prefix=list(i["prefix"]), tok=i["tok"], kind=i["kind"]) for i in d["items"]],
            "train_txt": [(list(p), list(c)) for p, c in d["train_txt"]]}


def load_b2_state(path=None):
    """Load items and train_txt from a b2 checkpoint (.pkl) or the shipped .json.gz.

    Order: explicit path; else the pickle checkpoint if a local B2 run produced one; else the
    shipped json.gz.
    """
    if path is None:
        path = PKL if os.path.exists(PKL) else JSON_GZ
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            st = pickle.load(f)          # only ever a file written by your own b2_gpu.py run
        return {"items": st["items"], "train_txt": st["train_txt"]}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return _from_json(json.load(f))


def export(pkl=PKL, out=JSON_GZ):
    with open(pkl, "rb") as f:
        st = pickle.load(f)
    d = {"about": "B2 self/other items and distillation continuations (GPT-2 token ids); "
                  "exported losslessly from b2_checkpoint.pkl",
         "n_items": len(st["items"]), "n_train_txt": len(st["train_txt"]),
         "items": st["items"], "train_txt": [[list(p), list(c)] for p, c in st["train_txt"]]}
    with gzip.GzipFile(out, "wb", mtime=0) as g:          # mtime=0: byte-reproducible file
        g.write(json.dumps(d, separators=(",", ":")).encode("utf-8"))
    back = load_b2_state(out)
    assert back["items"] == st["items"] and back["train_txt"] == list(st["train_txt"]), "round trip differs"
    print(f"wrote {out} ({os.path.getsize(out)} bytes); round trip identical")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "export":
        export(*sys.argv[2:4])
    else:
        st = load_b2_state()
        print(len(st["items"]), "items,", len(st["train_txt"]), "continuations")
