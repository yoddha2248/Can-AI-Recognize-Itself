"""Redraw the paper's data figures from the committed results/ (and, for Figure B1, the model).

    python paper/figures/make_figures.py [--out DIR]        (default runs/figures/)

Figures (numbered as in the paper):
    fig3_glance       Figure 3   every scored test on one axis, chance (0) to perfect (1)
    figA1_v4a         Figure 4   V4a, robustness without a bearer
    figA2_v4b         Figure 5   V4b, novel self-prediction
    fig3_v4c          Figure 6   V4c, recognizing one's own past
    fig4a_b2          Figure 7a  B2, AUROC per scorer
    fig4b_b2gap       Figure 7b  B2, paired gaps
    fig_j1            Figure 8   J1, probe AUROC by subspace
    figB1_run         Figure B1  a sample run of the model (seed 0)
File names keep the labels they had when first drawn. Figures 1, 2, 9 and the Box 1 timeline are
diagrams; their TikZ sources are in tikz/ (compile with xelatex, TeX Gyre Pagella).

With xelatex and the TeX Gyre Pagella fonts installed, the figures are typeset exactly as in the
paper (matplotlib's pgf backend). Otherwise they are drawn with matplotlib's default backend and
fonts: same data, different typography. FIG_STYLE=default forces this.
"""
import argparse, json, os, re, shutil, sys
import numpy as np
import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
import spanda_self_model as ssm                                        # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--out", default=os.path.join(REPO, "runs", "figures"), help="output directory")
_args = _ap.parse_args()
OUT = os.path.join(_args.out, "")
os.makedirs(OUT, exist_ok=True)
R = os.path.join(REPO, "results")
G = "/usr/share/texmf/fonts/opentype/public/tex-gyre/"
PAPER_STYLE = os.environ.get("FIG_STYLE") != "default" and shutil.which("xelatex") is not None and os.path.exists(G + "texgyrepagella-regular.otf")
if PAPER_STYLE:
    mpl.use("pgf")
    from matplotlib import font_manager as fm
    for f in ["texgyrepagella-regular", "texgyrepagella-bold", "texgyrepagella-italic"]:
        fm.fontManager.addfont(G + f + ".otf")
    mpl.rcParams.update({"font.family": "TeX Gyre Pagella", "pgf.texsystem": "xelatex", "pgf.rcfonts": False,
                         "pgf.preamble": r"\usepackage{fontspec}\setmainfont{TeX Gyre Pagella}"})
else:
    mpl.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
mpl.rcParams.update({"font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.bbox": "tight"})
ACC, GREY, LG, INK, ORG = "#1f5fbf", "#8a8a8a", "#c8c8c8", "#222222", "#c0661e"

# ---- Figure 6 (file fig3_v4c): V4c ----
t = open(R + "/toy_ladder/v4c_tests.txt").read()
rows = {lv: [float(x) for x in re.search(rf"^{lv}\s+(.+)$", t, re.M).group(1).split()] for lv in ("rings", "partial", "total")}
# columns: self-raw self-efcopy outsider outsider+rule
fig, ax = plt.subplots(figsize=(5.6, 2.9))
xs = [0, 1, 2]; lv = ["rings", "partial", "total"]
ax.axhline(0.5, color=GREY, lw=0.7, ls=(0, (3, 3)))
ax.text(-0.55, 0.507, "chance", va="bottom", fontsize=8, color=GREY)
off = {"self-raw": -0.22, "outsider": 0.22}
ax.scatter([x - 0.22 for x in xs], [rows[l][0] for l in lv], s=22, color=LG, zorder=3, label="System, current anchor only")
ax.scatter([x + 0.22 for x in xs], [rows[l][2] for l in lv], s=22, facecolors="white", edgecolors=GREY, lw=0.9, zorder=3, label="Outsider, ring phases only")
ax.scatter(xs, [rows[l][1] for l in lv], s=30, color=ACC, zorder=4, label="System, undoing its own log")
ax.scatter(xs, [rows[l][3] for l in lv], s=150, facecolors="none", edgecolors=ACC, lw=1.1, zorder=4, label="Outsider given the update rule")
ax.set_xticks(xs, ["Anchor intact", "Anchor nudged", "Anchor reset"]); ax.set_xlim(-0.6, 2.75)
ax.set_ylim(0.4, 1.05); ax.set_ylabel("Accuracy")
ax.annotate("system and outsider agree\non all 600 trials", xy=(0.06, 0.985), xytext=(0.3, 0.80), fontsize=8,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
ax.legend(loc="upper center", frameon=False, ncol=2, handletextpad=0.3, bbox_to_anchor=(0.5, -0.12))
fig.savefig(OUT + "fig3_v4c.pdf"); plt.close(fig)

# ---- Figure 4a: B2 AUROC with intervals ----
b2 = json.load(open(R + "/transformer/b2_results.json"))["table"]
nc = json.load(open(R + "/controls/negative_controls.json"))
def parse(s):
    m = re.match(r"([\d.]+) \[([\d.]+)-([\d.]+)\]", s); return [float(m.group(i)) for i in (1, 2, 3)]
names = {"target": "Target: GPT-2 Medium", "copy": "Exact copy of the target", "gpt2-small": "GPT-2 Small",
         "neo-125m": "GPT-Neo 125M", "neo-1.3b": "GPT-Neo 1.3B (4× the target)",
         "neo-125m distilled 1ep": "GPT-Neo 125M, distilled 1 epoch", "neo-125m distilled 3ep": "GPT-Neo 125M, distilled 3 epochs"}
items = [(names[r["scorer"]], *parse(r["auroc_fluent"]), r["scorer"] in ("target", "copy")) for r in b2]
def find_nc(d):
    out = []
    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items(): walk(v, path + [k])
        elif isinstance(x, list) and len(x) == 3 and all(isinstance(v, (int, float)) for v in x) and path[-1] == "b2_auroc" and "fluent" in path:
            out.append((path, x))
    walk(d, []); return out
ncs = find_nc(nc)
lab = {"S-all": "Scrambled copy, all weights", "S-blocks": "Scrambled copy, embeddings kept"}
for path, v in ncs:
    key = [p for p in path if p in lab]
    if key: items.append((lab[key[0]], v[0], v[1], v[2], False))
fig, ax = plt.subplots(figsize=(5.8, 3.3))
y = list(range(len(items)))[::-1]
for yi, (n, m, lo, hi, full) in zip(y, items):
    c = ACC if full else GREY
    ax.plot([lo, hi], [yi, yi], color=c, lw=1.4, solid_capstyle="butt")
    ax.plot(m, yi, "o", color=c, ms=4.5)
    ax.text(0.715, yi, f"{m:.3f}", va="center", fontsize=8.5, color=INK, fontweight="bold" if full else "normal")
ax.axvline(0.5, color=GREY, lw=0.7, ls=(0, (3, 3))); ax.axvline(0.637, color=ACC, lw=0.7, ls=(0, (3, 3)))
ax.set_yticks(y, [i[0] for i in items]); ax.set_xlim(0.43, 0.70); ax.set_ylim(-0.7, len(items) - 0.3)
ax.set_xlabel("AUROC for detecting inserted tokens (0.5 = chance)")
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.text(0.5, len(items) - 0.35, "chance", ha="center", fontsize=8, color=GREY)
ax.text(0.637, len(items) - 0.35, "target", ha="center", fontsize=8, color=ACC)
fig.savefig(OUT + "fig4a_b2.pdf"); plt.close(fig)

# ---- Figure 4b: paired gaps ----
def gp(s):
    m = re.match(r"([+-][\d.]+) \[([+-][\d.]+) to ([+-][\d.]+)\]", s); return [float(m.group(i)) for i in (1, 2, 3)]
gaps = [(names[r["scorer"]], *gp(r["self_minus_this_fluent"])) for r in b2 if r["scorer"] != "target"]
fig, ax = plt.subplots(figsize=(5.8, 2.4))
y = list(range(len(gaps)))[::-1]
for yi, (n, m, lo, hi) in zip(y, gaps):
    c = GREY if n.startswith("Exact") else ACC
    ax.plot([lo, hi], [yi, yi], color=c, lw=1.4); ax.plot(m, yi, "o", color=c, ms=4.5)
    ax.text(0.092, yi, f"{m:+.3f}", va="center", fontsize=8.5)
ax.axvline(0, color=GREY, lw=0.7, ls=(0, (3, 3)))
ax.set_yticks(y, [g[0] for g in gaps]); ax.set_xlim(-0.01, 0.10)
ax.set_xlabel("Target AUROC minus this scorer's, same prompts (95% paired bootstrap interval)")
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
fig.savefig(OUT + "fig4b_b2gap.pdf"); plt.close(fig)

# ---- Figure 4 (file figA1_v4a): V4a ----
t = open(R + "/toy_ladder/v4a_tests.txt").read()
ks = [float(x) for x in re.search(r"ablated fraction k:\s+(.+)", t).group(1).split()]
S = {n: [float(x) for x in re.search(rf"^{n}\s+(.+)$", t, re.M).group(1).split()] for n in ("none", "single", "replicated", "shamir")}
fig, ax = plt.subplots(figsize=(5.6, 2.9))
px = [100 * k for k in ks]
ax.axhline(0.5, color=GREY, lw=0.7, ls=(0, (3, 3)))
ax.plot(px, S["none"], "-o", color=LG, ms=3, lw=1, label="Model as built, no record")
ax.plot(px, S["single"], "-o", color=GREY, ms=3, lw=1, label="One copied variable")
ax.plot(px, S["replicated"], "-o", color=ACC, ms=3.5, lw=1.4, label="Cheat: the copy on every node")
ax.plot(px, S["shamir"], "--o", color=ACC, ms=6, mfc="white", lw=1.1, label="Cheat: 3-of-21 secret sharing")
ax.axvline(86, color=GREY, lw=0.6, ls=":"); ax.text(85, 0.42, "86%", ha="right", fontsize=8, color=GREY)
ax.set_xlabel("Share of the 21 nodes destroyed (%)"); ax.set_ylabel("Accuracy, self vs. injected")
ax.set_ylim(0.4, 1.05); ax.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.2))
fig.savefig(OUT + "figA1_v4a.pdf"); plt.close(fig)

# ---- Figure 5 (file figA2_v4b): V4b ----
t = open(R + "/toy_ladder/v4b_tests.txt").read()
types = [("A", "Small phase kicks\n(seen)"), ("B", "Momentum kicks\n(new)"), ("C", "Anchor shifts\n(new)"), ("D", "Large phase kicks\n(new)")]
vals = {}
for k, _ in types:
    v = [float(x) for x in re.search(rf"^{k} .*?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", t, re.M).groups()]
    vals[k] = dict(zip(["null", "lookup", "ridge", "self", "outsider"], v))
fig, ax = plt.subplots(figsize=(5.8, 2.9))
import numpy as np
w = 0.26; x = np.arange(4)
for i, (m, c, lab) in enumerate([("lookup", LG, "Lookup of past records"), ("ridge", GREY, "Ridge regression from logs"), ("outsider", ACC, "Outsider from ring phases")]):
    r = [100 * vals[k][m] / vals[k]["null"] for k, _ in types]
    ax.bar(x + (i - 1) * w, r, w * 0.92, color=c, label=lab)
    if m == "outsider":
        for xi, ri in zip(x, r): ax.text(xi + w, ri + 2, f"{ri:.1f}%", ha="center", fontsize=7.5, color=ACC)
ax.axhline(100, color=GREY, lw=0.7, ls=(0, (3, 3))); ax.text(3.72, 102, "null", ha="right", va="bottom", fontsize=8, color=GREY)
ax.set_xticks(x, [l for _, l in types]); ax.set_ylabel("Prediction error, % of null")
ax.set_ylim(0, 125); ax.set_xlim(-0.5, 3.75)
ax.legend(loc="upper left", frameon=False, ncol=3, bbox_to_anchor=(0, 1.13))
ax.text(-0.45, 112, "The system itself: 0% on every type", fontsize=8, color=INK)
fig.savefig(OUT + "figA2_v4b.pdf"); plt.close(fig)
print("ok", ncs)

# ================= Figure 3: results at a glance =================
# skill = (score - chance) / (perfect - chance): 0 = chance or no-effect predictor, 1 = perfect.
acc = lambda a: (a - 0.5) / 0.5
v4a = open(R + "/toy_ladder/v4a_tests.txt").read()
row = lambda name: [float(x) for x in re.search(rf"^{name}\s+(.+)$", v4a, re.M).group(1).split()]
k50 = 3  # column for 50% of nodes destroyed
v4b = open(R + "/toy_ladder/v4b_tests.txt").read()
v4b_rows = [[float(x) for x in re.search(rf"^{t}.*?\)\s+(.+)$", v4b, re.M).group(1).split()] for t in ("B", "C", "D")]
skill_b = lambda col: [max(0.0, 1 - r[col] / r[0]) for r in v4b_rows]   # cols: null lookup ridge self outsider
v4c = open(R + "/toy_ladder/v4c_tests.txt").read()
rings = [float(x) for x in re.search(r"^rings\s+(.+)$", v4c, re.M).group(1).split()]  # raw efcopy outsider outsider+rule
b2 = json.load(open(R + "/transformer/b2_results.json"))["table"]
au = lambda s: float(s.split()[0])
b2d = {r["scorer"]: r for r in b2}
outs = [k for k in b2d if k not in ("target", "copy")]

# rows: (label, sublabel, {series: value or [lo, hi]})
rows3 = [
 ("V3  Self vs. injected change", "as built, then with one stored variable",
  {"sys": 0.0, "cheat": 1.0}),  # identical states -> chance; efference copy -> perfect
 ("V4a  Robust, distributed self", "half of the 21 nodes destroyed",
  {"sys": acc(row("none")[k50]), "cheat": [acc(row("single")[k50]), acc(row("replicated")[k50])]}),
 ("V4b  Novel self-prediction", "three disturbance types never seen",
  {"sys": 1.0, "cheat": [min(skill_b(1) + skill_b(2)), max(skill_b(1) + skill_b(2))], "out": [min(skill_b(4)), max(skill_b(4))]}),
 ("V4c  Re-recognition against a twin", "anchor intact",
  {"sys": acc(rings[1]), "out": acc(rings[3])}),
 ("B2  Detecting inserted tokens", "fluent insertions",
  {"sys": acc(au(b2d["target"]["auroc_fluent"])), "copy": acc(au(b2d["copy"]["auroc_fluent"])),
   "out": [min(acc(au(b2d[k]["auroc_fluent"])) for k in outs), max(acc(au(b2d[k]["auroc_fluent"])) for k in outs)]}),
 ("B2  Detecting inserted tokens", "off-context insertions",
  {"sys": acc(au(b2d["target"]["auroc_offctx"])), "copy": acc(au(b2d["copy"]["auroc_offctx"])),
   "out": [min(acc(au(b2d[k]["auroc_offctx"])) for k in outs), max(acc(au(b2d[k]["auroc_offctx"])) for k in outs)]}),
]
STY = {"sys": dict(marker="o", s=34, color=INK, label="The system itself", z=5, dy=0.0),
       "cheat": dict(marker="^", s=40, color=ORG, label="Cheapest cheat (a record or a lookup)", z=4, dy=0.17),
       "out": dict(marker="s", s=30, color=ACC, label="Outsider (models fitted from outside)", z=4, dy=-0.17),
       "copy": dict(marker="D", s=46, color="none", label="Exact copy of the system", z=6, dy=0.0)}
fig, ax = plt.subplots(figsize=(6.0, 3.6))
n = len(rows3)
for i, (lab, sub, d) in enumerate(rows3):
    y = n - 1 - i
    if i % 2 == 0:
        ax.axhspan(y - 0.5, y + 0.5, color="#f4f4f4", zorder=0, lw=0)
    for key, v in d.items():
        st = STY[key]; yy = y + st["dy"]
        if isinstance(v, list):
            ax.plot(v, [yy, yy], color=st["color"], lw=2.2, alpha=0.55, solid_capstyle="round", zorder=st["z"] - 1)
            xs = v
        else:
            xs = [v]
        for x in xs:
            if key == "copy":
                ax.scatter([x], [yy], marker="D", s=st["s"], facecolors="none", edgecolors="#2e8b57", lw=1.2, zorder=st["z"])
            else:
                ax.scatter([x], [yy], marker=st["marker"], s=st["s"], color=st["color"], zorder=st["z"], lw=0)
ax.set_yticks(range(n), [f"{l}\n{s}" for l, s, _ in rows3][::-1])
for t in ax.get_yticklabels():
    t.set_fontsize(8)
ax.set_ylim(-0.6, n - 0.4); ax.set_xlim(-0.05, 1.05)
ax.axvline(0, color=GREY, lw=0.6, ls=(0, (3, 3))); ax.axvline(1, color=GREY, lw=0.6, ls=(0, (3, 3)))
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["chance", "0.25", "0.5", "0.75", "perfect"])
ax.set_xlabel("Skill, scaled from chance (0) to perfect (1)")
ax.tick_params(axis="y", length=0)
h = [plt.Line2D([], [], marker=STY[k]["marker"], ls="", color=STY[k]["color"] if k != "copy" else "#2e8b57",
                markerfacecolor=(STY[k]["color"] if k != "copy" else "none"), markersize=6, label=STY[k]["label"]) for k in ("sys", "cheat", "out", "copy")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.32, -0.15), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.2)
fig.savefig(OUT + "fig3_glance.pdf"); plt.close(fig)
print("glance rows:")
for r in rows3: print(" ", r[0], r[1], {k: (np.round(v, 3).tolist()) for k, v in r[2].items()})

# ================= Figure: J1 subspaces =================
j1 = json.load(open(R + "/transformer/j1_results.json"))["table"]
SUBS = ["J-top32", "random32", "J-bottom32", "J-top128", "random128", "J-bottom128", "full", "shuffled"]
SL = ["J-space top 32", "random 32", "lowest-gain 32", "J-space top 128", "random 128", "lowest-gain 128", "full state", "shuffled labels"]
MODELS = [("target gpt2-medium", "Target, GPT-2 Medium", INK, "o", 30), ("gpt2-small", "GPT-2 Small (outsider)", ACC, "s", 20),
          ("neo-125m", "GPT-Neo 125M (outsider)", ACC, "^", 22)]
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.7), sharey=True)
for ax, kind, title in zip(axs, ("offctx", "fluent"), ("Off-context insertions", "Fluent insertions")):
    ax.axhline(0.5, color=GREY, lw=0.6, ls=(0, (3, 3)))
    for xi, s in enumerate(SUBS):
        if s.startswith("J-top"):
            ax.axvspan(xi - 0.42, xi + 0.42, color="#e6eefa", lw=0, zorder=0)
    for j, (m, lab, c, mk, sz) in enumerate(MODELS):
        ys = [next(r["auroc"] for r in j1 if r["kind"] == kind and r["model"] == m and r["subspace"] == s) for s in SUBS]
        xs = np.arange(len(SUBS)) + (j - 1) * 0.2
        if m.startswith("target"):
            ax.scatter(xs, ys, marker=mk, s=sz, color=c, zorder=4, label=lab, lw=0)
        else:
            ax.scatter(xs, ys, marker=mk, s=sz, facecolors="white", edgecolors=c, lw=0.9, zorder=3, label=lab)
    ax.set_xticks(range(len(SUBS)), SL, fontsize=7, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_title(title); ax.set_ylim(0.4, 1.0)
axs[0].set_ylabel("Probe AUROC, own vs. inserted")
axs[0].text(-0.4, 0.508, "chance", fontsize=7.5, color=GREY, va="bottom")
axs[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.42), ncol=3, frameon=False, handletextpad=0.3)
fig.subplots_adjust(wspace=0.08)
fig.savefig(OUT + "fig_j1.pdf"); plt.close(fig)

# ================= Figure B1: a sample run =================
def trace(mode):
    s = ssm.Sim(seed=0, mode=mode)
    out = []
    for t in range(5000):
        if t == 2000:
            s.theta = np.mod(s.theta + 0.5, 2 * np.pi)
        s.run(1)
        z = np.mean(np.exp(1j * s.theta))
        out.append((s.t * ssm.dt, np.angle(z), s.anchor, ssm.energy(s.theta, s.p, s.anchor)))
    return np.array(out)
tr = trace("full"); tc = trace("cut")
t = tr[:, 0]
rel = lambda a: np.unwrap(np.angle(np.exp(1j * (a - np.pi))))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
a1.axvline(20, color=GREY, lw=0.6, ls=(0, (3, 3)))
a1.plot(t, rel(tr[:, 1]), color=LG, lw=1.0, label="Mean ring phase")
a1.plot(t, tr[:, 2] - np.pi, color=ACC, lw=1.4, label="Anchor $\\Omega$ (reflexive loop on)")
a1.plot(t, tc[:, 2] - np.pi, color=GREY, lw=0.9, ls=(0, (4, 2)), label="Anchor with the loop cut")
a1.set_ylabel("Phase $-\\ \\pi$ (rad)")
a1.text(20.6, 0.52, "rings rotated by 0.5 rad", fontsize=7.8, color=INK, va="top")
a1.legend(loc="lower left", frameon=False, ncol=3, fontsize=7.6, handlelength=1.6, bbox_to_anchor=(0, 0.98))
w = (t >= 18) & (t <= 32)
a2.axvline(20, color=GREY, lw=0.6, ls=(0, (3, 3)))
a2.step(t, (tr[:, 2] - np.pi) * 1e3, where="post", color=ACC, lw=1.1)
a2.set_ylabel("$\\Omega - \\pi$ (mrad)")
a2.set_xlabel("Time (model units; one step = 0.01)")
ups = t[(np.diff(np.r_[tr[0, 2], tr[:, 2]]) != 0)]
a2.text(0.995, 0.06, f"{len(ups)} reflexive updates, one every 0.5 time units", transform=a2.transAxes, ha="right", fontsize=7.6, color=INK)
fig.align_ylabels([a1, a2])
fig.savefig(OUT + "figB1_run.pdf"); plt.close(fig)
dE = np.abs(np.diff(tc[:, 3])).max()
print("B1: anchor range full", tr[:, 2].min() - np.pi, tr[:, 2].max() - np.pi, "ring phase range", rel(tr[:, 1]).min(), rel(tr[:, 1]).max(),
      "cut-run max step energy change", dE, "energy drift cut", tc[-1, 3] - tc[0, 3])

print("figures written to", OUT, "(paper typography)" if PAPER_STYLE else "(default typography)")
