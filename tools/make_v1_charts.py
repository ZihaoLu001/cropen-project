"""Results chart of the CR-Open project page: the complete physics-v1 study (8 seeds per variant).

Reads the recorded analysis outputs (w5_stats table, w5_gd; frozen code, see their code_sha256 blocks):
  <data>/task/table.json, <data>/n1/table.json      per-seed and across-seed success at epoch 10000 (Table 2)
  <data>/v1/gd_task.json, <data>/v1/gd_n1.json      the G-D ablations at epoch 6000 (quoted in the page's text and
                                                    its "Study details"; no chart)
  <data>/v1/cost.json                               N1 minus task-only at epoch 10000 (kept for the paper)
and writes
  tools/data/v1_numbers.json                        every number the page quotes, with the sha256 of each source
  static/images/v1_success_e10000.svg               success per closure type, both variants side by side in one
                                                    panel: seed dots (beeswarm), mean + 95 % CI, mean in percent
  static/images/v1_success_e10000_phone.svg         the same, one small panel per type (2 x 2), for screens up to
                                                    600 px wide
The page calls the two physics-v1 arms "without / with posture regularization" (the N1 terms: a relaxed reset pose, a
flexion floor, posture and action-rate costs) and the families "closure types". Without posture regularization is
drawn in the blue of the method figure's Observation group, with it in dark grey (orange already means squeeze).
Intervals are the ones the analysis recorded: 95 % t-intervals of the mean over seeds (df = n - 1), clipped to
[0, 1] (every column of the chart); an arm whose seeds all have one value would take Agresti-Coull at that value over
its pooled episodes (the no-curriculum and N1 one-step-history ablations). Nothing is recomputed except the display
rounding (means in whole percent).

Text is Noto Sans (the page's font; tools/fonts/ holds a Latin subset of the OFL font), converted to paths.
Seeds are placed as a beeswarm in screen pixels (drawn for 820 px, the width of the page's results column, and
420 px for phones): x only spreads the seeds of a column, never their values, and centre spacing is at least the
ring's outer diameter + 1.5 px.

Usage: python tools/make_v1_charts.py --data DIR [--preview-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "static" / "images"
FAMS = ["M1", "M2", "M3", "M6"]
ABL = ("hist1", "nounlock", "nocurr")              # one component removed: observation history, unlock bonus, curriculum
TYPE_NAME = {"M1": "Plain Thread", "M2": "Push-and-Turn", "M3": "Squeeze-and-Turn", "M6": "Stuck Seal"}
ARM_TITLE = {"task-only": "Without posture regularization", "N1": "With posture regularization"}

for f in sorted((ROOT / "tools" / "fonts").glob("*.ttf")):
    font_manager.fontManager.addfont(str(f))

# dataviz reference palette: slot 1 (the method figure's Observation blue) and a dark grey for the two variants
BLUE, GRAY = "#2a78d6", "#52514e"
INK, INK2, GRID, RULE = "#0b0b0b", "#52514e", "#e8e7e1", "#e3e1dc"
DPI = 100                       # 8.2 in -> 820 px on the page
RING_MS, RING_MEW = 3.8, 0.9    # points: outer diameter (3.8 + 0.9) pt = 6.5 px
PITCH = 8.0                     # px between seed centres (>= outer diameter + 1.5 px)

plt.rcParams.update({
    "font.family": "Noto Sans", "svg.fonttype": "path", "svg.hashsalt": "cropen-v1", "figure.dpi": DPI,
    "axes.linewidth": 0, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "font.size": 9, "savefig.facecolor": "white",
})


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract(data: Path) -> dict:
    src = {k: data / k for k in ("task/table.json", "n1/table.json", "v1/gd_task.json", "v1/gd_n1.json",
                                 "v1/cost.json")}
    out = {"sources": {k: sha(p) for k, p in src.items()}, "success_e10000": {}, "gd_m3_e6000": {},
           "cost_n1_minus_task_e10000_pp": {}}
    cost = json.loads(src["v1/cost.json"].read_text(encoding="utf-8"))
    assert cost["complete"] and not any(cost["failed"].values())
    for f, r in cost["rows"].items():   # N1 minus task-only, Welch 95 % interval (reported, no rule)
        out["cost_n1_minus_task_e10000_pp"][f] = [100 * r["diff"], 100 * r["lo"], 100 * r["hi"]]
    for arm, key in (("task-only", "task/table.json"), ("N1", "n1/table.json")):
        t = json.loads(src[key].read_text(encoding="utf-8"))
        assert t["complete"] and not t["failed"], key
        acr = t["across"]["epoch10000"]
        arm_out = {"runs": t["runs"], "seeds": t["seeds"], "families": {}}
        for f in FAMS + ["balanced"]:
            a = acr[f]
            vals = a["rates"] if f in FAMS else a["values"]
            arm_out["families"][f] = {"per_seed": vals, "mean": a["mean"], "sd": a["sd"], "lo": a["lo"],
                                      "hi": a["hi"], "method": a["method"]}
        out["success_e10000"][arm] = arm_out
    for arm, key in (("task-only", "v1/gd_task.json"), ("N1", "v1/gd_n1.json")):
        g = json.loads(src[key].read_text(encoding="utf-8"))
        assert g["complete"] and g["epoch"] == 6000 and not g["failed"], key
        arm_out = {"wording_suffix": g["wording_suffix"], "primary_seeds": g["primary"]["seeds"], "ablations": {},
                   "other_families": {}}
        for ab in ABL:
            r = g["results"][ab]["M3"]
            arm_out["primary"] = {"per_seed": r["primary"]["rates"], "mean": r["primary"]["mean"],
                                  "lo": r["primary"]["lo"], "hi": r["primary"]["hi"],
                                  "method": r["primary"]["method"]}
            arm_out["ablations"][ab] = {
                "change": g["ablations"][ab]["change"], "seeds": g["ablations"][ab]["seeds"],
                "per_seed": r["ablation"]["rates"], "mean": r["ablation"]["mean"], "lo": r["ablation"]["lo"],
                "hi": r["ablation"]["hi"], "method": r["ablation"]["method"], "gap_pp": r["gap_pp"],
                "necessary": r["necessary"]["necessary"], "difference_resolved": r["difference_resolved"],
                "diff_pp": [100 * r["difference"]["lo"], 100 * r["difference"]["hi"]],
                "diff_method": r["difference"]["method"], "wording": r["wording"]}
            # per-seed rates of the other families (quoted in the caption: whole runs that open nothing)
            arm_out["other_families"][ab] = {
                f: {"per_seed": g["results"][ab][f]["ablation"]["rates"],
                    "necessary": g["results"][ab][f]["necessary"]["necessary"]} for f in ("M1", "M2", "M6")}
        out["gd_m3_e6000"][arm] = arm_out
    return out


# ---------------------------------------------------------------------------------------------------- marks
def swarm(ax, vals, pitch):
    """beeswarm x offsets (data units) for the values of one column: each seed keeps its value, seeds that would
    touch are pushed sideways to the nearest free place; the column is centred on its extent"""
    to_px = ax.transData.transform
    x_per_px = 1.0 / (to_px((1, 0))[0] - to_px((0, 0))[0])
    ys = [to_px((0, v))[1] for v in vals]
    placed = []                                      # (x_px, y_px)
    out = [0.0] * len(vals)
    for i in sorted(range(len(vals)), key=lambda k: (-vals[k], k)):
        y = ys[i]
        blocks = []
        for (px, py) in placed:
            dy = abs(py - y)
            if dy < pitch:
                h = math.sqrt(pitch ** 2 - dy ** 2)
                blocks.append((px - h, px + h))
        cands = [0.0] + [b[0] for b in blocks] + [b[1] for b in blocks]
        ok = [c for c in cands if all(not (lo + 1e-6 < c < hi - 1e-6) for lo, hi in blocks)]
        bal = sum(p[0] for p in placed)
        best = min(ok, key=lambda c: (round(abs(c), 3), (c > 0) if bal > 0 else (c < 0)))
        placed.append((best, y))
        out[i] = best
    shift = (max(out) + min(out)) / 2
    return [(o - shift) * x_per_px for o in out]


def column(ax, xc, vals, mean, lo, hi, color, sw_px=-8.0, ci_px=31.0, tick_px=5.0, label_y=1.06, fs=8.5):
    """one column: hollow seed rings (beeswarm centred sw_px from xc, 2 px white halo), the 95 % interval as a thin
    bar ci_px right of xc with the mean as a black tick, and the mean in percent above it"""
    to_px = ax.transData.transform
    px = to_px((1, 0))[0] - to_px((0, 0))[0]
    offs = swarm(ax, vals, PITCH)
    r = (RING_MS + RING_MEW) / 72 * DPI / 2
    room = (ci_px - tick_px - 4 - r - sw_px) / px              # rightmost seed centre that keeps 4 px to the tick
    excess = max(offs) - room
    if excess > 0:
        offs = [o - excess for o in offs]
    xs = xc + sw_px / px
    for dx, v in zip(offs, vals):
        ax.plot(xs + dx, v, "o", ms=RING_MS + RING_MEW + 2.9, mfc="white", mec="none", zorder=3.9, clip_on=False)
        ax.plot(xs + dx, v, "o", ms=RING_MS, mfc="white", mec=color, mew=RING_MEW, zorder=4, clip_on=False)
    xi, tk = xc + ci_px / px, tick_px / px
    ax.plot([xi, xi], [lo, hi], color=color, lw=1.5, alpha=0.75, solid_capstyle="butt", zorder=3)
    ax.plot([xi - tk, xi + tk], [mean, mean], color=INK, lw=2.0, solid_capstyle="butt", zorder=5)
    ax.text(xi, label_y, f"{100 * mean:.0f}%", ha="center", va="bottom", fontsize=fs, color=INK)


def style(ax, ymax=1.13, labels=True):
    ax.set_ylim(-0.05, ymax)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"] if labels else [])
    for y in [0, 0.25, 0.5, 0.75, 1.0]:
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5, pad=3)
    ax.tick_params(axis="x", labelsize=9.5, pad=5, colors=INK)


def legend(fig, rows):
    """legend entries (figure inches, from the top left): the identity of the marks is said in words, never by
    colour alone. rows: [(x_in, y_in, kind, colour, text)], kind 'ring' or 'bar'"""
    W, H = fig.get_size_inches()
    for x, y, kind, color, text in rows:
        yf = (H - y) / H
        if kind == "ring":
            fig.lines.append(plt.Line2D([(x + 0.05) / W], [yf], marker="o", ms=RING_MS, mfc="white", mec=color,
                                        mew=RING_MEW, transform=fig.transFigure, ls=""))
        else:
            fig.lines.append(plt.Line2D([(x + 0.05) / W] * 2, [yf - 0.08 / H, yf + 0.08 / H], color=color, lw=1.5,
                                        alpha=0.75, transform=fig.transFigure))
            fig.lines.append(plt.Line2D([(x + 0.0) / W, (x + 0.10) / W], [yf, yf], color=INK, lw=2.0,
                                        transform=fig.transFigure))
        fig.text((x + 0.16) / W, yf, text, fontsize=8.5, color=INK2, va="center")


def text_w(fig, s, size=8.5):
    t = fig.text(0, 0, s, fontsize=size)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / DPI
    t.remove()
    return w


def save(fig, stem, preview, page_px=None):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None})
    (OUT / f"{stem}.svg").write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    w, h = fig.get_size_inches()
    if preview:
        fig.savefig(Path(preview) / f"{stem}.png", dpi=200)
        fig.savefig(Path(preview) / f"{stem}_page.png", dpi=DPI * (page_px or w * DPI) / (w * DPI))
    plt.close(fig)
    return round(w * DPI), round(h * DPI)


VARIANTS = (("task-only", BLUE), ("N1", GRAY))


def legend_rows(fig, x0, y0, stacked):
    items = [("ring", BLUE, ARM_TITLE["task-only"]), ("ring", GRAY, ARM_TITLE["N1"]),
             ("bar", "#898781", "Mean (number above) and 95% interval")]
    if stacked:
        return [(x0, y0 + 0.2 * i, k, c, t) for i, (k, c, t) in enumerate(items)]
    rows, x = [], x0
    for k, c, t in items:
        rows.append((x, y0, k, c, t))
        x += 0.16 + text_w(fig, t) + 0.34
    return rows


def chart_success(nums, preview):
    """one panel: the four closure types, each with both variants side by side (eight seeds each)"""
    W, L, R = 8.20, 0.62, 0.10
    legend_h, ax_h, tick_h = 0.40, 1.86, 0.46
    H = legend_h + ax_h + tick_h + 0.04
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((L / W, (tick_h + 0.04) / H, (W - L - R) / W, ax_h / H))
    ax.set_xlim(-0.5, 3.5)
    style(ax)
    for t, f in enumerate(FAMS):
        for k, (arm, color) in enumerate(VARIANTS):
            a = nums["success_e10000"][arm]["families"][f]
            column(ax, t + (k - 0.5) * 0.46, a["per_seed"], a["mean"], a["lo"], a["hi"], color)
    for xv in (0.5, 1.5, 2.5):
        ax.axvline(xv, ymin=0.02, ymax=0.97, color=RULE, lw=1.0, zorder=0)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{f}\n{TYPE_NAME[f]}" for f in FAMS], linespacing=1.2, va="top")
    ax.set_ylabel("Success rate", fontsize=8.5, color=INK2, labelpad=6)
    for row in [legend_rows(fig, L, 0.17, stacked=False)]:
        legend(fig, row)
    return save(fig, "v1_success_e10000", preview)


def chart_success_phone(nums, preview):
    """the same numbers for screens up to 600 px: one small panel per type (2 x 2), the variants labelled below"""
    W, L, R, mid = 4.20, 0.58, 0.06, 0.14
    legend_h, title_h, ax_h, tick_h, gap = 0.70, 0.26, 1.30, 0.22, 0.10
    H = legend_h + 2 * (title_h + ax_h + tick_h) + gap + 0.04
    fig = plt.figure(figsize=(W, H))
    pw = (W - L - R - mid) / 2
    for i, f in enumerate(FAMS):
        r, k = divmod(i, 2)
        x0 = L + k * (pw + mid)
        top = H - legend_h - r * (title_h + ax_h + tick_h + gap) - title_h
        ax = fig.add_axes((x0 / W, (top - ax_h) / H, pw / W, ax_h / H))
        ax.set_xlim(-0.5, 1.5)
        style(ax, labels=(k == 0))
        for j, (arm, color) in enumerate(VARIANTS):
            a = nums["success_e10000"][arm]["families"][f]
            column(ax, j, a["per_seed"], a["mean"], a["lo"], a["hi"], color)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Without", "With"], fontsize=8.5, color=INK2)
        ax.tick_params(axis="x", pad=3)
        fig.text((x0 + pw / 2) / W, (top + 0.07) / H, f"{f} {TYPE_NAME[f]}", fontsize=9.5, color=INK,
                 weight="bold", ha="center", va="bottom")
        if k == 0:
            ax.set_ylabel("Success rate", fontsize=8.5, color=INK2, labelpad=4)
    legend(fig, legend_rows(fig, L, 0.14, stacked=True))
    return save(fig, "v1_success_e10000_phone", preview, page_px=343)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--preview-dir", default=None)
    a = ap.parse_args()
    found = {f.name for f in font_manager.fontManager.ttflist if f.name == "Noto Sans"}
    assert found, "Noto Sans not registered (tools/fonts/)"
    nums = extract(a.data)
    (ROOT / "tools" / "data").mkdir(exist_ok=True)
    (ROOT / "tools" / "data" / "v1_numbers.json").write_text(json.dumps(nums, indent=1), encoding="utf-8",
                                                              newline="\n")
    print("v1_success_e10000.svg", chart_success(nums, a.preview_dir))
    print("v1_success_e10000_phone.svg", chart_success_phone(nums, a.preview_dir))
    print("ok")


if __name__ == "__main__":
    main()
