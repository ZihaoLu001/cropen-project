"""Results charts of the CR-Open project page: the complete physics-v1 study (8 seeds per arm).

Reads the recorded analysis outputs (w5_stats table, w5_gd; frozen code, see their code_sha256 blocks):
  <data>/task/table.json, <data>/n1/table.json      per-seed and across-seed success at epoch 10000 (Table 2)
  <data>/v1/gd_task.json, <data>/v1/gd_n1.json      the G-D ablation calls at epoch 6000
  <data>/v1/cost.json                               N1 minus task-only at epoch 10000 (quoted in a caption)
and writes
  tools/data/v1_numbers.json                        every number the page shows, with the sha256 of each source
  static/images/v1_success_e10000.svg               per-family success, one row per arm: seed dots (beeswarm),
                                                    mean + 95 % CI
  static/images/v1_ablation_m3_e6000.svg            M3 at epoch 6000: full recipe vs each component removed, with the
                                                    change in mean below each removal
  (with --html) the per-seed success table between the "v1 success table" markers of index.html
The page calls the two physics-v1 arms "without / with posture regularization" (the N1 terms: a relaxed reset pose, a
flexion floor, posture and action-rate costs) and the families "closure types".
Intervals are the ones the analysis recorded: 95 % t-intervals of the mean over seeds (df = n - 1), clipped to
[0, 1]; an arm whose seeds all have one value takes Agresti-Coull at that value over its pooled episodes. The change
intervals are Welch, or a shifted one-sample t when one arm is constant. Nothing is recomputed except the display
rounding (means 2 decimals in the charts; the table keeps 4). The pre-registered calls ("necessary") are not drawn;
the page states them in words.

Text is Noto Sans (the page's font; tools/fonts/ holds a Latin subset of the OFL font), converted to paths.
Seeds are placed as a beeswarm in screen pixels (the chart is drawn for 768 px width): x only spreads the seeds of a
column, never their values, and centre spacing is at least the ring's outer diameter + 2 px.

Usage: python tools/make_v1_charts.py --data DIR [--preview-dir DIR] [--html index.html]
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
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "static" / "images"
FAMS = ["M1", "M2", "M3", "M6"]
ABL = [("hist1", "no history"), ("nounlock", "no unlock bonus"), ("nocurr", "no curriculum")]
ARM_TITLE = {"task-only": "Without posture regularization", "N1": "With posture regularization"}

for f in sorted((ROOT / "tools" / "fonts").glob("*.ttf")):
    font_manager.fontManager.addfont(str(f))

# dataviz reference palette: slot 1 for the plotted series; recessive chrome
BLUE, GRAY = "#2a78d6", "#898781"
INK, INK2, GRID, RULE = "#0b0b0b", "#52514e", "#e8e7e1", "#e3e1dc"
DPI = 100                       # 7.68 in -> 768 px on the page
RING_MS, RING_MEW = 5.2, 1.2    # points: outer diameter (5.2 + 1.2) pt = 8.9 px
PITCH = 11.0                    # px between seed centres (>= outer diameter + 2 px)

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
        for ab, _ in ABL:
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
def swarm(ax, vals):
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
            if dy < PITCH:
                h = math.sqrt(PITCH ** 2 - dy ** 2)
                blocks.append((px - h, px + h))
        cands = [0.0] + [b[0] for b in blocks] + [b[1] for b in blocks]
        ok = [c for c in cands if all(not (lo + 1e-6 < c < hi - 1e-6) for lo, hi in blocks)]
        bal = sum(p[0] for p in placed)
        best = min(ok, key=lambda c: (round(abs(c), 3), (c > 0) if bal > 0 else (c < 0)))
        placed.append((best, y))
        out[i] = best
    shift = (max(out) + min(out)) / 2
    return [(o - shift) * x_per_px for o in out]


def column(ax, xc, vals, mean, lo, hi, color, label_y=1.06, ci_dx=0.32, sw_dx=-0.10):
    """one column: hollow seed rings (beeswarm, 2 px white halo), the 95 % interval as a thin bar and the mean as a
    black tick to their right, the mean (2 decimals) above"""
    offs = swarm(ax, vals)
    to_px = ax.transData.transform
    px = to_px((1, 0))[0] - to_px((0, 0))[0]
    r = (RING_MS + RING_MEW) / 72 * DPI / 2
    room = (ci_dx - 0.05 - sw_dx) - (r + 4) / px             # rightmost seed centre that keeps 4 px to the tick
    excess = max(offs) - room
    if excess > 0:
        offs = [o - excess for o in offs]
    for dx, v in zip(offs, vals):
        ax.plot(xc + sw_dx + dx, v, "o", ms=RING_MS + RING_MEW + 2.9, mfc="white", mec="none", zorder=3.9,
                clip_on=False)
        ax.plot(xc + sw_dx + dx, v, "o", ms=RING_MS, mfc="white", mec=color, mew=RING_MEW, zorder=4, clip_on=False)
    xi = xc + ci_dx
    ax.plot([xi, xi], [lo, hi], color=color, lw=1.5, alpha=0.7, solid_capstyle="butt", zorder=3)
    ax.plot([xi - 0.05, xi + 0.05], [mean, mean], color=INK, lw=2.0, solid_capstyle="butt", zorder=5)
    ax.text(xi, label_y, f"{mean:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)


def style(ax, ymax=1.13):
    ax.set_ylim(-0.05, ymax)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    for y in [0, 0.25, 0.5, 0.75, 1.0]:
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5, pad=3)
    ax.tick_params(axis="x", labelsize=9.5, pad=5, colors=INK)


def legend(fig, x, y, color):
    """one legend row (figure fractions): the identity of the marks is described in words, never by colour alone"""
    W, H = fig.get_size_inches()
    fig.lines.append(plt.Line2D([x + 0.06 / W], [y + 0.055 / H], marker="o", ms=RING_MS, mfc="white", mec=color,
                                mew=RING_MEW, transform=fig.transFigure, ls=""))
    fig.text(x + 0.16 / W, y, "one training seed", fontsize=8.5, color=INK2, va="bottom")
    x2 = x + 1.55 / W
    fig.lines.append(plt.Line2D([x2, x2], [y - 0.03 / H, y + 0.15 / H], color=color, lw=1.5, alpha=0.7,
                                transform=fig.transFigure))
    fig.lines.append(plt.Line2D([x2 - 0.05 / W, x2 + 0.05 / W], [y + 0.06 / H, y + 0.06 / H], color=INK, lw=2.0,
                                transform=fig.transFigure))
    fig.text(x2 + 0.12 / W, y, "mean over seeds (number above) and its 95% interval", fontsize=8.5, color=INK2,
             va="bottom")


def save(fig, stem, preview):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None})
    (OUT / f"{stem}.svg").write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    if preview:
        fig.savefig(Path(preview) / f"{stem}.png", dpi=200)
        fig.savefig(Path(preview) / f"{stem}_page.png", dpi=DPI)
    w, h = fig.get_size_inches()
    plt.close(fig)
    return round(w * DPI), round(h * DPI)


def chart_success(nums, preview):
    """two rows (one per arm), five columns each; the rows are stacked so that eight seeds at the same rate fit
    side by side at full spacing"""
    W, L, R = 7.68, 0.66, 0.10
    title_h, legend_h, sub_h, ax_h, tick_h, gap = 0.36, 0.34, 0.30, 1.62, 0.46, 0.14
    H = title_h + legend_h + 2 * (sub_h + ax_h + tick_h) + gap + 0.04
    fig = plt.figure(figsize=(W, H))
    xs = [0, 1, 2, 3, 4.42]
    y_top = H - title_h - legend_h
    for r, arm in enumerate(("task-only", "N1")):
        top = y_top - r * (sub_h + ax_h + tick_h + gap) - sub_h
        ax = fig.add_axes((L / W, (top - ax_h) / H, (W - L - R) / W, ax_h / H))
        ax.set_xlim(-0.56, 4.92)
        style(ax)
        d = nums["success_e10000"][arm]["families"]
        for x, f in zip(xs, FAMS + ["balanced"]):
            a = d[f]
            column(ax, x, a["per_seed"], a["mean"], a["lo"], a["hi"], BLUE)
        ax.axvline(3.71, ymin=0.02, ymax=0.9, color=RULE, lw=1.0, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(["M1\nplain thread", "M2\npush-and-turn", "M3\nsqueeze-and-turn", "M6\nstuck seal",
                            "balanced\n(mean of 4)"], linespacing=1.15, va="top")
        ax.text(-0.53, 1.06, "mean", fontsize=8, color=INK2, va="bottom")
        fig.text(L / W, (top + 0.10) / H, ARM_TITLE[arm], fontsize=10, color=INK, va="bottom")
        ax.set_ylabel("success rate", fontsize=8.5, color=INK2, labelpad=6)
    fig.text(L / W, (H - 0.10) / H, "Success per closure type at epoch 10,000 · physics v1 · 8 seeds per variant",
             fontsize=10.5, color=INK, weight="bold", va="top")
    legend(fig, L / W, (H - title_h - legend_h + 0.08) / H, BLUE)
    return save(fig, "v1_success_e10000", preview)


def chart_ablation(nums, preview):
    W, L, R, mid = 7.68, 0.66, 0.10, 0.22
    title_h, legend_h, sub_h, ax_h, tick_h, rows_h = 0.36, 0.34, 0.30, 1.95, 0.50, 0.40
    H = title_h + legend_h + sub_h + ax_h + tick_h + rows_h + 0.04
    fig = plt.figure(figsize=(W, H))
    pw = (W - L - R - mid) / 2
    top = H - title_h - legend_h - sub_h
    xs = [0, 1, 2, 3]
    for p, arm in enumerate(("task-only", "N1")):
        x0 = L + p * (pw + mid)
        ax = fig.add_axes((x0 / W, (top - ax_h) / H, pw / W, ax_h / H))
        ax.set_xlim(-0.55, 3.55)
        style(ax)
        if p:
            ax.set_yticklabels([])
        rows_tr = blended_transform_factory(ax.transData, fig.dpi_scale_trans)   # x: data, y: inches
        row_change = rows_h + 0.0
        g = nums["gd_m3_e6000"][arm]
        pr = g["primary"]
        column(ax, 0, pr["per_seed"], pr["mean"], pr["lo"], pr["hi"], BLUE, ci_dx=0.36, sw_dx=-0.08)
        for i, (ab, _) in enumerate(ABL, start=1):
            a = g["ablations"][ab]
            column(ax, i, a["per_seed"], a["mean"], a["lo"], a["hi"], GRAY, ci_dx=0.30, sw_dx=-0.06)
            ch = -a["gap_pp"]
            ax.text(i + 0.08, row_change, f"{ch:+.1f} pp".replace("-", "−"), ha="center", va="top",
                    fontsize=8.5, color=INK, transform=rows_tr)
        # row header in the full-recipe slot, which has no change of its own
        ax.text(0.12, row_change, "change in\nmean", ha="center", va="top", fontsize=8, color=INK2,
                style="italic", linespacing=1.1, transform=rows_tr)
        ax.set_xticks([x + 0.08 for x in xs])
        ax.set_xticklabels(["full recipe\n(8 seeds)", "no history\n(H = 1)\n(3 seeds)", "no unlock\nbonus\n(3 seeds)",
                            "no\ncurriculum\n(3 seeds)"], fontsize=8.5, linespacing=1.15, va="top")
        ax.text(-0.52, 1.06, "mean", fontsize=8, color=INK2, va="bottom")
        fig.text(x0 / W, (top + 0.10) / H, ARM_TITLE[arm], fontsize=10, color=INK, va="bottom")
        if not p:
            ax.set_ylabel("M3 success rate", fontsize=8.5, color=INK2, labelpad=6)
    fig.text(L / W, (H - 0.10) / H, "Squeeze-and-turn (M3) at epoch 6,000 · one component removed · physics v1",
             fontsize=10.5, color=INK, weight="bold", va="top")
    legend(fig, L / W, (H - title_h - legend_h + 0.08) / H, GRAY)
    return save(fig, "v1_ablation_m3_e6000", preview)


# ---------------------------------------------------------------------------------------------------- table views
def num(v, d=4):
    return f"{v:.{d}f}"


def table_success(nums) -> str:
    cols = FAMS + ["balanced"]
    rows = ['<table class="table is-narrow result-table">',
            "<caption>Success rate of each seed's final checkpoint (epoch 10,000), physics v1, with the mean ± SD "
            "over the eight seeds and the 95% t-interval of the mean. Balanced: the mean over the four types."
            "</caption>",
            '<thead><tr><th scope="col">Seed</th>' + "".join(f'<th scope="col">{c}</th>' for c in cols)
            + "</tr></thead>", "<tbody>"]
    for arm in ("task-only", "N1"):
        d = nums["success_e10000"][arm]
        rows.append(f'<tr class="group-row"><th scope="rowgroup" colspan="{len(cols) + 1}">{ARM_TITLE[arm]}</th></tr>')
        for i, s in enumerate(d["seeds"]):
            rows.append(f'<tr><th scope="row">{s}</th>'
                        + "".join(f"<td>{num(d['families'][c]['per_seed'][i])}</td>" for c in cols) + "</tr>")
        rows.append('<tr class="summary-row"><th scope="row">mean ± SD</th>'
                    + "".join(f"<td>{num(d['families'][c]['mean'])} ± {num(d['families'][c]['sd'])}</td>"
                              for c in cols) + "</tr>")
        rows.append('<tr class="summary-row"><th scope="row">95% interval</th>'
                    + "".join(f"<td>{num(d['families'][c]['lo'])} to {num(d['families'][c]['hi'])}</td>"
                              for c in cols) + "</tr>")
    rows += ["</tbody>", "</table>"]
    return "\n".join(rows)


def patch_html(path: Path, marker: str, snippet: str):
    """replace the text between the marker comments; the page keeps its own line endings"""
    raw = path.read_bytes().decode("utf-8")
    eol = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    b, e = f"<!-- BEGIN {marker} (tools/make_v1_charts.py) -->", f"<!-- END {marker} -->"
    i, j = text.index(b) + len(b), text.index(e)
    text = text[:i] + "\n" + snippet + "\n" + text[j:]
    path.write_bytes(text.replace("\n", eol).encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--preview-dir", default=None)
    ap.add_argument("--html", type=Path, default=None, help="write the per-seed success table into this page")
    a = ap.parse_args()
    found = {f.name for f in font_manager.fontManager.ttflist if f.name == "Noto Sans"}
    assert found, "Noto Sans not registered (tools/fonts/)"
    nums = extract(a.data)
    (ROOT / "tools" / "data").mkdir(exist_ok=True)
    (ROOT / "tools" / "data" / "v1_numbers.json").write_text(json.dumps(nums, indent=1), encoding="utf-8",
                                                              newline="\n")
    print("v1_success_e10000.svg", chart_success(nums, a.preview_dir))
    print("v1_ablation_m3_e6000.svg", chart_ablation(nums, a.preview_dir))
    if a.html:
        patch_html(a.html, "v1 success table", table_success(nums))
    print("ok")


if __name__ == "__main__":
    main()
