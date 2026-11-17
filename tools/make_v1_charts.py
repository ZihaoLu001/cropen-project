"""Results charts of the CR-Open project page: the complete physics-v1 study (8 seeds per arm).

Reads the recorded analysis outputs (w5_stats table, w5_gd; frozen code, see their code_sha256 blocks):
  <data>/task/table.json, <data>/n1/table.json      per-seed and across-seed success at epoch 10000 (Table 2)
  <data>/v1/gd_task.json, <data>/v1/gd_n1.json      the G-D ablation calls at epoch 6000
  <data>/v1/cost.json                               N1 minus task-only at epoch 10000 (quoted in a caption)
and writes
  tools/data/v1_numbers.json                        every number the page shows, with the sha256 of each source
  static/images/v1_success_e10000.svg               per-family success, both arms, seed dots + mean + 95 % CI
  static/images/v1_ablation_m3_e6000.svg            M3 at epoch 6000: full recipe vs each component removed
Intervals are the ones the analysis recorded: 95 % t-intervals of the mean over seeds (df = n - 1), clipped to
[0, 1]; an arm whose seeds all have one value takes Agresti-Coull at that value over its pooled episodes.
Nothing is recomputed except the display rounding.

Usage: python tools/make_v1_charts.py --data DIR [--preview-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "static" / "images"
FAMS = ["M1", "M2", "M3", "M6"]
ABL = [("hist1", "history $H{=}1$"), ("nounlock", "no unlock bonus"), ("nocurr", "no curriculum")]

# dataviz reference palette: slot 1 for the plotted series; recessive chrome
BLUE, GRAY = "#2a78d6", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "Segoe UI", "svg.fonttype": "path", "svg.hashsalt": "cropen-v1",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
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
        arm_out = {"wording_suffix": g["wording_suffix"], "primary_seeds": g["primary"]["seeds"], "ablations": {}}
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
                "diff_pp": [100 * r["difference"]["lo"], 100 * r["difference"]["hi"]], "wording": r["wording"]}
        out["gd_m3_e6000"][arm] = arm_out
    return out


def offsets(n, half=0.17):
    return [0.0] if n == 1 else [-half + 2 * half * i / (n - 1) for i in range(n)]


def strip(ax, x, vals, mean, lo, hi, color, label_fmt="{:.3f}", label=True, label_y=None):
    """one column: hollow seed dots (seed order, evenly spread), the mean as a tick, the 95 % interval as a bar"""
    for dx, v in zip(offsets(len(vals)), vals):
        ax.plot(x - 0.06 + dx, v, "o", ms=5.2, mfc="white", mec=color, mew=1.3, zorder=4)
    xi = x + 0.26
    ax.plot([xi, xi], [lo, hi], color=color, lw=2.0, solid_capstyle="round", zorder=3)
    ax.plot([xi - 0.07, xi + 0.07], [mean, mean], color=INK, lw=2.0, solid_capstyle="butt", zorder=5)
    if label:
        ax.text(xi, label_y if label_y is not None else hi + 0.035, label_fmt.format(mean), ha="center",
                va="bottom", fontsize=8.5, color=INK)


def style(ax, ymax=1.0):
    ax.set_ylim(-0.04, ymax)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
    for y in [0, 0.25, 0.5, 0.75, 1.0]:
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5, pad=2)
    ax.tick_params(axis="x", labelsize=9.5, pad=6, colors=INK)


def save(fig, stem, preview):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None})
    (OUT / f"{stem}.svg").write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    if preview:
        fig.savefig(Path(preview) / f"{stem}.png", dpi=150)
    plt.close(fig)


def chart_success(nums, preview):
    fig, axes = plt.subplots(1, 2, figsize=(7.68, 3.7), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.82, bottom=0.17, wspace=0.08)
    titles = {"task-only": "Task reward only", "N1": "With posture regularisers (N1)"}
    for ax, arm in zip(axes, ("task-only", "N1")):
        d = nums["success_e10000"][arm]["families"]
        style(ax, 1.13)
        for i, f in enumerate(FAMS + ["balanced"]):
            x = i + (0.35 if f == "balanced" else 0)
            a = d[f]
            strip(ax, x, a["per_seed"], a["mean"], a["lo"], a["hi"], BLUE, label_y=1.035, label_fmt="{:.4f}")
        ax.axvline(3.67, color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=0)
        ax.set_xticks([0, 1, 2, 3, 4.35])
        ax.set_xticklabels(["M1", "M2", "M3", "M6", "balanced"])
        ax.set_xlim(-0.55, 4.85)
        ax.set_title(titles[arm], fontsize=10, color=INK, loc="left", pad=16)
        ax.text(-0.5, 1.13, "mean", fontsize=8, color=INK2, va="bottom")
    axes[0].set_ylabel("success rate (first episodes)", fontsize=9, color=INK2)
    fig.text(0.075, 0.965, "Per-family success at epoch 10,000 · physics v1 · 8 seeds per arm", fontsize=10.5,
             color=INK, weight="semibold", va="top")
    legend(fig, 0.035, BLUE)
    save(fig, "v1_success_e10000", preview)


def legend(fig, y, color, x=0.075):
    """one legend row: the identity of the marks is described in words, never by colour alone"""
    fig.lines.append(plt.Line2D([x + 0.008], [y + 0.012], marker="o", ms=5.2, mfc="white", mec=color, mew=1.3,
                                transform=fig.transFigure, ls=""))
    fig.text(x + 0.022, y, "one training seed", fontsize=8.5, color=INK2, va="bottom")
    x2 = x + 0.19
    fig.lines.append(plt.Line2D([x2, x2], [y - 0.006, y + 0.034], color=color, lw=2.0, transform=fig.transFigure))
    fig.lines.append(plt.Line2D([x2 - 0.009, x2 + 0.009], [y + 0.014, y + 0.014], color=INK, lw=2.0,
                                transform=fig.transFigure))
    fig.text(x2 + 0.016, y, "mean over seeds (number above) and its 95 % interval", fontsize=8.5, color=INK2,
             va="bottom")


def chart_ablation(nums, preview):
    fig, axes = plt.subplots(1, 2, figsize=(7.68, 4.5), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.85, bottom=0.36, wspace=0.08)
    titles = {"task-only": "Task reward only", "N1": "With posture regularisers (N1)"}
    calls = {  # the recorded wording (README 状态 2026-11-17 14:12:38 +0100), short form
        ("task-only", "hist1"): "necessary", ("task-only", "nocurr"): "necessary",
        ("task-only", "nounlock"): "not necessary\n(intervals overlap)",
        ("N1", "nocurr"): "necessary", ("N1", "hist1"): "raised M3",
        ("N1", "nounlock"): "no difference\nresolved",
    }
    for ax, arm in zip(axes, ("task-only", "N1")):
        g = nums["gd_m3_e6000"][arm]
        style(ax, 1.13)
        p = g["primary"]
        strip(ax, 0, p["per_seed"], p["mean"], p["lo"], p["hi"], BLUE, label_y=1.035)
        for i, (ab, _) in enumerate(ABL, start=1):
            a = g["ablations"][ab]
            strip(ax, i, a["per_seed"], a["mean"], a["lo"], a["hi"], GRAY, label_y=1.035)
            gap = -a["gap_pp"]
            txt = f"{gap:+.1f} pp\n{calls[(arm, ab)]}"
            ax.text(i + 0.1, -0.42, txt.replace("-", "−"), ha="center", va="top", fontsize=8.5,
                    color=INK, linespacing=1.25)
        ax.set_xticks([0.1, 1.1, 2.1, 3.1])
        ax.set_xticklabels(["full recipe\n(8 seeds)", "history H = 1\n(3 seeds)", "no unlock\nbonus\n(3 seeds)",
                            "no curriculum\n(3 seeds)"], fontsize=8.5, linespacing=1.2, va="top")
        ax.set_xlim(-0.55, 3.65)
        ax.set_title(titles[arm], fontsize=10, color=INK, loc="left", pad=16)
        ax.text(-0.5, 1.13, "mean", fontsize=8, color=INK2, va="bottom")
    axes[0].set_ylabel("M3 success rate", fontsize=9, color=INK2)
    fig.text(0.075, 0.965, "M3 at epoch 6,000 · one component removed at a time · physics v1", fontsize=10.5,
             color=INK, weight="semibold", va="top")
    legend(fig, 0.02, GRAY)
    save(fig, "v1_ablation_m3_e6000", preview)


# ---------------------------------------------------------------------------------------------------- table views
def num(v, d=4):
    return f"{v:.{d}f}"


def pp(v):
    return f"{v:+.1f}".replace("-", "−")


def table_success(nums) -> str:
    cols = FAMS + ["balanced"]
    rows = ['<table class="table is-narrow result-table">',
            "<caption>Success rate of each training seed's final checkpoint (epoch 10,000) on its first episodes, "
            "physics v1. Mean ± SD over the 8 seeds and the 95 % t-interval of the mean (df 7, clipped to [0, 1]). "
            "Balanced success is the mean over the four families.</caption>",
            '<thead><tr><th scope="col">Seed</th>' + "".join(f'<th scope="col">{c}</th>' for c in cols)
            + "</tr></thead>", "<tbody>"]
    for arm, title in (("task-only", "Task reward only"), ("N1", "With posture regularisers (N1)")):
        d = nums["success_e10000"][arm]
        rows.append(f'<tr class="group-row"><th scope="rowgroup" colspan="{len(cols) + 1}">{title}</th></tr>')
        for i, s in enumerate(d["seeds"]):
            rows.append(f'<tr><th scope="row">{s}</th>'
                        + "".join(f"<td>{num(d['families'][c]['per_seed'][i])}</td>" for c in cols) + "</tr>")
        rows.append('<tr class="summary-row"><th scope="row">mean ± SD</th>'
                    + "".join(f"<td>{num(d['families'][c]['mean'])} ± {num(d['families'][c]['sd'])}</td>"
                              for c in cols) + "</tr>")
        rows.append('<tr class="summary-row"><th scope="row">95 % interval</th>'
                    + "".join(f"<td>{num(d['families'][c]['lo'])} to {num(d['families'][c]['hi'])}</td>"
                              for c in cols) + "</tr>")
    rows += ["</tbody>", "</table>"]
    return "\n".join(rows)


def table_ablation(nums) -> str:
    rows = ['<table class="table is-narrow result-table">',
            "<caption>M3 success at epoch 6,000 (196,608,000 frames per run), physics v1. Per-seed rates in seed "
            "order (full recipe: seeds 1, 2, 4, 5, 6, 7, 8, 9; each removal: seeds 1, 2, 4). The change is the "
            "removal's mean minus the full recipe's, with its 95 % interval; the call follows the pre-registered "
            "rule (necessary: a drop of at least 20 pp and non-overlapping 95 % intervals of the two means).</caption>",
            '<thead><tr><th scope="col">Run</th><th scope="col">M3 per seed</th><th scope="col">mean</th>'
            '<th scope="col">95 % interval</th><th scope="col">change</th><th scope="col">call</th></tr></thead>',
            "<tbody>"]
    names = {"hist1": "history H = 1", "nounlock": "no unlock bonus", "nocurr": "no curriculum"}
    for arm, title in (("task-only", "Task reward only"), ("N1", "With posture regularisers (N1)")):
        g = nums["gd_m3_e6000"][arm]
        rows.append(f'<tr class="group-row"><th scope="rowgroup" colspan="6">{title}</th></tr>')
        p = g["primary"]
        rows.append('<tr><th scope="row">full recipe</th><td class="seeds">' + ", ".join(num(v, 3) for v in p["per_seed"])
                    + f"</td><td>{num(p['mean'], 3)}</td><td>{num(p['lo'], 3)} to {num(p['hi'], 3)}</td>"
                      "<td></td><td></td></tr>")
        for ab, _ in ABL:
            a = g["ablations"][ab]
            ch = -a["gap_pp"]
            lo, hi = sorted((-a["diff_pp"][1], -a["diff_pp"][0]))
            if a["necessary"]:
                call = "necessary"
            elif ab == "nounlock" and arm == "task-only":
                call = "not necessary (intervals overlap)"
            elif a["difference_resolved"]:
                call = "raised M3" if ch > 0 else "lowered M3, not necessary"
            else:
                call = "no difference resolved"
            rows.append(f'<tr><th scope="row">{names[ab]}</th><td class="seeds">'
                        + ", ".join(num(v, 3) for v in a["per_seed"])
                        + f"</td><td>{num(a['mean'], 3)}</td><td>{num(a['lo'], 3)} to {num(a['hi'], 3)}</td>"
                        + f'<td class="wrap">{pp(ch)} pp<br>({pp(lo)} to {pp(hi)})</td><td class="wrap">{call}</td></tr>')
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
    ap.add_argument("--html", type=Path, default=None, help="write the two table views into this page")
    a = ap.parse_args()
    nums = extract(a.data)
    (ROOT / "tools" / "data").mkdir(exist_ok=True)
    (ROOT / "tools" / "data" / "v1_numbers.json").write_text(json.dumps(nums, indent=1), encoding="utf-8",
                                                              newline="\n")
    chart_success(nums, a.preview_dir)
    chart_ablation(nums, a.preview_dir)
    if a.html:
        patch_html(a.html, "v1 success table", table_success(nums))
        patch_html(a.html, "v1 ablation table", table_ablation(nums))
    print("ok")


if __name__ == "__main__":
    main()
