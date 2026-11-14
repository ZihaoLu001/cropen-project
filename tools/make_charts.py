"""Make the two result charts of the CR-Open project page as static SVG (no data beyond the cited run).

Data: the formal first-episode evaluations of the one finished seed (cropen_ppo_gate30_s3, seed 3; 2048 first
episodes per checkpoint, full gate thresholds, deterministic policy), copied from its eval.jsonl. Per-family counts at
epoch 8000 come from dexrl-cropen/cropen/nf_recover.py (M1 and M6 use its 512 fallback; their rate is 1.0).
Palette: dataviz reference categorical slots 1 and 2 (validated: CVD dE 24.7, normal 33.6, contrast >= 3:1).
Usage: python -I tools/make_charts.py <seed3 eval.jsonl> <out dir>
"""
import json
import math
import sys
from pathlib import Path

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"
FONT = "'Noto Sans', 'Google Sans', system-ui, sans-serif"


def ac95(k, n):
    """Agresti-Coull 95 % interval"""
    z = 1.96
    nt = n + z * z
    p = (k + z * z / 2) / nt
    h = z * math.sqrt(p * (1 - p) / nt)
    return max(0.0, p - h), min(1.0, p + h)


def svg_open(w, h, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d" '
            f'font-family="{FONT}" style="background:{SURF}">',
            f'<title id="t">{title}</title><desc id="d">{desc}</desc>',
            f'<rect width="{w}" height="{h}" fill="{SURF}"/>']


def curve(rows, out):
    W, H, L, R, T, B = 760, 420, 64, 120, 76, 56
    xs = [r["epoch"] for r in rows]
    m3 = [r["succ_M3"] for r in rows]
    bal = [(r["succ_M1"] + r["succ_M2"] + r["succ_M3"] + r["succ_M6"]) / 4 for r in rows]
    X = lambda e: L + (W - L - R) * e / 8000
    Y = lambda v: T + (H - T - B) * (1 - v)
    s = svg_open(W, H, "Success rate over training, one seed",
                 "Seed 3, formal first-episode evaluation every 500 epochs (2048 episodes each, full gate thresholds). "
                 "M3 rises from 0 to 0.84 at epoch 4000, dips to 0.41 at 7000 and ends at 0.61 at 8000; "
                 "the family-balanced rate ends at 0.90.")
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L - 10}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="13" fill="{INK2}">{v:.2f}</text>')
    for e in (0, 2000, 4000, 6000, 8000):
        s.append(f'<text x="{X(e):.1f}" y="{H - B + 22}" text-anchor="middle" font-size="13" fill="{INK2}">{e}</text>')
    s.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 12}" text-anchor="middle" font-size="13" fill="{INK2}">'
             f'training epoch (32,768 frames each)</text>')
    s.append(f'<text x="{L}" y="{T - 16}" font-size="13" fill="{INK2}">success rate (first episodes, full thresholds)</text>')
    for ys, col, name in ((bal, S2, "Family-balanced"), (m3, S1, "M3 squeeze-and-turn")):
        pts = " ".join(f"{X(e):.1f},{Y(v):.1f}" for e, v in zip(xs, ys))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        for e, v in zip(xs, ys):
            s.append(f'<circle cx="{X(e):.1f}" cy="{Y(v):.1f}" r="4" fill="{col}" stroke="{SURF}" stroke-width="2">'
                     f'<title>{name}, epoch {e}: {v:.3f}</title></circle>')
        short = "M3" if name.startswith("M3") else "Balanced"
        s.append(f'<text x="{X(xs[-1]) + 10:.1f}" y="{Y(ys[-1]) + 4:.1f}" font-size="13" fill="{INK}">'
                 f'{short} {ys[-1]:.2f}</text>')
    # legend (two series)
    for i, (col, name) in enumerate(((S1, "M3 squeeze-and-turn"), (S2, "Family-balanced (M1, M2, M3, M6)"))):
        lx, ly = L + i * 230, T - 44
        s.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{ly}" y2="{ly}" stroke="{col}" stroke-width="2"/>')
        s.append(f'<circle cx="{lx + 9}" cy="{ly}" r="4" fill="{col}"/>')
        s.append(f'<text x="{lx + 24}" y="{ly + 4}" font-size="12" fill="{INK2}">{name}</text>')
    s.append("</svg>")
    (out / "curve_seed3.svg").write_text("\n".join(s), encoding="utf-8")


def bars(counts, out):
    W, H, L, R, T, B = 760, 340, 64, 24, 40, 64
    names = [("M1", "thread"), ("M2", "push-and-turn"), ("M3", "squeeze-and-turn"), ("M6", "stuck seal")]
    bw, gap = 110, 2
    step = (W - L - R) / 4
    Y = lambda v: T + (H - T - B) * (1 - v)
    s = svg_open(W, H, "Per-family success at epoch 8000, one seed",
                 "Seed 3 at epoch 8000, 2048 first episodes: M1 512 of 512, M2 497 of 498, M3 317 of 516, "
                 "M6 512 of 512, with 95 % Agresti-Coull intervals.")
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L - 10}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="13" fill="{INK2}">{v:.2f}</text>')
    s.append(f'<text x="{L}" y="{T - 16}" font-size="13" fill="{INK2}">success rate at epoch 8000 (95 % interval)</text>')
    for i, (f, label) in enumerate(names):
        k, n = counts[f]
        p = k / n
        lo, hi = ac95(k, n)
        cx = L + step * (i + 0.5)
        x0, y0, h = cx - bw / 2, Y(p), Y(0) - Y(p)
        # 4 px rounded data end, square at the baseline
        s.append(f'<path d="M{x0:.1f},{Y(0):.1f} V{y0 + 4:.1f} Q{x0:.1f},{y0:.1f} {x0 + 4:.1f},{y0:.1f} '
                 f'H{x0 + bw - 4:.1f} Q{x0 + bw:.1f},{y0:.1f} {x0 + bw:.1f},{y0 + 4:.1f} V{Y(0):.1f} Z" fill="{S1}">'
                 f'<title>{f} {label}: {k}/{n} = {p:.3f} (95 % {lo:.3f}-{hi:.3f})</title></path>')
        s.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{Y(hi):.1f}" y2="{Y(lo):.1f}" stroke="{INK}" stroke-width="1.5"/>')
        for yy in (Y(hi), Y(lo)):
            s.append(f'<line x1="{cx - 6:.1f}" x2="{cx + 6:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{INK}" stroke-width="1.5"/>')
        if p < 0.9:                     # label beside the whisker, above the bar
            s.append(f'<text x="{cx + 12:.1f}" y="{y0 - 8:.1f}" font-size="14" font-weight="600" fill="{INK}">{p:.2f}</text>')
        else:
            s.append(f'<text x="{cx:.1f}" y="{y0 + 24:.1f}" text-anchor="middle" font-size="14" font-weight="600" '
                     f'fill="{SURF}">{p:.2f}</text>')
        s.append(f'<text x="{cx:.1f}" y="{H - B + 20}" text-anchor="middle" font-size="13" fill="{INK}">{f}</text>')
        s.append(f'<text x="{cx:.1f}" y="{H - B + 38}" text-anchor="middle" font-size="12" fill="{INK2}">{label}</text>')
    s.append("</svg>")
    (out / "bars_seed3_e8000.svg").write_text("\n".join(s), encoding="utf-8")


def main():
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["epoch"])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    curve(rows, out)
    bars({"M1": (512, 512), "M2": (497, 498), "M3": (317, 516), "M6": (512, 512)}, out)
    print("wrote", sorted(p.name for p in out.glob("*.svg")))


if __name__ == "__main__":
    main()
