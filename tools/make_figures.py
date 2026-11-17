"""Method overview and mechanism figures of the CR-Open project page (static/images/method.svg, mechanisms.svg).

Drawn with matplotlib: serif (STIX) text and math, converted to paths so the SVG looks the same in every browser;
the render thumbnails in tools/fig_assets/ are embedded as JPEG (matplotlib embeds PNG; this script swaps in a
JPEG of the same pixels to keep the files small). Every label is taken from the code of the training checkout
(dexrl-cropen-v2/cropen: env_cropen.py, vml.py, closure_urdf.py, train_ppo.py) and the recipe's command line
(README P1, recipe N1):
  actor input  x_t = [q, qdot, q*, f (4 fingertips x 3), a_{t-1}] (76 numbers, the order of _frame()), stacked over
               --obs-history 8 steps;
  critic input s_t = [x_t, q_c, qdot_c, family one-hot, mechanism state (6), mechanism parameters (5)] (101);
  rl_games MLPs 512-256-128 (ELU) for the actor and the separate central-value critic; the actor is Gaussian with a
               learned, state-independent log-std (fixed_sigma=True, initial sigma 1) and acts with its mean at test;
  PhysX at 120 Hz, decimation 4 (30 Hz control), action scale 0.08 rad; the joint PD drives run every substep;
  VML gate: engaged while z_p > th_p (M2) or z_s > th_s (M3; z_s = mean of the two tab slides), with 0.8 mm
               hysteresis, then for a further unlock window of 0-0.5 rad of turning (redrawn every episode);
  reward: thread progress, release, lift, success, a one-off 2.0 when a gated cap first engages (--r-unlock 2.0),
  jam / action / force costs; N1 adds --w-posture -0.01, --w-rate -0.001, a flexion floor and a relaxed reset pose;
  gate curriculum: --gate-init 0.25 --gate-step 0.05 --gate-up 0.3 (over >= 200 episodes); evaluation at full gates.

Every figure is checked before it is written: no text may cross the border of the box that holds it.

Usage: python tools/make_figures.py [--preview-dir DIR]   (writes static/images/*.svg; PNG previews to DIR)
"""
from __future__ import annotations

import argparse
import base64
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tools" / "fig_assets"
OUT = ROOT / "static" / "images"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "svg.fonttype": "path", "svg.hashsalt": "cropen",
    "axes.linewidth": 0, "savefig.facecolor": "white", "figure.dpi": 100,
})

INK, INK2, MUTED = "#1d1d1b", "#52514e", "#898781"
ARROW = "#2b2b29"
# group colours (pale fill, darker stroke)
SIM = ("#f6f6f4", "#9a978f")
VML = ("#fff5e6", "#cf8a1f")           # amber: the simulator's internals (also its arrows)
OBS = ("#eef4fc", "#2a78d6")
POL = ("#eef8f0", "#2f8f46")
TRN = ("#f3f5f8", "#5b6b7f")           # slate: training only
# family inputs, as in the clip overlay (validated: dataviz validate_palette.js, all pairs, light)
PUSH, SQUEEZE, TWIST, JAM = "#00a38d", "#eb6834", "#8b55e8", "#d03b3b"
FORCE = "#c4407e"                      # fingertip-force segment of the observation (not the orange tab paint)
DASH = (0, (4, 3))


# ---------------------------------------------------------------------------------------------------- primitives
class Canvas:
    def __init__(self, w, h, scale=0.01):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w * scale, h * scale), dpi=100)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")
        self.regions = []          # boxes whose texts must stay inside (checked before saving)
        self.free = []             # texts allowed to cross a border (labels on arrows, with a white backing)

    def box(self, x0, y0, x1, y1, fc="white", ec=INK2, lw=1.0, r=8, ls="-", z=1, alpha=1.0, region=True):
        p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                           lw=lw, ls=ls, zorder=z, alpha=alpha, joinstyle="round")
        self.ax.add_patch(p)
        if region:
            self.regions.append((x0, y0, x1, y1))
        return p

    def text(self, x, y, s, size=10, ha="center", va="center", color=INK, z=6, free=False, **kw):
        t = self.ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, zorder=z, **kw)
        if free:
            self.free.append(t)
        return t

    def line(self, pts, color=ARROW, lw=1.0, ls="-", z=4):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw, ls=DASH if ls == "--" else ls, zorder=z, solid_capstyle="butt",
                     dash_capstyle="butt", dash_joinstyle="miter", solid_joinstyle="miter")

    def head(self, p_from, p_to, color, size, z):
        """filled triangular head with its tip at p_to, pointing from p_from"""
        (x0, y0), (x1, y1) = p_from, p_to
        d = np.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        L, hw = size, 0.42 * size
        bx, by = x1 - ux * L, y1 - uy * L
        tri = [(x1, y1), (bx - uy * hw, by + ux * hw), (bx + uy * hw, by - ux * hw)]
        self.ax.add_patch(Polygon(tri, closed=True, fc=color, ec=color, lw=0.4, zorder=z + 0.1, joinstyle="miter"))
        return bx, by

    def arrow(self, pts, color=ARROW, lw=1.0, ls="-", head=7.5, z=5, both=False):
        """orthogonal polyline with a solid filled head at the end (and at the start if both); the shaft may be
        dashed, the head never is"""
        pts = [tuple(map(float, p)) for p in pts]
        bx, by = self.head(pts[-2], pts[-1], color, head, z)
        shaft = pts[:-1] + [(bx, by)]
        if both:
            sx, sy = self.head(pts[1], pts[0], color, head, z)
            shaft = [(sx, sy)] + shaft[1:]
        self.line(shaft, color, lw, ls, z)

    def ellipse_arrow(self, x, y, w, h, deg0, deg1, color=ARROW, lw=1.0, head=7.5, z=7, ls="-"):
        """arc of the ellipse centred (x, y) from deg0 to deg1 (data angles; y grows downwards, so 270 is the
        top), drawn as a polyline with a head at deg1"""
        t = np.radians(np.linspace(deg0, deg1, 80))
        pts = list(zip(x + w / 2 * np.cos(t), y + h / 2 * np.sin(t)))
        # the head follows the chord over its own length, so it sits on the arc
        L = head
        k = len(pts) - 1
        while k > 1 and np.hypot(pts[-1][0] - pts[k][0], pts[-1][1] - pts[k][1]) < L:
            k -= 1
        bx, by = self.head(pts[k], pts[-1], color, head, z)
        self.line(pts[:k + 1] + [(bx, by)], color, lw, ls, z)

    def image(self, path, x0, y0, w, r=6, z=3, h=None, frame=True):
        im = np.asarray(Image.open(path).convert("RGB"))
        h = h if h is not None else w * im.shape[0] / im.shape[1]
        art = self.ax.imshow(im, extent=(x0, x0 + w, y0 + h, y0), interpolation="none", zorder=z, aspect="auto")
        clip = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec="none",
                              transform=self.ax.transData)
        art.set_clip_path(clip)
        if frame:
            self.box(x0, y0, x0 + w, y0 + h, fc="none", ec="#d6d4cd", lw=0.8, r=r, z=z + 0.5, region=False)
        return h

    def trapezoid(self, x0, x1, yc, h_in, h_out, fc, ec=None, z=4):
        pts = [(x0, yc - h_in / 2), (x1, yc - h_out / 2), (x1, yc + h_out / 2), (x0, yc + h_in / 2)]
        self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec or fc, lw=1.0, zorder=z, joinstyle="round"))

    def eye_slash(self, x, y, s=1.0, color=INK2, z=7, lw=1.0):
        self.ax.add_patch(Ellipse((x, y), 20 * s, 11 * s, fc="white", ec=color, lw=lw, zorder=z))
        self.ax.add_patch(Circle((x, y), 3.0 * s, fc=color, ec=color, lw=0, zorder=z + 0.1))
        self.line([(x - 9 * s, y + 7 * s), (x + 9 * s, y - 7 * s)], color=color, lw=lw + 0.4, z=z + 0.2)

    def feature_bar(self, x0, y0, w, h, parts, colors, z=5, gap=1.2):
        """horizontal bar split into parts (proportional widths); returns the x range of every part"""
        tot = float(sum(parts))
        x, spans = x0, []
        for p, col in zip(parts, colors):
            pw = w * p / tot
            self.box(x + gap / 2, y0, x + pw - gap / 2, y0 + h, fc=col, ec="none", lw=0, r=1.5, z=z, region=False)
            spans.append((x, x + pw))
            x += pw
        return spans

    def width(self, s, size):
        """rendered width of a text (data units = pixels)"""
        t = self.ax.text(0, 0, s, fontsize=size)
        bb = t.get_window_extent(self.fig.canvas.get_renderer())
        t.remove()
        return bb.width

    def check(self, name):
        """every text must lie inside the smallest registered box around its centre (3 px margin)"""
        r = self.fig.canvas.get_renderer()
        bad = []
        for t in self.ax.texts:
            if t in self.free or not t.get_text().strip():
                continue
            bb = t.get_window_extent(r)
            x0, x1, y0, y1 = bb.x0, bb.x1, self.h - bb.y1, self.h - bb.y0
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if x0 < 2 or x1 > self.w - 2 or y0 < 1 or y1 > self.h - 1:
                bad.append(f"{t.get_text()!r} [{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}] outside the canvas")
                continue
            inside = [g for g in self.regions if g[0] <= cx <= g[2] and g[1] <= cy <= g[3]]
            if not inside:
                continue
            g = min(inside, key=lambda g: (g[2] - g[0]) * (g[3] - g[1]))
            if x0 < g[0] + 3 or x1 > g[2] - 3 or y0 < g[1] + 1 or y1 > g[3] - 1:
                bad.append(f"{t.get_text()!r} [{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}] in {tuple(round(v) for v in g)}")
        if bad:
            raise SystemExit(f"{name}: text crosses a box border:\n  " + "\n  ".join(bad))

    def save(self, stem, preview_dir):
        self.check(stem)
        svg = io.BytesIO()
        self.fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": None})
        text = jpeg_images(svg.getvalue().decode("utf-8"))
        (OUT / f"{stem}.svg").write_text(text, encoding="utf-8", newline="\n")
        if preview_dir:
            self.fig.savefig(Path(preview_dir) / f"{stem}.png", dpi=200)
            self.fig.savefig(Path(preview_dir) / f"{stem}_page.png", dpi=int(100 * 940 / self.w))
        plt.close(self.fig)
        return len(text)


def jpeg_images(svg: str, quality=86) -> str:
    """replace matplotlib's embedded PNGs (photographic renders) by JPEGs of the same pixels"""
    def swap(m):
        raw = base64.b64decode(m.group(1))
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return re.sub(r"data:image/png;base64,([A-Za-z0-9+/=\s]+)", lambda m: swap(m), svg)


# ---------------------------------------------------------------------------------------------------- method
def method_figure(preview_dir=None):
    W, H = 1200, 724
    c = Canvas(W, H)
    T = 56  # top of the group boxes (the action lane runs above them)
    seg_parts = [16, 16, 16, 12, 16]                                         # the order of _frame()
    seg_cols = ["#2a78d6", "#5598e7", "#86b6ef", FORCE, "#b7d3f6"]           # q, qdot, q*, f, a_{t-1}

    # ---------------- group boxes (titles at the bottom centre)
    A = (12, T, 412, 676)
    B = (432, T, 792, 352)
    C = (812, 182, 1188, 352)
    D = (432, 370, 1188, 676)
    c.box(*A, fc=SIM[0], ec=SIM[1], lw=1.5, r=12, z=0)
    c.box(*B, fc=OBS[0], ec=OBS[1], lw=1.5, r=12, z=0)
    c.box(*C, fc=POL[0], ec=POL[1], lw=1.5, r=12, z=0)
    c.box(*D, fc=TRN[0], ec=TRN[1], lw=1.5, r=12, z=0, ls=(0, (5, 3)))
    c.text((A[0] + A[2]) / 2, A[3] - 14, "Simulation (IsaacLab, PhysX)", size=13)
    c.text((B[0] + B[2]) / 2, B[3] - 14, "Deployable observation", size=13)
    c.text((C[0] + C[2]) / 2 + 40, C[3] - 14, "Policy", size=13)
    c.text((D[0] + D[2]) / 2, D[3] - 14, "Training only: PPO with an asymmetric critic", size=13)

    # ---------------- A: close-up of the closure and its joints; render of the hand over the bottle
    cx, cy, cw = 30, T + 54, 164
    ch = c.image(ASSETS / "cap_closeup.jpg", cx, cy, cw)
    mid = cx + cw / 2
    # the far rim moves to the left when the shell turns the opening way (+z, counter-clockwise from above)
    c.ellipse_arrow(mid, cy - 2, 124, 22, 340, 200, color=ARROW, lw=1.2, head=7)
    c.arrow([(mid, cy - 26), (mid, cy + 4)], color=ARROW, lw=1.2, head=7)
    c.text(mid + 70, cy - 13, r"$\theta$", size=12)
    c.text(mid + 6, cy - 26, r"$z_p$", size=12, ha="left")
    ty_tab = cy + ch * 0.31
    c.arrow([(cx - 14, ty_tab), (cx + 4, ty_tab)], color=ARROW, lw=1.2, head=7)
    c.text(cx - 14, ty_tab - 12, r"$z_s$", size=12, ha="left")
    c.text(mid, cy + ch + 15, r"closure joints  $q_c$", size=11.5)
    c.text(mid, cy + ch + 32, r"shell $\theta$, press $z_p$, lift $h$,", size=10.5, color=INK2)
    c.text(mid, cy + ch + 48, r"tabs $z_s$ (mean of the two)", size=10.5, color=INK2)
    hx, hy, hw = 222, T + 14, 172
    hh = c.image(ASSETS / "hand_over_bottle.jpg", hx, hy, hw)
    c.text(hx + hw / 2, hy + hh + 14, "Allegro hand, 16 joints, palm down", size=10, color=INK2)

    # ---------------- A: the virtual mechanism layer (nested); its arrows are simulator internals (amber)
    V = (26, 336, 398, 642)
    c.box(*V, fc=VML[0], ec=VML[1], lw=1.4, r=10, z=1)
    c.text((V[0] + V[2]) / 2, V[3] - 14, "Virtual mechanism layer, every physics substep", size=11.5)
    xr, xw = 84, 112
    y0, y1 = cy + ch + 60, V[1]
    c.arrow([(xr, y0), (xr, y1)], color=VML[1], lw=1.4)
    c.arrow([(xw, y1), (xw, y0)], color=VML[1], lw=1.4)
    c.text(xr - 8, (y0 + y1) / 2, r"read $q_c$", size=11, ha="right")
    c.text(xw + 8, (y0 + y1) / 2 - 8, r"write drive targets $\hat q_c$", size=11, ha="left")
    c.text(xw + 8, (y0 + y1) / 2 + 9, r"and effort limits $\bar\tau_c$", size=11, ha="left")
    c.eye_slash(V[0] + 22, V[1] + 22, s=0.9, color=INK)
    c.text(V[0] + 38, V[1] + 21, r"hidden family $k\in\{\mathrm{M1,\,M2,\,M3,\,M6}\}$", size=11.5, ha="left")
    c.text(V[0] + 38, V[1] + 41, r"parameters $\xi=(\theta_p,\theta_s,n,\tau_b,\tau_t)$, new each episode",
           size=10.5, ha="left", color=INK2)
    rules = [
        ("gate", [r"engaged while $z_p>\theta_p$ (M2) or $z_s>\theta_s$ (M3),",
                  r"with hysteresis, then for a further 0–0.5 rad"]),
        ("jam", [r"$z_p>\theta_p$ on M1, M3 or M6 holds the shell"]),
        ("clutch", [r"M6: held until $|\tau|\geq\tau_b$, then slips free"]),
        ("thread", [r"friction $\tau_t$; after $n$ engaged turns the cap is free"]),
    ]
    y = V[1] + 80
    for name, lines in rules:
        c.box(V[0] + 12, y - 11, V[0] + 68, y + 11, fc="white", ec=VML[1], lw=0.9, r=5, z=2)
        c.text(V[0] + 40, y, name, size=11, style="italic")
        if len(lines) == 1:
            c.text(V[0] + 78, y, lines[0], size=10.5, ha="left")
        else:
            c.text(V[0] + 78, y - 8, lines[0], size=10.5, ha="left")
            c.text(V[0] + 78, y + 9, lines[1], size=10.5, ha="left")
        y += 50 if len(lines) > 1 else 40
    c.text(V[0] + 12, y - 6, "at run time only drive targets and effort limits change", size=10, ha="left",
           color=INK2)

    # ---------------- B: deployable observation
    tx = B[0] + 12                                   # token list (left)
    tokens = [(r"$q_t\in\mathbb{R}^{16}$", "joint positions", 0),
              (r"$\dot q_t\in\mathbb{R}^{16}$", "joint velocities", 1),
              (r"$q^{*}_t\in\mathbb{R}^{16}$", "joint targets", 2),
              (r"$a_{t-1}\in\mathbb{R}^{16}$", "last action", 4)]
    math_w = max(c.width(m, 11) for m, _, _ in tokens)
    desc_x = tx + 20 + math_w + 8
    ty = T + 24
    for i, (m, s, j) in enumerate(tokens):
        yy = ty + i * 25
        c.feature_bar(tx, yy - 4.5, 14, 9, [1], [seg_cols[j]])
        c.text(tx + 20, yy - 1, m, size=11, ha="left")
        c.text(desc_x, yy, s, size=10.5, ha="left", color=INK2)
    list_right = desc_x + max(c.width(s, 10.5) for _, s, _ in tokens)
    # one step of 76 numbers across the box; the fingertip render sits over the force segment
    bx0, bw, by = B[0] + 12, (B[2] - 12) - (B[0] + 12), T + 158
    spans = c.feature_bar(bx0, by, bw, 13, seg_parts, seg_cols)
    f_mid = sum(spans[3]) / 2
    fw = 104
    fx, fy = f_mid - fw / 2, T + 12
    assert fx > list_right + 6, (fx, list_right)
    fh = c.image(ASSETS / "fingertips_on_cap.jpg", fx, fy, fw)
    c.feature_bar(f_mid - 52, fy + fh + 8.5, 12, 9, [1], [FORCE])
    c.text(f_mid - 34, fy + fh + 12, "fingertip forces", size=11, ha="left")
    c.text(f_mid, fy + fh + 30, r"$f_t\in\mathbb{R}^{4\times 3}$", size=11)
    c.arrow([(f_mid, fy + fh + 41), (f_mid, by - 1)], lw=1.1, head=6.5)
    # the list feeds q, qdot, q* and a: a bracket over the strip, open where the forces come in
    yb = by - 7
    c.line([(spans[0][0] + 1, yb + 4), (spans[0][0] + 1, yb), (spans[2][1] + 2, yb)], lw=1.0)
    c.line([(spans[4][0] - 2, yb), (spans[4][1] - 1, yb), (spans[4][1] - 1, yb + 4)], lw=1.0)
    c.line([(spans[2][1] + 2, yb), (f_mid - 9, yb)], lw=1.0)
    c.line([(f_mid + 9, yb), (spans[4][0] - 2, yb)], lw=1.0)
    lx = (spans[1][0] + spans[1][1]) / 2
    c.arrow([(lx, ty + 3 * 25 + 8), (lx, yb)], lw=1.1, head=6.5)
    c.text(bx0, by + 28, r"one step  $x_t\in\mathbb{R}^{76}$", size=11.5, ha="left")
    # the history: 8 stacked frames
    n, dx, dy = 8, 4.5, 3.0
    sx0, sy0, swid, shgt = bx0, by + 46, 196, 20
    for i in range(n):
        k = n - 1 - i                         # back cards first
        x, y = sx0 + k * dx, sy0 + i * dy
        c.box(x, y, x + swid, y + shgt, fc="white", ec="#9ec5f4", lw=0.8, r=4, z=3 + i * 0.1, region=False)
    fy0 = sy0 + (n - 1) * dy                  # the front card (x_t) is at sx0
    c.feature_bar(sx0 + 7, fy0 + 6, swid - 14, 8, seg_parts, seg_cols, z=4.5)
    xs_ = bx0 + c.width(r"one step  $x_t\in\mathbb{R}^{76}$", 11.5) + 30
    c.arrow([(xs_, by + 14), (xs_, sy0 - 1)], lw=1.0, head=6)
    c.text(xs_ + 8, (by + 14 + sy0) / 2 + 1, "stack 8 steps", size=10.5, ha="left", color=INK2)
    ay = fy0 + shgt / 2
    stack_right = sx0 + swid + (n - 1) * dx
    c.text((stack_right + B[2]) / 2 - 2, ay - 14, r"$o_t=x_{t-7:t}$", size=12)
    c.eye_slash(B[0] + 28, B[3] - 39, s=0.9, color=INK, lw=1.2)
    c.text(B[0] + 44, B[3] - 39, r"no family, no closure state:  $k\notin o_t$", size=11.5, ha="left")

    # ---------------- C: actor
    c.arrow([(stack_right + 4, ay), (C[0] + 20, ay)], lw=1.2)
    tx0, tx1 = C[0] + 22, C[0] + 142
    c.trapezoid(tx0, tx1, ay, 112, 56, fc=POL[1])
    c.text((tx0 + tx1) / 2 - 4, ay - 9, "MLP", size=12, color="white")
    c.text((tx0 + tx1) / 2 - 4, ay + 9, "512-256-128", size=10.5, color="white")
    c.text(C[0] + 16, ay - 86, r"actor  $\pi_\psi(a_t\,|\,o_t)$", size=12.5, ha="left")
    c.text(C[0] + 16, ay - 68, r"Gaussian, learned state-independent $\sigma$; mean at test", size=10.5, ha="left",
           color=INK2)
    c.arrow([(tx1, ay), (tx1 + 42, ay)], lw=1.2)
    c.box(tx1 + 44, ay - 17, tx1 + 176, ay + 17, fc="white", ec=POL[1], lw=1.0, r=8, z=3)
    c.text(tx1 + 110, ay, r"$a_t\in[-1,1]^{16}$", size=12)
    lane = 30
    c.arrow([(tx1 + 176, ay), (C[2] - 14, ay), (C[2] - 14, lane), (hx + hw / 2, lane), (hx + hw / 2, hy - 1)],
            lw=1.3)
    c.text((hx + hw / 2 + C[2]) / 2, lane - 14,
           r"new joint targets  $q^{*}_{t+1}=\mathrm{clip}(q^{*}_t+0.08\,a_t)$  at 30 Hz;  PD drives at 120 Hz",
           size=12)
    # hand -> observation
    c.arrow([(hx + hw, ty + 12), (tx - 3, ty + 12)], lw=1.2)
    c.text((hx + hw + tx) / 2 - 2, ty, r"$q_t,\dot q_t,f_t$", size=11, free=True,
           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    # ---------------- D: training-only signals (every arrow dashed)
    P = (D[0] + 26, D[1] + 22, D[0] + 214, D[1] + 206)
    c.box(*P, fc="white", ec=TRN[1], lw=1.0, r=8, z=2)
    c.text((P[0] + P[2]) / 2, P[1] + 17, r"critic input  $s_t\in\mathbb{R}^{101}$", size=11.5)
    priv = [(r"$x_t$", "current step (76)"), (r"$q_c,\dot q_c$", "closure joints (5+5)"),
            (r"$k$", "family one-hot (4)"), (r"$m_t$", "mechanism state (6)"), (r"$\xi$", "parameters (5)")]
    rows = [P[1] + 44 + i * 25 for i in range(len(priv))]
    c.box(P[0] + 5, rows[1] - 12, P[2] - 5, rows[-1] + 12, fc=VML[0], ec="none", lw=0, r=4, z=2.2, region=False)
    for yy, (m, s) in zip(rows, priv):
        c.text(P[0] + 12, yy, m, size=11.5, ha="left")
        c.text(P[0] + 62, yy, s, size=10.5, ha="left", color=INK2)
    c.text(P[0] + 12, P[3] - 13, "shaded: privileged (sim only)", size=10, ha="left", color="#8a5a10")
    # x_t (the front card of the stack) to the critic, down the left gutter of B and D
    gx = (A[2] + B[0]) / 2
    c.arrow([(sx0, ay), (gx, ay), (gx, rows[0]), (P[0], rows[0])], lw=1.1, ls="--", color=TRN[1], head=6.5)
    # q_c, k, m_t, xi from the mechanism layer
    c.arrow([(V[2], rows[2]), (P[0], rows[2])], lw=1.1, ls="--", color=TRN[1], head=6.5)
    cyc = (rows[0] + rows[-1]) / 2
    c.arrow([(P[2], cyc), (P[2] + 18, cyc)], lw=1.1, color=TRN[1], head=6.5, ls="--")
    k0, k1 = P[2] + 20, P[2] + 112
    c.trapezoid(k0, k1, cyc, 98, 48, fc=TRN[1])
    c.text((k0 + k1) / 2 - 3, cyc - 8, "MLP", size=11.5, color="white")
    c.text((k0 + k1) / 2 - 3, cyc + 9, "512-256-128", size=10, color="white")
    c.text((k0 + k1) / 2, cyc - 70, r"critic  $V_\phi(s_t)$", size=12.5)
    Q = (k1 + 28, cyc - 21, k1 + 92, cyc + 21)
    c.arrow([(k1, cyc), (Q[0], cyc)], lw=1.1, color=TRN[1], head=6.5, ls="--")
    c.box(*Q, fc="white", ec=TRN[1], lw=1.0, r=8, z=3)
    c.text((Q[0] + Q[2]) / 2, cyc, "PPO", size=12.5)
    qx = (Q[0] + Q[2]) / 2
    act_x = (tx0 + tx1) / 2
    act_bottom = ay + (112 + 56) / 4 + 2
    gap_y = (B[3] + D[1]) / 2
    c.arrow([(qx, Q[1]), (qx, gap_y), (act_x, gap_y), (act_x, act_bottom)], lw=1.2, ls="--", color=TRN[1])
    c.text(qx - 7, D[1] + 17, r"update $\psi$", size=11, ha="right")
    crit_bottom = cyc + (98 + 48) / 4 + 1
    yr = Q[3] + 30
    c.arrow([(qx, Q[3]), (qx, yr), ((k0 + k1) / 2, yr), ((k0 + k1) / 2, crit_bottom)], lw=1.1, ls="--",
            color=TRN[1], head=6.5)
    c.text((qx + (k0 + k1) / 2) / 2, yr + 13, r"update $\phi$", size=11)
    # reward (right column)
    R = (Q[2] + 30, D[1] + 16, D[2] - 14, D[1] + 152)
    c.box(*R, fc="white", ec=TRN[1], lw=1.0, r=8, z=2)
    lx = R[0] + 10
    c.text(lx, R[1] + 18, r"reward  $r_t=r_{\mathrm{thread}}+r_{\mathrm{release}}+r_{\mathrm{lift}}+r_{\mathrm{success}}$",
           size=11, ha="left")
    c.text(lx + 46, R[1] + 40, r"$+\;2\cdot\mathbb{1}[\mathrm{a\ gated\ cap\ first\ engages}]$", size=11, ha="left")
    c.text(lx + 46, R[1] + 62, r"$-\;c_{\mathrm{jam}}-c_{\mathrm{action}}-c_{\mathrm{force}}$", size=11, ha="left")
    c.text(lx, R[1] + 89, r"N1 arm:  $-\,0.01\,c_{\mathrm{posture}}-0.001\,\|a_t-a_{t-1}\|^2$", size=11, ha="left")
    c.text(lx, R[1] + 110, "plus a flexion floor on the joint targets", size=10.5, ha="left", color=INK2)
    c.text(lx, R[1] + 125, "and a relaxed reset pose", size=10.5, ha="left", color=INK2)
    c.arrow([(R[0], cyc), (Q[2], cyc)], lw=1.1, color=TRN[1], head=6.5, ls="--")
    c.text((R[0] + Q[2]) / 2, cyc - 11, r"$r_t$", size=11.5, free=True)
    # curriculum (right column, below the reward)
    G = (R[0], R[3] + 12, R[2], R[3] + 104)
    c.box(*G, fc="white", ec=TRN[1], lw=1.0, r=8, z=2)
    c.text(lx, G[1] + 18, r"gate curriculum:  $s\,\theta_p$ (M2),  $s\,\theta_s$ (M3)", size=11, ha="left")
    c.text(lx, G[1] + 40, r"$s:0.25\rightarrow 1$, $+0.05$ once a family's success", size=11, ha="left")
    c.text(lx, G[1] + 60, r"over $\geq 200$ episodes exceeds $0.3$", size=11, ha="left")
    c.text(lx, G[1] + 81, "evaluation always at the full thresholds", size=10.5, ha="left", color=INK2)
    yl = G[3] - 26
    c.arrow([(G[0], yl), (V[2], yl)], lw=1.2, ls="--", color=TRN[1])
    c.text((k0 + k1) / 2 + 20, yl - 11, r"thresholds $\theta_p,\theta_s$", size=11)

    # ---------------- legend (one row, centred)
    ly = H - 20
    items = [(ARROW, "-", "30 Hz control loop: the policy's inputs are only what a real hand senses"),
             (VML[1], "-", "simulator internals, every physics substep (120 Hz)"),
             (TRN[1], "--", "training only")]
    widths = [44 + 8 + c.width(s, 11.5) for _, _, s in items]
    gap = 34
    x = (W - sum(widths) - gap * (len(items) - 1)) / 2
    for (col, ls, s), wdt in zip(items, widths):
        c.arrow([(x, ly), (x + 40, ly)], color=col, lw=1.4, ls=ls, head=7)
        c.text(x + 48, ly, s, size=11.5, ha="left")
        x += wdt + gap
    return c.save("method", preview_dir)


# ---------------------------------------------------------------------------------------------------- mechanisms
SHELL = ("#f4f4f1", "#7d7b74")
INNER = ("#d9d8d3", "#7d7b74")
BAND = "#14a88f"          # render paint MechBand (0, 0.62, 0.52), a touch lighter
TAB = "#f39a3d"           # render paint MechTab, as it looks under the scene light
BOTTLE = ("#f2bd6b", "#b8741a")
PART = "#55534d"


def closure_section(c, X, fam):
    """section view of the closure (not to scale): outer shell with tabs, inner threaded cap with the teal band,
    bottle neck and shoulder; the inside differs by family (coupling keys, press spring and gate teeth, tab pawls,
    seal)"""
    y_sh, so, sw, sh, st = 122, 64, 8, 72, 9           # shell: top y, outer half width, wall, height, top
    y_in, io, iw = y_sh + st + 13, 53, 8                # inner cap: top y, outer half width, wall
    y_band0, y_band1 = y_sh + sh, y_sh + sh + 13        # teal band below the shell
    no, y_neck1, bo, y_cut = 39, 230, 66, 270           # neck half width and bottom, shoulder half width, cut
    # bottle (neck + shoulder), cut at the bottom
    body = [(X - no, y_in + iw), (X + no, y_in + iw), (X + no, y_neck1), (X + bo, y_neck1 + 16), (X + bo, y_cut),
            (X - bo, y_cut), (X - bo, y_neck1 + 16), (X - no, y_neck1)]
    c.ax.add_patch(Polygon(body, closed=True, fc=BOTTLE[0], ec="none", lw=0, zorder=2))
    c.line(body[-3:] + body[:5], color=BOTTLE[1], lw=1.0, z=2.1)
    c.line([(X - bo, y_cut), (X + bo, y_cut)], color=BOTTLE[1], lw=0.8, ls=(0, (3, 2)), z=2.1)
    # neck thread (teeth on both sides)
    for yy in np.arange(y_in + 18, y_band1 - 4, 9):
        for sgn in (-1, 1):
            xb = X + sgn * no
            c.ax.add_patch(Polygon([(xb, yy - 3.5), (xb + sgn * 4.5, yy), (xb, yy + 3.5)], closed=True,
                                   fc=BOTTLE[1], ec="none", zorder=2.5))
    # inner cap (U section) and its band
    for sgn in (-1, 1):
        c.box(min(X + sgn * io, X + sgn * (io - iw)), y_in, max(X + sgn * io, X + sgn * (io - iw)), y_band1,
              fc=INNER[0], ec=INNER[1], lw=0.9, r=1.5, z=3, region=False)
        c.box(min(X + sgn * io, X + sgn * (io - iw)), y_band0, max(X + sgn * io, X + sgn * (io - iw)), y_band1,
              fc=BAND, ec=INNER[1], lw=0.9, r=1.5, z=3.2, region=False)
    c.box(X - io, y_in, X + io, y_in + iw, fc=INNER[0], ec=INNER[1], lw=0.9, r=1.5, z=3.1, region=False)
    # outer shell (U section) with the orange tabs
    for sgn in (-1, 1):
        c.box(min(X + sgn * so, X + sgn * (so - sw)), y_sh, max(X + sgn * so, X + sgn * (so - sw)), y_sh + sh,
              fc=SHELL[0], ec=SHELL[1], lw=1.0, r=1.5, z=4, region=False)
        c.box(min(X + sgn * so, X + sgn * (so + 12)), y_sh + 20, max(X + sgn * so, X + sgn * (so + 12)),
              y_sh + 46, fc=TAB, ec="#b8641c", lw=0.9, r=2, z=4.1, region=False)
    c.box(X - so, y_sh, X + so, y_sh + st, fc=SHELL[0], ec=SHELL[1], lw=1.0, r=2, z=4.1, region=False)
    gap0, gap1 = y_sh + st, y_in
    if fam in ("M1", "M6"):           # shell keyed to the inner cap: turning the shell turns the thread
        for sgn in (-1, 1):
            c.box(X + sgn * 22 - 4, gap0, X + sgn * 22 + 4, gap1, fc=PART, ec="none", lw=0, r=1, z=4.3,
                  region=False)
    if fam == "M2":                   # press spring and gate teeth (open until pushed)
        zz = np.linspace(gap0 + 1, gap1 - 1, 7)
        c.line([(X + (5 if i % 2 else -5), y) for i, y in enumerate(zz)], color=PART, lw=1.0, z=4.4)
        for sgn in (-1, 1):
            xt = X + sgn * 28
            c.ax.add_patch(Polygon([(xt - 5, gap0), (xt + 5, gap0), (xt, gap0 + 6)], closed=True, fc=PART,
                                   ec="none", zorder=4.4))
            xb = X + sgn * 38
            c.ax.add_patch(Polygon([(xb - 5, gap1), (xb + 5, gap1), (xb, gap1 - 6)], closed=True, fc=PART,
                                   ec="none", zorder=4.4))
    if fam == "M3":                   # each tab drives a pawl toward the inner cap
        for sgn in (-1, 1):
            c.line([(X + sgn * (so + 2), y_sh + 33), (X + sgn * (io + 7), y_sh + 33)], color=PART, lw=1.6, z=4.5)
            c.box(min(X + sgn * (io + 1), X + sgn * (io + 7)), y_sh + 27, max(X + sgn * (io + 1), X + sgn * (io + 7)),
                  y_sh + 39, fc=PART, ec="none", lw=0, r=1, z=4.5, region=False)
    if fam == "M6":                   # seal between the inner cap and the bottle
        for sgn in (-1, 1):
            x0, x1 = sorted((X + sgn * no, X + sgn * (io - iw)))
            c.box(x0 - 1, y_band1 - 9, x1 + 1, y_band1 + 3, fc="#3d3b37", ec="none", lw=0, r=1, z=3.4,
                  region=False)
    return dict(y_sh=y_sh, so=so, sh=sh, y_band1=y_band1, gap0=gap0)


def burst(c, x, y, r, color, z=8):
    t = np.linspace(0, 2 * np.pi, 17)[:-1]
    rr = np.where(np.arange(16) % 2 == 0, r, r * 0.45)
    c.ax.add_patch(Polygon(list(zip(x + rr * np.cos(t), y + rr * np.sin(t))), closed=True, fc=color, ec="none",
                           zorder=z))


def mechanism_figure(preview_dir=None):
    W, H = 1200, 420
    c = Canvas(W, H)
    top, bot = 12, 384
    # left: the render (identical for all four families)
    L = (12, top, 232, bot)
    c.box(*L, fc="#f6f6f4", ec="#c9c7c0", lw=1.0, r=10, z=0)
    c.text((L[0] + L[2]) / 2, 34, "One shape", size=14)
    iw = 196
    ih = c.image(ASSETS / "cap_closeup.jpg", L[0] + 12, 54, iw)
    c.text((L[0] + L[2]) / 2, 54 + ih + 15, "the simulated closure, rendered", size=11, color=INK2)
    ye = 54 + ih + 47
    c.eye_slash(L[0] + 26, ye, s=1.0, color=INK, lw=1.2)
    c.text(L[0] + 46, ye - 9, "type hidden", size=12, ha="left")
    c.text(L[0] + 46, ye + 9, "from the policy", size=12, ha="left")
    lines = ["Shell, tabs and band are", "the same in every episode;", "the mechanism layer behind",
             "the joints sets the family", "and its parameters."]
    for i, s in enumerate(lines):
        c.text(L[0] + 14, ye + 40 + i * 19, s, size=11, ha="left", color=INK)
    c.text(L[0] + 14, bot - 16, "Paint as in the clips.", size=10.5, ha="left", color=INK2)

    cols = [("M1", "plain thread", [r"Turn to unscrew; pressing", r"past $\theta_p$ jams the thread."]),
            ("M2", "push-and-turn", [r"Push past $\theta_p$ while turning;", r"the thread advances while",
                                     r"pushed (plus a short window)."]),
            ("M3", "squeeze-and-turn", [r"Squeeze both tabs past $\theta_s$", r"while turning; squeezed turns",
                                        r"count (plus a short window)."]),
            ("M6", "stuck seal", [r"Twist past the breakaway torque", r"$\tau_b$, then unscrew as M1."])]
    x_start, cw = 246, 236.5
    for i, (code, name, desc) in enumerate(cols):
        x0 = x_start + i * cw
        x1 = x0 + cw - 8
        X = (x0 + x1) / 2
        c.box(x0, top, x1, bot, fc="white", ec="#c9c7c0", lw=1.0, r=10, z=0)
        c.text(X, 34, f"{code}  {name}", size=14)
        g = closure_section(c, X, code)
        y_sh, so = g["y_sh"], g["so"]
        # every family turns the shell (front of the shell moves right = counter-clockwise from above = opening);
        # colour marks only the input that differs between the families
        turn_col = TWIST if code == "M6" else ARROW
        c.ellipse_arrow(X, y_sh + 50, 172, 42, 163, 20, color=turn_col, lw=2.2 if code == "M6" else 1.5,
                        head=10 if code == "M6" else 8.5, z=9)
        c.text(X - 84, y_sh + 80, "twist" if code == "M6" else "turn", size=11.5)
        if code == "M1":
            c.arrow([(X, 62), (X, y_sh - 3)], color=MUTED, lw=1.6, head=9, z=9)
            c.line([(X - 8, 78), (X + 8, 94)], color=JAM, lw=2.6, z=10)
            c.line([(X - 8, 94), (X + 8, 78)], color=JAM, lw=2.6, z=10)
            c.text(X + 14, 86, r"press $\Rightarrow$ jam", size=12, ha="left")
        if code == "M2":
            c.arrow([(X, 56), (X, y_sh - 3)], color=PUSH, lw=3.2, head=12, z=9)
            c.text(X + 12, 80, r"push  $z_p>\theta_p$", size=12, ha="left")
        if code == "M3":
            yy = y_sh + 33
            c.arrow([(x0 + 10, yy), (X - so - 14, yy)], color=SQUEEZE, lw=3.0, head=11, z=9)
            c.arrow([(x1 - 10, yy), (X + so + 14, yy)], color=SQUEEZE, lw=3.0, head=11, z=9)
            c.text(X, 84, r"squeeze both tabs  $z_s>\theta_s$", size=12)
        if code == "M6":
            c.text(X, 84, r"twist  $|\tau|\geq\tau_b$  breaks the seal", size=12)
            burst(c, X + 58, g["y_band1"] - 3, 9, TWIST, z=8)
            c.line([(X + 64, g["y_band1"] + 4), (X + 80, g["y_band1"] + 18)], color=MUTED, lw=0.8, z=8)
            c.text(x1 - 12, g["y_band1"] + 26, "seal", size=11, ha="right")
        for j, s in enumerate(desc):
            c.text(X, 302 + j * 21, s, size=12)
    c.text((x_start + x_start + 4 * cw - 8) / 2, H - 19,
           r"Redrawn every episode (full thresholds): $\theta_p$ 3–5 mm, $\theta_s$ 2–3 mm, "
           r"$\tau_b$ 0.10–0.25 N m, 1–2 turns, unlock window 0–0.5 rad.", size=11.5, color=INK2)
    return c.save("mechanisms", preview_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preview-dir", default=None)
    a = p.parse_args()
    print("method.svg", method_figure(a.preview_dir), "bytes")
    print("mechanisms.svg", mechanism_figure(a.preview_dir), "bytes")


if __name__ == "__main__":
    main()
