"""Method overview and mechanism figures of the CR-Open project page (static/images/method.svg, mechanisms.svg).

Drawn with matplotlib: serif (STIX) text and math, converted to paths so the SVG looks the same in every browser;
the render thumbnails in tools/fig_assets/ are embedded as JPEG (matplotlib embeds PNG; this script swaps in a
JPEG of the same pixels to keep the files small).

Labels are short Title-Case nouns, as in the method figures of related papers (group titles Isaac Lab, Observation,
Policy, Training; a legend of arrow types). Red badges 1-4 mark the CR-Open-specific parts and are cited by the
Method paragraph of index.html: (1) Mechanism Layer, (2) Type Hidden, (3) Unlock Bonus, (4) Threshold Curriculum.
What the boxes show follows the training code (dexrl-cropen-v2/cropen: env_cropen.py, vml.py, train_ppo.py) and
the recipe's command line (P1A3 recipe R2: --r-unlock 2.0 --obs-history 8 --gate-curriculum --gate-init 0.25
--gate-step 0.05 --gate-up 0.3):
  actor input (Observation) = [joint positions, joint velocities, joint targets, 4 fingertip contact forces (3D),
               last action] (the order of _frame()), stacked over --obs-history 8 control steps; no type, no closure
               state;
  critic input (Privileged State) = the current step plus closure joint states, the type (one-hot), the mechanism
               state and the mechanism parameters (training only);
  Mechanism Layer (vml.py) = every physics substep the env reads the closure joints, steps the layer and writes the
               closure drive targets and effort limits (amber arrows);
  Reward = thread progress, release, lift, success, a one-off bonus when a gated (M2/M3) cap first engages, jam /
               action / force costs (plus the posture terms of the N1 variant; not drawn);
  Threshold Curriculum = the M2 press and M3 squeeze thresholds start at 0.25 of their value and rise by 0.05 when
               that type's success over >= 200 episodes exceeds 0.3; evaluation uses the full thresholds;
  Policy = Gaussian MLP actor trained with PPO; joint targets move by 0.08 rad x a_t per 30 Hz control step.
Rates, network sizes, dimensions and the reward terms are left to the paper.

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
GT, LB = 16, 14            # group titles (bold) and box labels
BADGE = "#c8102e"          # numbered key components, cited as (1)-(4) in the page's Method paragraph


def badge(c, x, y, n, r=10.0):
    """red numbered badge (as in the Method paragraph of the page)"""
    c.ax.add_patch(Circle((x, y), r, fc=BADGE, ec="white", lw=1.2, zorder=9))
    c.text(x, y + 0.5, str(n), size=11.5, color="white", weight="bold", z=9.5, free=True)


def method_figure(preview_dir=None):
    W, H = 1200, 538
    c = Canvas(W, H)
    T, lane = 48, 24                      # top of the groups; the action loop runs above them
    seg_parts = [16, 16, 16, 12, 16]                                         # the order of _frame()
    seg_cols = ["#2a78d6", "#5598e7", "#86b6ef", FORCE, "#b7d3f6"]           # q, qdot, q*, f, a_{t-1}

    # ---------------- groups (bold titles top left)
    A = (12, T, 420, 490)
    B = (440, T, 800, 272)
    C = (820, T, 1188, 272)
    D = (440, 290, 1188, 490)
    c.box(*A, fc=SIM[0], ec=SIM[1], lw=1.5, r=12, z=0)
    c.box(*B, fc=OBS[0], ec=OBS[1], lw=1.5, r=12, z=0)
    c.box(*C, fc=POL[0], ec=POL[1], lw=1.5, r=12, z=0)
    c.box(*D, fc=TRN[0], ec=TRN[1], lw=1.5, r=12, z=0, ls=(0, (5, 3)))
    for G, s in ((A, "Isaac Lab"), (B, "Observation"), (C, "Policy"), (D, "Training")):
        c.text(G[0] + 14, G[1] + 21, s, size=GT, ha="left", weight="bold")

    # ---------------- Isaac Lab: the closure, the hand, and the mechanism layer (1)
    cx, cy, cw = 28, 96, 190
    ch = c.image(ASSETS / "cap_closeup.jpg", cx, cy, cw)
    c.text(cx + cw / 2, cy + ch + 15, "Closure", size=LB)
    hx, hy, hw = 240, 80, 166
    hh = c.image(ASSETS / "hand_over_bottle.jpg", hx, hy, hw)
    c.text(hx + hw / 2, hy + hh + 15, "Allegro Hand", size=LB)
    V = (28, 342, 404, 452)
    c.box(*V, fc=VML[0], ec=VML[1], lw=1.4, r=10, z=1)
    badge(c, V[0] + 22, V[1] + 24, 1)
    c.text(V[0] + 40, V[1] + 24, "Mechanism Layer", size=15, ha="left", weight="bold")
    c.text(V[0] + 16, V[1] + 73, r"Type $k$", size=LB, ha="left")
    for i, (code, col) in enumerate(zip(("M1", "M2", "M3", "M6"), (ARROW, PUSH, SQUEEZE, TWIST))):
        x0 = V[0] + 86 + i * 72
        c.box(x0, V[1] + 60, x0 + 62, V[1] + 86, fc="white", ec=col, lw=1.5, r=6, z=2)
        c.text(x0 + 31, V[1] + 73, code, size=13.5)
    # every physics substep the layer reads the closure joints and sets their drive targets (amber: simulator)
    y0, y1 = cy + ch + 26, V[1]
    xr, xw = 92, 128
    c.arrow([(xr, y0), (xr, y1)], color=VML[1], lw=1.4)
    c.arrow([(xw, y1), (xw, y0)], color=VML[1], lw=1.4)
    c.text(xr - 8, (y0 + y1) / 2, r"$q_c$", size=LB, ha="right")
    c.text(xw + 8, (y0 + y1) / 2, "Drive Targets", size=LB, ha="left")

    # ---------------- Observation: five signals of one control step, stacked over 8 steps; the type is hidden (2)
    names = ["Joint Positions", "Joint Velocities", "Joint Targets", "Fingertip Forces", "Last Action"]
    rx = B[0] + 14
    ys = [94 + 22 * i for i in range(5)]
    for y, s, col in zip(ys, names, seg_cols):
        c.feature_bar(rx, y - 4.5, 14, 9, [1], [col])
        c.text(rx + 22, y, s, size=LB, ha="left")
    fw = 112
    fx, fy = B[2] - 12 - fw, 96
    fh = c.image(ASSETS / "fingertips_on_cap.jpg", fx, fy, fw)
    c.box(fx, fy, fx + fw, fy + fh, fc="none", ec=FORCE, lw=1.4, r=6, z=3.6, region=False)
    xf = rx + 22 + c.width("Fingertip Forces", LB) + 8
    c.line([(xf, ys[3]), (fx - 2, ys[3])], color=FORCE, lw=1.1)
    c.arrow([(hx + hw + 3, ys[1]), (rx - 4, ys[1])], lw=1.3)          # hand -> observation
    n, dx, dy = 8, 3.5, 2.5
    sx0, sy0, sw, sh = rx, 194, 176, 18
    for i in range(n):
        k = n - 1 - i                           # back cards first
        x, y = sx0 + k * dx, sy0 + i * dy
        c.box(x, y, x + sw, y + sh, fc="white", ec="#9ec5f4", lw=0.8, r=4, z=3 + i * 0.1, region=False)
    fy0 = sy0 + (n - 1) * dy                    # the front card is the current step
    c.feature_bar(sx0 + 7, fy0 + 5, sw - 14, 8, seg_parts, seg_cols, z=4.5)
    ay = fy0 + sh / 2
    stack_right = sx0 + sw + (n - 1) * dx
    yh = 252
    badge(c, rx + 8, yh, 2)
    c.eye_slash(rx + 36, yh, s=0.95, color=INK, lw=1.2)
    c.text(rx + 54, yh, "Type Hidden", size=LB, ha="left")

    # ---------------- Policy
    tx0, tx1 = C[0] + 32, C[0] + 152
    c.arrow([(stack_right + 4, ay), (tx0 - 2, ay)], lw=1.3)
    c.text((stack_right + B[2]) / 2, ay - 13, "History (8 Steps)", size=13)
    c.text(C[0] + 16, ay - 13, r"$o_t$", size=LB + 1)
    c.trapezoid(tx0, tx1, ay, 92, 46, fc=POL[1])
    c.text((tx0 + tx1) / 2 - 4, ay, r"$\pi$", size=22, color="white")
    c.arrow([(tx1, ay), (tx1 + 34, ay)], lw=1.3)
    Pa = (tx1 + 36, ay - 17, tx1 + 106, ay + 17)
    c.box(*Pa, fc="white", ec=POL[1], lw=1.1, r=8, z=3)
    c.text((Pa[0] + Pa[2]) / 2, ay, r"$a_t$", size=LB + 1)
    lx = hx + hw / 2
    c.arrow([(Pa[2], ay), (C[2] - 20, ay), (C[2] - 20, lane), (lx, lane), (lx, hy - 1)], lw=1.3)
    c.text((lx + C[2] - 20) / 2, lane - 12, "Joint Targets", size=LB)

    # ---------------- Training only (dashed): asymmetric critic, PPO, reward with the unlock bonus (3), curriculum (4)
    cyc = 358
    P = (456, 324, 634, 432)
    c.box(*P, fc=VML[0], ec=TRN[1], lw=1.0, r=8, z=2)
    c.text((P[0] + P[2]) / 2, P[1] + 18, "Privileged State", size=LB)
    for i, s in enumerate(["Closure Joints", r"Type $k$", "Mechanism State", "Mechanism Parameters"]):
        c.text(P[0] + 12, P[1] + 42 + i * 17, s, size=12, ha="left", color=INK2)
    c.arrow([(V[2], 378), (P[0] - 1, 378)], lw=1.1, ls="--", color=TRN[1], head=6.5)
    k0, k1 = 666, 748
    xm = (P[2] + k0) / 2                        # the critic also sees the current observation step
    c.line([(P[2], cyc), (xm, cyc)], color=TRN[1], lw=1.1, ls="--")
    c.arrow([(xm, B[3]), (xm, cyc), (k0 - 1, cyc)], lw=1.1, ls="--", color=TRN[1], head=6.5)
    c.trapezoid(k0, k1, cyc, 88, 44, fc=TRN[1])
    c.text((k0 + k1) / 2 - 3, cyc, r"$V$", size=18, color="white")
    c.text((k0 + k1) / 2 - 4, cyc + 54, "Critic", size=LB)
    Q = (776, cyc - 21, 842, cyc + 21)
    c.arrow([(k1, cyc), (Q[0] - 1, cyc)], lw=1.1, ls="--", color=TRN[1], head=6.5)
    c.box(*Q, fc="white", ec=TRN[1], lw=1.0, r=8, z=3)
    c.text((Q[0] + Q[2]) / 2, cyc, "PPO", size=LB + 0.5)
    qx, pxc = (Q[0] + Q[2]) / 2, (tx0 + tx1) / 2
    gap_y = (B[3] + D[1]) / 2
    c.arrow([(qx, Q[1]), (qx, gap_y), (pxc, gap_y), (pxc, ay + 36)], lw=1.2, ls="--", color=TRN[1])
    R = (884, 324, 1172, 392)
    c.box(*R, fc="white", ec=TRN[1], lw=1.0, r=8, z=2)
    c.text(R[0] + 16, R[1] + 21, "Reward", size=LB, ha="left")
    c.text(R[0] + 16, R[1] + 47, "+ Unlock Bonus", size=LB, ha="left")
    badge(c, R[0] + 16 + c.width("+ Unlock Bonus", LB) + 18, R[1] + 47, 3)
    c.arrow([(R[0], cyc), (Q[2] + 1, cyc)], lw=1.1, ls="--", color=TRN[1], head=6.5)
    c.text((R[0] + Q[2]) / 2, cyc - 12, r"$r_t$", size=LB)
    G = (884, 408, 1172, 474)
    yg = (G[1] + G[3]) / 2
    c.box(*G, fc="white", ec=TRN[1], lw=1.0, r=8, z=2)
    badge(c, G[0] + 22, yg, 4)
    c.text(G[0] + 40, yg, "Threshold Curriculum", size=LB, ha="left")
    s0 = G[2] - 70
    steps = [(s0, yg + 14)]
    for j in range(4):
        steps += [(s0 + 12 * j + 12, yg + 14 - 9 * j), (s0 + 12 * j + 12, yg + 5 - 9 * j)]
    c.line(steps, color=TRN[1], lw=1.4)
    yt = yg + 1
    c.arrow([(G[0], yt), (V[2] + 1, yt)], lw=1.2, ls="--", color=TRN[1])
    c.text(Q[0] + 34, yt - 12, "Thresholds", size=13)

    # ---------------- legend (one row, centred)
    ly = H - 22
    items = [(ARROW, "-", "Control Loop"), (VML[1], "-", "Simulator"), (TRN[1], "--", "Training Only")]
    widths = [44 + 8 + c.width(s, LB) for _, _, s in items]
    gap = 44
    x = (W - sum(widths) - gap * (len(items) - 1)) / 2
    for (col, ls, s), wdt in zip(items, widths):
        c.arrow([(x, ly), (x + 40, ly)], color=col, lw=1.4, ls=ls, head=7)
        c.text(x + 48, ly, s, size=LB, ha="left")
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
    W, H = 1200, 304
    c = Canvas(W, H)
    top, bot = 10, 292
    # left: the render (identical for all four types)
    L = (12, top, 232, bot)
    c.box(*L, fc="#f6f6f4", ec="#c9c7c0", lw=1.0, r=10, z=0)
    c.text((L[0] + L[2]) / 2, 34, "Same Geometry", size=15, weight="bold")
    iw = 196
    ih = c.image(ASSETS / "cap_closeup.jpg", L[0] + 12, 66, iw)
    ye = 66 + ih + 42
    tw = c.width("Type Hidden", 13.5)
    xe = (L[0] + L[2]) / 2 - (tw + 30) / 2 + 10
    c.eye_slash(xe, ye, s=1.0, color=INK, lw=1.2)
    c.text(xe + 20, ye, "Type Hidden", size=13.5, ha="left")

    cols = [("M1", "Plain Thread"), ("M2", "Push-and-Turn"), ("M3", "Squeeze-and-Turn"), ("M6", "Stuck Seal")]
    x_start, cw = 246, 236.5
    for i, (code, name) in enumerate(cols):
        x0 = x_start + i * cw
        x1 = x0 + cw - 8
        X = (x0 + x1) / 2
        c.box(x0, top, x1, bot, fc="white", ec="#c9c7c0", lw=1.0, r=10, z=0)
        c.text(X, 34, f"{code}  {name}", size=15, weight="bold")
        g = closure_section(c, X, code)
        y_sh, so = g["y_sh"], g["so"]
        # every type turns the shell (front of the shell moves right = counter-clockwise from above = opening);
        # colour marks only the input that differs between the types
        turn_col = TWIST if code == "M6" else ARROW
        c.ellipse_arrow(X, y_sh + 50, 172, 42, 163, 20, color=turn_col, lw=2.2 if code == "M6" else 1.5,
                        head=10 if code == "M6" else 8.5, z=9)
        c.text(X - 84, y_sh + 80, "Twist" if code == "M6" else "Turn", size=13)
        if code == "M1":
            c.arrow([(X, 62), (X, y_sh - 3)], color=MUTED, lw=1.6, head=9, z=9)
            c.line([(X - 8, 78), (X + 8, 94)], color=JAM, lw=2.6, z=10)
            c.line([(X - 8, 94), (X + 8, 78)], color=JAM, lw=2.6, z=10)
            c.text(X + 16, 86, "Press Jams", size=13, ha="left")
        if code == "M2":
            c.arrow([(X, 56), (X, y_sh - 3)], color=PUSH, lw=3.2, head=12, z=9)
            c.text(X + 14, 82, "Push", size=13, ha="left")
        if code == "M3":
            yy = y_sh + 33
            c.arrow([(x0 + 10, yy), (X - so - 14, yy)], color=SQUEEZE, lw=3.0, head=11, z=9)
            c.arrow([(x1 - 10, yy), (X + so + 14, yy)], color=SQUEEZE, lw=3.0, head=11, z=9)
            c.text(X, 84, "Squeeze", size=13)
        if code == "M6":
            c.text(X, 84, "Break Seal", size=13)
            burst(c, X + 58, g["y_band1"] - 3, 9, TWIST, z=8)
            c.line([(X + 64, g["y_band1"] + 4), (X + 80, g["y_band1"] + 18)], color=MUTED, lw=0.8, z=8)
            c.text(x1 - 12, g["y_band1"] + 26, "Seal", size=12.5, ha="right")
    return c.save("mechanisms", preview_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preview-dir", default=None)
    a = p.parse_args()
    print("method.svg", method_figure(a.preview_dir), "bytes")
    print("mechanisms.svg", mechanism_figure(a.preview_dir), "bytes")


if __name__ == "__main__":
    main()
