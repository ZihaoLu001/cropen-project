"""Method overview and mechanism figures of the CR-Open project page (static/images/method.svg, mechanisms.svg).

Drawn with matplotlib: serif (STIX) text and math, converted to paths so the SVG looks the same in every browser;
the render thumbnails in tools/fig_assets/ are embedded as JPEG (matplotlib embeds PNG; this script swaps in a
JPEG of the same pixels to keep the files small). Every label is taken from the code of the training checkout
(dexrl-cropen-v2/cropen: env_cropen.py, vml.py, train_ppo.py) and the recipe's command line (README P1, recipe N1):
  actor input  x_t = [q, qdot, q*, f (4 fingertips x 3), a_{t-1}] (76 numbers), stacked over --obs-history 8 steps;
  critic input s_t = [x_t, q_c, qdot_c, family one-hot, mechanism state (6), mechanism parameters (5)] (101);
  rl_games MLPs 512-256-128 (ELU) for the actor and the separate central-value critic;
  PhysX at 120 Hz, decimation 4 (30 Hz control), action scale 0.08 rad;
  reward: thread progress, release, lift, success, a one-off 2.0 when a gated cap first engages (--r-unlock 2.0),
  jam / action / force costs; N1 adds --w-posture -0.01, --w-rate -0.001, a flexion floor and a relaxed reset pose;
  gate curriculum: --gate-init 0.25 --gate-step 0.05 --gate-up 0.3 (over >= 200 episodes); evaluation at full gates.

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
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tools" / "fig_assets"
OUT = ROOT / "static" / "images"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "svg.fonttype": "path", "svg.hashsalt": "cropen",
    "axes.linewidth": 0, "savefig.facecolor": "white",
})

INK, INK2, MUTED = "#1d1d1b", "#52514e", "#898781"
ARROW = "#3b3b38"
# group colours (pale fill, darker stroke)
SIM = ("#f6f6f4", "#9a978f")
VML = ("#fff5e6", "#cf8a1f")
OBS = ("#eef4fc", "#2a78d6")
POL = ("#eef8f0", "#2f8f46")
TRN = ("#f6f3fd", "#7a62c9")
# family inputs (validated: dataviz validate_palette.js, all pairs, light): push aqua, squeeze orange, twist violet
PUSH, SQUEEZE, TWIST, JAM = "#1baf7a", "#eb6834", "#6250d6", "#d03b3b"


# ---------------------------------------------------------------------------------------------------- primitives
class Canvas:
    def __init__(self, w, h, scale=0.01):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w * scale, h * scale))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")

    def box(self, x0, y0, x1, y1, fc="white", ec=INK2, lw=1.0, r=8, ls="-", z=1, alpha=1.0):
        p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                           lw=lw, ls=ls, zorder=z, alpha=alpha, joinstyle="round")
        self.ax.add_patch(p)
        return p

    def text(self, x, y, s, size=10, ha="center", va="center", color=INK, z=6, **kw):
        return self.ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, zorder=z, **kw)

    def line(self, pts, color=ARROW, lw=1.0, ls="-", z=4):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt", dash_capstyle="butt",
                     dash_joinstyle="miter", solid_joinstyle="miter")

    def arrow(self, pts, color=ARROW, lw=1.0, ls="-", head=7.5, z=5, both=False):
        """orthogonal polyline with a filled head at the end (and the start if both)"""
        if len(pts) > 2:
            self.line(pts[:-1], color, lw, (0, (4, 3)) if ls == "--" else ls, z)
        style = "<|-|>" if both and len(pts) == 2 else "-|>"
        a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle=f"{style},head_length=0.55,head_width=0.28",
                            mutation_scale=head * 1.6, color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0,
                            linestyle=(0, (4, 3)) if ls == "--" else ls)
        self.ax.add_patch(a)
        if both and len(pts) > 2:
            b = FancyArrowPatch(pts[1], pts[0], arrowstyle="-|>,head_length=0.55,head_width=0.28",
                                mutation_scale=head * 1.6, color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0)
            self.ax.add_patch(b)

    def ellipse_arrow(self, x, y, w, h, deg0, deg1, color=ARROW, lw=1.0, head=7.5, z=7, ls="-"):
        """arc of the ellipse centred (x, y) from deg0 to deg1 (data angles; y grows downwards, so 270 is the
        top), drawn as a polyline with a head at deg1"""
        t = np.radians(np.linspace(deg0, deg1, 60))
        pts = list(zip(x + w / 2 * np.cos(t), y + h / 2 * np.sin(t)))
        self.arrow(pts, color=color, lw=lw, head=head, z=z, ls=ls)

    def image(self, path, x0, y0, w, r=6, z=3, h=None, frame=True):
        im = np.asarray(Image.open(path).convert("RGB"))
        h = h if h is not None else w * im.shape[0] / im.shape[1]
        art = self.ax.imshow(im, extent=(x0, x0 + w, y0 + h, y0), interpolation="none", zorder=z, aspect="auto")
        clip = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec="none",
                              transform=self.ax.transData)
        art.set_clip_path(clip)
        if frame:
            self.box(x0, y0, x0 + w, y0 + h, fc="none", ec="#d6d4cd", lw=0.8, r=r, z=z + 0.5)
        return h

    def trapezoid(self, x0, x1, yc, h_in, h_out, fc, ec=None, z=4):
        pts = [(x0, yc - h_in / 2), (x1, yc - h_out / 2), (x1, yc + h_out / 2), (x0, yc + h_in / 2)]
        self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec or fc, lw=1.0, zorder=z, joinstyle="round"))

    def eye_slash(self, x, y, s=1.0, color=INK2, z=7):
        self.ax.add_patch(Ellipse((x, y), 20 * s, 11 * s, fc="white", ec=color, lw=1.0, zorder=z))
        self.ax.add_patch(Circle((x, y), 3.0 * s, fc=color, ec=color, lw=0, zorder=z + 0.1))
        self.line([(x - 9 * s, y + 7 * s), (x + 9 * s, y - 7 * s)], color=color, lw=1.3, z=z + 0.2)

    def feature_bar(self, x0, y0, w, h, parts, colors, z=5, gap=1.2):
        """horizontal bar split into parts (proportional widths)"""
        tot = float(sum(parts))
        x = x0
        for p, c in zip(parts, colors):
            pw = w * p / tot
            self.box(x + gap / 2, y0, x + pw - gap / 2, y0 + h, fc=c, ec="none", lw=0, r=1.5, z=z)
            x += pw

    def save(self, stem, preview_dir):
        svg = io.BytesIO()
        self.fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": None})
        text = jpeg_images(svg.getvalue().decode("utf-8"))
        (OUT / f"{stem}.svg").write_text(text, encoding="utf-8", newline="\n")
        if preview_dir:
            self.fig.savefig(Path(preview_dir) / f"{stem}.png", dpi=110)
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
    W, H = 1200, 720
    c = Canvas(W, H)
    T = 56  # top of the group boxes (the action lane runs above them)
    seg_parts = [16, 16, 16, 12, 16]
    seg_cols = ["#2a78d6", "#5598e7", "#86b6ef", "#eb6834", "#b7d3f6"]   # q, qdot, q*, f, a_{t-1}

    # ---------------- group boxes (titles at the bottom centre)
    A = (12, T, 412, 676)
    B = (432, T, 792, 352)
    C = (812, T, 1188, 352)
    D = (432, 370, 1188, 676)
    c.box(*A, fc=SIM[0], ec=SIM[1], lw=1.5, r=12, z=0)
    c.box(*B, fc=OBS[0], ec=OBS[1], lw=1.5, r=12, z=0)
    c.box(*C, fc=POL[0], ec=POL[1], lw=1.5, r=12, z=0)
    c.box(*D, fc=TRN[0], ec=TRN[1], lw=1.5, r=12, z=0, ls=(0, (5, 3)))
    c.text((A[0] + A[2]) / 2, A[3] - 14, "Simulation (IsaacLab, PhysX)", size=13)
    c.text((B[0] + B[2]) / 2, B[3] - 14, "Deployable observation", size=13)
    c.text((C[0] + C[2]) / 2, C[3] - 14, "Policy", size=13)
    c.text((D[0] + D[2]) / 2, D[3] - 14, "Training only: PPO with an asymmetric critic", size=13)

    # ---------------- A: render of the hand over the bottle; close-up of the closure and its joints
    hx, hy, hw = 226, T + 14, 162
    hh = c.image(ASSETS / "hand_over_bottle.jpg", hx, hy, hw)
    c.text(hx + hw / 2, hy + hh + 13, "Allegro hand, 16 joints, palm down", size=10, color=INK2)
    cx, cy, cw = 30, T + 58, 160
    ch = c.image(ASSETS / "cap_closeup.jpg", cx, cy, cw)
    # zoom lines from the cap in the render (about x 296..350, y 268..300 of the 480 x 496 crop)
    capx0, capx1 = hx + hw * 296 / 480, hx + hw * 350 / 480
    capy0, capy1 = hy + hh * 268 / 496, hy + hh * 302 / 496
    c.box(capx0, capy0, capx1, capy1, fc="none", ec=INK2, lw=0.8, r=3, z=6)
    c.line([(cx + cw, cy + 2), (capx0, capy0)], color=MUTED, lw=0.7, ls=(0, (2, 2)), z=6)
    c.line([(cx + cw, cy + ch - 2), (capx0, capy1)], color=MUTED, lw=0.7, ls=(0, (2, 2)), z=6)
    # joint glyphs: shell rotation (arc), press (down), tabs (inward)
    mid = cx + cw / 2
    # the far rim moves to the left when the shell turns the opening way (+z, counter-clockwise from above)
    c.ellipse_arrow(mid, cy - 1, 120, 22, 340, 200, color=INK, lw=1.2, head=6)
    c.arrow([(mid, cy - 24), (mid, cy + 4)], color=INK, lw=1.2, head=6)
    c.text(mid + 66, cy - 12, r"$\theta$", size=11)
    c.text(mid + 6, cy - 25, r"$z_p$", size=11, ha="left")
    ty_tab = cy + ch * 0.45
    c.arrow([(cx - 13, ty_tab), (cx + 3, ty_tab)], color=INK, lw=1.2, head=6)
    c.text(cx - 15, ty_tab - 11, r"$z_s$", size=11, ha="left")
    c.text(mid, cy + ch + 14, r"closure joints  $q_c$", size=11)
    c.text(mid, cy + ch + 30, r"shell $\theta$, press $z_p$, tabs $z_s$, lift $h$", size=10, color=INK2)

    # ---------------- A: the virtual mechanism layer (nested)
    V = (26, 330, 398, 640)
    c.box(*V, fc=VML[0], ec=VML[1], lw=1.4, r=10, z=1)
    c.text((V[0] + V[2]) / 2, V[3] - 13, "Virtual mechanism layer, every physics substep", size=11)
    xr, xw = 84, 112
    y0, y1 = cy + ch + 44, V[1]
    c.arrow([(xr, y0), (xr, y1)], lw=1.1)
    c.arrow([(xw, y1), (xw, y0)], lw=1.1)
    c.text(xr - 8, (y0 + y1) / 2, r"read $q_c$", size=10.5, ha="right")
    c.text(xw + 8, (y0 + y1) / 2 - 8, r"write drive targets $\hat q_c$", size=10.5, ha="left")
    c.text(xw + 8, (y0 + y1) / 2 + 8, r"and effort limits $\bar\tau_c$", size=10.5, ha="left")
    c.eye_slash(V[0] + 22, V[1] + 23, s=0.9)
    c.text(V[0] + 38, V[1] + 22, r"hidden family $k\in\{\mathrm{M1,\,M2,\,M3,\,M6}\}$", size=11, ha="left")
    c.text(V[0] + 38, V[1] + 42, r"parameters $\xi=(\theta_p,\theta_s,n,\tau_b,\tau_t)$, new each episode",
           size=10, ha="left", color=INK2)
    rules = [
        ("gate", r"engaged only while $z_p>\theta_p$ (M2) or $z_s>\theta_s$ (M3)"),
        ("jam", r"$z_p>\theta_p$ on M1, M3 or M6 holds the shell"),
        ("clutch", r"M6: held until $|\tau|\geq\tau_b$, then slips free"),
        ("thread", r"friction $\tau_t$; after $n$ engaged turns the cap is free"),
    ]
    y = V[1] + 78
    for name, rule in rules:
        c.box(V[0] + 12, y - 11, V[0] + 68, y + 11, fc="white", ec=VML[1], lw=0.9, r=5, z=2)
        c.text(V[0] + 40, y, name, size=10.5, style="italic")
        c.text(V[0] + 78, y, rule, size=10, ha="left")
        y += 44
    c.text(V[0] + 12, y - 8, "at run time only drive targets and effort limits change",
           size=9.5, ha="left", color=INK2)

    # ---------------- B: deployable observation
    fx, fy, fw = B[0] + 16, T + 14, 128
    fh = c.image(ASSETS / "fingertips_on_cap.jpg", fx, fy, fw)
    c.feature_bar(fx + 6, fy + fh + 8, 16, 9, [1], [seg_cols[3]])
    c.text(fx + 28, fy + fh + 12, "fingertip forces", size=10.5, ha="left")
    c.text(fx + fw / 2, fy + fh + 29, r"$f_t\in\mathbb{R}^{4\times 3}$", size=10.5)
    tokens = [(r"$q_t\in\mathbb{R}^{16}$", "joint positions", 0),
              (r"$\dot q_t\in\mathbb{R}^{16}$", "joint velocities", 1),
              (r"$q^{*}_t\in\mathbb{R}^{16}$", "joint targets", 2),
              (r"$a_{t-1}\in\mathbb{R}^{16}$", "last action", 4)]
    tx, ty = B[0] + 164, T + 24
    for i, (m, s, j) in enumerate(tokens):
        yy = ty + i * 25
        c.feature_bar(tx, yy - 4.5, 16, 9, [1], [seg_cols[j]])
        c.text(tx + 23, yy - 1, m, size=10.5, ha="left")
        c.text(tx + 92, yy, s, size=10, ha="left", color=INK2)
    # one frame of 76 numbers
    bx0, bw, by = B[0] + 16, 228, T + 160
    c.feature_bar(bx0, by, bw, 12, seg_parts, seg_cols)
    c.text(bx0 + bw + 10, by + 6, r"one step  $x_t\in\mathbb{R}^{76}$", size=10.5, ha="left")
    c.arrow([(fx + fw / 2, fy + fh + 39), (fx + fw / 2, by - 1)], lw=1.0, head=6)
    c.arrow([(tx + 8, ty + 3 * 25 + 6), (tx + 8, by - 1)], lw=1.0, head=6)
    # the history: 8 stacked frames
    n, dx, dy = 8, 4.5, 3.0
    sx0, sy0, swid, shgt = bx0, by + 34, 196, 20
    for i in range(n):
        k = n - 1 - i                         # back cards first
        x, y = sx0 + k * dx, sy0 + i * dy
        c.box(x, y, x + swid, y + shgt, fc="white", ec="#9ec5f4", lw=0.8, r=4, z=3 + i * 0.1)
    fy0 = sy0 + (n - 1) * dy                  # the front card (x_t) is at sx0
    c.feature_bar(sx0 + 7, fy0 + 6, swid - 14, 8, seg_parts, seg_cols, z=4.5)
    c.arrow([(bx0 + 40, by + 13), (bx0 + 40, sy0 - 1)], lw=1.0, head=6)
    stack_mid = fy0 + shgt / 2 - 6
    c.text(sx0 + swid + 40, stack_mid - 8, r"$o_t=x_{t-7:t}$", size=11.5, ha="left")
    c.text(sx0 + swid + 40, stack_mid + 10, r"$H=8$ steps", size=10, ha="left", color=INK2)
    c.eye_slash(B[0] + 28, B[3] - 40, s=0.8)
    c.text(B[0] + 42, B[3] - 40, r"no family and no closure state:  $k\notin o_t$", size=10.5, ha="left",
           color=INK2)

    # ---------------- C: actor
    ay = stack_mid + 1
    c.arrow([(B[2] - 2, ay), (C[0] + 20, ay)], lw=1.1)
    tx0, tx1 = C[0] + 22, C[0] + 142
    c.trapezoid(tx0, tx1, ay, 112, 56, fc=POL[1])
    c.text((tx0 + tx1) / 2 - 4, ay - 9, "MLP", size=12, color="white")
    c.text((tx0 + tx1) / 2 - 4, ay + 9, "512-256-128", size=10, color="white")
    c.text((tx0 + tx1) / 2, ay - 82, r"actor  $\pi_\theta(a_t\,|\,o_t)$", size=12)
    c.text((tx0 + tx1) / 2, ay - 66, r"Gaussian (fixed $\sigma$), mean at test", size=10, color=INK2)
    c.arrow([(tx1, ay), (tx1 + 42, ay)], lw=1.1)
    c.box(tx1 + 44, ay - 17, tx1 + 176, ay + 17, fc="white", ec=POL[1], lw=1.0, r=8, z=3)
    c.text(tx1 + 110, ay, r"$a_t\in[-1,1]^{16}$", size=11.5)
    lane = 30
    c.arrow([(tx1 + 176, ay), (C[2] - 14, ay), (C[2] - 14, lane), (hx + hw / 2, lane), (hx + hw / 2, hy - 1)],
            lw=1.2)
    c.text((hx + hw / 2 + C[2]) / 2, lane - 13,
           r"new joint targets  $q^{*}_{t+1}=\mathrm{clip}(q^{*}_t+0.08\,a_t)$,  PD drives, 30 Hz", size=11.5)
    # hand -> observation
    c.arrow([(hx + hw, fy + 44), (fx - 2, fy + 44)], lw=1.1)
    c.text((hx + hw + fx) / 2, fy + 31, r"$q_t,\dot q_t,f_t$", size=10.5,
           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    # ---------------- D: training-only signals
    P = (D[0] + 16, D[1] + 22, D[0] + 196, D[1] + 182)
    c.box(*P, fc="white", ec=TRN[1], lw=1.0, r=8, z=2)
    c.text((P[0] + P[2]) / 2, P[1] + 16, r"critic input  $s_t\in\mathbb{R}^{101}$", size=11)
    priv = [(r"$x_t$", "current step (76)"), (r"$q_c,\dot q_c$", "closure joints (5+5)"),
            (r"$k$", "family one-hot (4)"), (r"$m_t$", "mechanism state (6)"), (r"$\xi$", "parameters (5)")]
    for i, (m, s) in enumerate(priv):
        yy = P[1] + 42 + i * 25
        c.text(P[0] + 12, yy, m, size=11, ha="left")
        c.text(P[0] + 60, yy, s, size=10, ha="left", color=INK2)
    c.arrow([(V[2], P[1] + 92), (P[0], P[1] + 92)], lw=1.1, ls="--", color=TRN[1])
    cyc = (P[1] + P[3]) / 2
    c.arrow([(P[2], cyc), (P[2] + 16, cyc)], lw=1.0, color=TRN[1], head=6)
    k0, k1 = P[2] + 18, P[2] + 116
    c.trapezoid(k0, k1, cyc, 98, 48, fc=TRN[1])
    c.text((k0 + k1) / 2 - 3, cyc - 8, "MLP", size=11.5, color="white")
    c.text((k0 + k1) / 2 - 3, cyc + 9, "512-256-128", size=9.5, color="white")
    c.text((k0 + k1) / 2, cyc - 68, r"critic  $V_\phi(s_t)$", size=12)
    Q = (k1 + 30, cyc - 21, k1 + 104, cyc + 21)
    c.arrow([(k1, cyc), (Q[0], cyc)], lw=1.0, color=TRN[1], head=6)
    c.box(*Q, fc="white", ec=TRN[1], lw=1.0, r=8, z=3)
    c.text((Q[0] + Q[2]) / 2, cyc, "PPO", size=12)
    qx = (Q[0] + Q[2]) / 2
    act_x = (tx0 + tx1) / 2
    act_bottom = ay + (112 + 56) / 4 + 2
    gap_y = (B[3] + D[1]) / 2
    c.arrow([(qx, Q[1]), (qx, gap_y), (act_x, gap_y), (act_x, act_bottom)], lw=1.1, ls="--", color=TRN[1])
    c.text(qx + 7, (Q[1] + D[1]) / 2 + 6, r"update $\theta,\phi$", size=10.5, ha="left")
    # reward (right column)
    R = (Q[2] + 28, D[1] + 16, D[2] - 14, D[1] + 150)
    c.box(*R, fc="white", ec=TRN[1], lw=1.0, r=8, z=2)
    lx = R[0] + 10
    c.text(lx, R[1] + 17, r"reward  $r_t=r_{\mathrm{thread}}+r_{\mathrm{release}}+r_{\mathrm{lift}}+r_{\mathrm{success}}$",
           size=11, ha="left")
    c.text(lx + 46, R[1] + 39, r"$+\;2\cdot\mathbb{1}[\mathrm{a\ gated\ cap\ first\ engages}]$", size=11, ha="left")
    c.text(lx + 46, R[1] + 61, r"$-\;c_{\mathrm{jam}}-c_{\mathrm{action}}-c_{\mathrm{force}}$", size=11, ha="left")
    c.text(lx, R[1] + 87, r"N1 arm:  $-\,0.01\,c_{\mathrm{posture}}-0.001\,\|a_t-a_{t-1}\|^2$", size=11, ha="left")
    c.text(lx, R[1] + 108, "plus a flexion floor on the joint targets", size=10, ha="left", color=INK2)
    c.text(lx, R[1] + 123, "and a relaxed reset pose", size=10, ha="left", color=INK2)
    c.arrow([(R[0], cyc), (Q[2], cyc)], lw=1.0, color=TRN[1], head=6)
    c.text((R[0] + Q[2]) / 2, cyc - 10, r"$r_t$", size=11)
    # curriculum (right column, below the reward)
    G = (R[0], R[3] + 12, R[2], R[3] + 100)
    c.box(*G, fc="white", ec=TRN[1], lw=1.0, r=8, z=2)
    c.text(lx, G[1] + 17, r"gate curriculum  $\theta_p$ (M2), $\theta_s$ (M3) $\leftarrow s\,\theta$", size=11,
           ha="left")
    c.text(lx, G[1] + 39, r"$s:0.25\rightarrow 1$, $+0.05$ once a family's success", size=11, ha="left")
    c.text(lx, G[1] + 59, r"over $\geq 200$ episodes exceeds $0.3$", size=11, ha="left")
    c.text(lx, G[1] + 78, "evaluation always at the full thresholds", size=10, ha="left", color=INK2)
    yl = G[3] - 28
    c.arrow([(G[0], yl), (V[2], yl)], lw=1.1, ls="--", color=TRN[1])
    c.text((P[0] + P[2]) / 2 + 20, yl - 10, r"thresholds $\theta_p,\theta_s$", size=10.5)

    # ---------------- legend
    ly = H - 18
    c.arrow([(352, ly), (392, ly)], lw=1.2)
    c.text(400, ly, "runs at every control step (needs only what a real hand senses)", size=11, ha="left")
    c.arrow([(788, ly), (828, ly)], lw=1.2, ls="--", color=TRN[1])
    c.text(836, ly, "training only", size=11, ha="left")
    return c.save("method", preview_dir)


# ---------------------------------------------------------------------------------------------------- mechanisms
SHELL = ("#f4f4f1", "#7d7b74")
INNER = ("#d9d8d3", "#7d7b74")
BAND = "#14a88f"          # render paint MechBand (0, 0.62, 0.52), a touch lighter
TAB = "#f39a3d"           # render paint MechTab, as it looks under the scene light
MARK = "#1f4fd8"          # render paint MechMark (0.03, 0.2, 0.85)
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
              fc=INNER[0], ec=INNER[1], lw=0.9, r=1.5, z=3)
        c.box(min(X + sgn * io, X + sgn * (io - iw)), y_band0, max(X + sgn * io, X + sgn * (io - iw)), y_band1,
              fc=BAND, ec=INNER[1], lw=0.9, r=1.5, z=3.2)
    c.box(X - io, y_in, X + io, y_in + iw, fc=INNER[0], ec=INNER[1], lw=0.9, r=1.5, z=3.1)
    # outer shell (U section) with the blue marker ridges and the orange tabs
    for sgn in (-1, 1):
        c.box(min(X + sgn * so, X + sgn * (so - sw)), y_sh, max(X + sgn * so, X + sgn * (so - sw)), y_sh + sh,
              fc=SHELL[0], ec=SHELL[1], lw=1.0, r=1.5, z=4)
        xm = X + sgn * (so - 2.5)
        c.box(xm - 1.5, y_sh + 12, xm + 1.5, y_sh + sh - 6, fc=MARK, ec="none", lw=0, r=0.8, z=4.2)
        c.box(min(X + sgn * so, X + sgn * (so + 12)), y_sh + 20, max(X + sgn * so, X + sgn * (so + 12)),
              y_sh + 46, fc=TAB, ec="#b8641c", lw=0.9, r=2, z=4.1)
    c.box(X - so, y_sh, X + so, y_sh + st, fc=SHELL[0], ec=SHELL[1], lw=1.0, r=2, z=4.1)
    gap0, gap1 = y_sh + st, y_in
    if fam in ("M1", "M6"):           # shell keyed to the inner cap: turning the shell turns the thread
        for sgn in (-1, 1):
            c.box(X + sgn * 22 - 4, gap0, X + sgn * 22 + 4, gap1, fc=PART, ec="none", lw=0, r=1, z=4.3)
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
                  y_sh + 39, fc=PART, ec="none", lw=0, r=1, z=4.5)
    if fam == "M6":                   # seal between the inner cap and the bottle
        for sgn in (-1, 1):
            x0, x1 = sorted((X + sgn * no, X + sgn * (io - iw)))
            c.box(x0 - 1, y_band1 - 9, x1 + 1, y_band1 + 3, fc="#3d3b37", ec="none", lw=0, r=1, z=3.4)
    return dict(y_sh=y_sh, so=so, sh=sh, y_band1=y_band1, gap0=gap0)


def burst(c, x, y, r, color, z=8):
    t = np.linspace(0, 2 * np.pi, 17)[:-1]
    rr = np.where(np.arange(16) % 2 == 0, r, r * 0.45)
    c.ax.add_patch(Polygon(list(zip(x + rr * np.cos(t), y + rr * np.sin(t))), closed=True, fc=color, ec="none",
                           zorder=z))


def mechanism_figure(preview_dir=None):
    W, H = 1200, 438
    c = Canvas(W, H)
    # left: the render (identical for all four families)
    L = (12, 12, 232, 382)
    c.box(*L, fc="#f6f6f4", ec="#c9c7c0", lw=1.0, r=10, z=0)
    c.text((L[0] + L[2]) / 2, 34, "One shape", size=14)
    iw = 196
    ih = c.image(ASSETS / "cap_closeup.jpg", L[0] + 12, 56, iw)
    c.text((L[0] + L[2]) / 2, 56 + ih + 16, "the simulated closure, rendered", size=10.5, color=INK2)
    c.eye_slash(L[0] + 26, 228, s=0.95)
    c.text(L[0] + 44, 228, "type hidden from the policy", size=11.5, ha="left")
    lines = ["Shell, tabs, band and thread are", "the same in every episode; only",
             "the mechanism layer behind the", "joints changes with the family."]
    for i, s in enumerate(lines):
        c.text(L[0] + 14, 260 + i * 19, s, size=11, ha="left", color=INK)
    c.text(L[0] + 14, 350, "Paint as in the clips: orange tabs,", size=10, ha="left", color=INK2)
    c.text(L[0] + 14, 366, "teal band, blue marker ridges.", size=10, ha="left", color=INK2)

    cols = [("M1", "plain thread", [r"Turn to unscrew; pressing", r"past $\theta_p$ jams the thread."]),
            ("M2", "push-and-turn", [r"Push past $\theta_p$ and turn; the", r"thread advances only while pushed."]),
            ("M3", "squeeze-and-turn", [r"Squeeze both tabs past $\theta_s$ and", r"turn; only squeezed turns count."]),
            ("M6", "stuck seal", [r"Twist past the breakaway torque", r"$\tau_b$, then unscrew as M1."])]
    x_start, cw = 246, 236.5
    for i, (code, name, desc) in enumerate(cols):
        x0 = x_start + i * cw
        X = x0 + (cw - 8) / 2
        c.box(x0, 12, x0 + cw - 8, 382, fc="white", ec="#c9c7c0", lw=1.0, r=10, z=0)
        c.text(X, 34, f"{code}  {name}", size=14)
        g = closure_section(c, X, code)
        y_sh, so = g["y_sh"], g["so"]
        # every family: turn the shell (front of the shell moves right = counter-clockwise from above = opening)
        turn_col = TWIST if code == "M6" else INK
        c.ellipse_arrow(X, y_sh + 50, 172, 42, 163, 20, color=turn_col, lw=2.0 if code == "M6" else 1.5, head=8,
                        z=9)
        c.text(X - 84, y_sh + 80, "turn", size=11)
        if code == "M1":
            c.arrow([(X, 62), (X, y_sh - 3)], color=MUTED, lw=1.6, head=8, z=9)
            c.line([(X - 8, 78), (X + 8, 94)], color=JAM, lw=2.4, z=10)
            c.line([(X - 8, 94), (X + 8, 78)], color=JAM, lw=2.4, z=10)
            c.text(X + 14, 86, r"press $\Rightarrow$ jam", size=11, ha="left")
        if code == "M2":
            c.arrow([(X, 58), (X, y_sh - 3)], color=PUSH, lw=3.0, head=10, z=9)
            c.text(X + 12, 80, r"push  $z_p>\theta_p$", size=11.5, ha="left")
        if code == "M3":
            yy = y_sh + 33
            c.arrow([(X - 112, yy), (X - so - 14, yy)], color=SQUEEZE, lw=3.0, head=10, z=9)
            c.arrow([(X + 112, yy), (X + so + 14, yy)], color=SQUEEZE, lw=3.0, head=10, z=9)
            c.text(X, 84, r"squeeze both tabs  $z_s>\theta_s$", size=11.5)
        if code == "M6":
            c.text(X, 84, r"twist  $|\tau|\geq\tau_b$  breaks the seal", size=11.5)
            burst(c, X + 58, g["y_band1"] - 3, 9, TWIST, z=8)
            c.line([(X + 66, g["y_band1"] + 2), (X + 92, g["y_band1"] + 22)], color=MUTED, lw=0.8, z=8)
            c.text(X + 95, g["y_band1"] + 25, "seal", size=10.5, ha="left")
        for j, s in enumerate(desc):
            c.text(X, 304 + j * 20, s, size=11.5)
    c.text(W / 2 + 117, H - 30,
           r"Section views, not to scale. Redrawn every episode (full thresholds): $\theta_p$ 3-5 mm, "
           r"$\theta_s$ 2-3 mm, $\tau_b$ 0.10-0.25 N m, 1-2 turns.",
           size=11, color=INK2)
    c.text(W / 2 + 117, H - 12, r"Pressing past $\theta_p$ also jams M3 and M6. The arrows mark the input that "
                                r"opens each closure; the layer reads it from the joints.", size=11, color=INK2)
    return c.save("mechanisms", preview_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preview-dir", default=None)
    a = p.parse_args()
    print("method.svg", method_figure(a.preview_dir), "bytes")
    print("mechanisms.svg", mechanism_figure(a.preview_dir), "bytes")


if __name__ == "__main__":
    main()
