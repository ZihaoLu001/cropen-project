"""Mechanism HUD for CR-Open demo clips (presentation media, not evaluation data).

Reads the PNG frames of a clip (render_demo.py --keep-frames; or the clip itself) and the per-step state log of the SAME
episode (render_demo.py --state-log) and writes a clip with a heads-up display drawn from that log: press depth with
the gate (M2) or jam threshold, squeeze travel with the M3 gate threshold, the torque on the shell with the M6 breakaway
torque, the shell rotation next to the engaged thread progress, the lift, the mechanism state in words, and labels at
the frames where the gate unlocks, the thread jams, the seal breaks, the thread comes free and the cap lifts off.
Row i of the log is frame i. Optional slow motion (each frame repeated) around the family's key event and around
extra row windows (--slow-rows).

With --render-log (the clip's render_demo.py log: camera eye, lookat, focal length, resolution) the family's own
input is also marked ON the cap, projected through the render camera from the logged joints, because the travel
is millimetres and the fingertips cover the tabs: M2 a teal arrow pushing down on the cap top (length from the
press depth), M3 an orange arrow pushing in on each tab (length from that tab's travel; faint when the tab is
behind the cap), M6 a violet twist arc in front of the cap while the seal holds (sweep from the torque on the
shell relative to the breakaway torque), fading out after the seal breaks. A mark is drawn only above 0.5 mm.

  full    : 1280x720, a band at the bottom (state, five strip charts, events), event banner just above it
  compact : a small frame (e.g. 640x360, a teaser quadrant): title, state and the family's key gauge only

Usage: python overlay_mech.py --frames m2_frames --state m2_state.json --out m2.mp4 --poster m2-poster.jpg
       [--render-log m2_render_log.json] [--mode full|compact] [--size 1280 720] [--slowmo 3]
       [--slow-window 30 20] [--slow-rows 260 280] [--poster-row 24] [--tail 1.5] [--crf 23]
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FAMILY = {
    "M1": ("plain thread", "Turn to unscrew, then lift. Pressing jams the thread."),
    "M2": ("push-and-turn", "Push the cap down while turning it."),
    "M3": ("squeeze-and-turn", "Squeeze the two side tabs while turning."),
    "M6": ("stuck seal", "Twist hard to break the seal, then unscrew."),
}
KEY_ROW = {"M1": "turn", "M2": "press", "M3": "squeeze", "M6": "torque"}
KEY_EVENT = {"M1": None, "M2": "unlock", "M3": "unlock", "M6": "seal_broken"}
# row colours match the scene's mechanism colours (render_demo.py --mech-colors): teal band, orange tabs, blue marker
C = dict(press=(0, 196, 170), squeeze=(255, 120, 30), torque=(190, 130, 255), turn=(90, 160, 255),
         lift=(140, 215, 100), text=(238, 240, 244), muted=(160, 168, 180), ok=(70, 210, 130), bad=(245, 80, 80),
         warn=(255, 190, 60), panel=(14, 16, 20))
EV = dict(unlock=("GATE UNLOCKED", C["ok"]), jam=("THREAD JAMMED", C["bad"]), seal_broken=("SEAL BROKEN", C["torque"]),
          thread_free=("THREAD FREE", C["turn"]), lifted_off=("LIFTED OFF", C["ok"]),
          reengage=("GATE UNLOCKED AGAIN", C["ok"]))
EV_ORDER = ("jam", "unlock", "seal_broken", "thread_free", "lifted_off")
GATED_EV = ("unlock", "reengage")   # banners that end when the gate locks again
DARK = (16, 18, 22)
# closure geometry, read off cropen/closure_urdf.py (ClosureGeom) and cropen/cap_geom.py at phys2 9b24195 (not
# imported: the overlay must not touch the frozen package). Env frame, cap axis = z: the closure base sits at
# z = 0.30; the shell bottom (press carrier) is at 0.30 + (neck 0.030 - 0.006) + press range 0.006 + lift - press;
# the shell is 22 mm high, radius 25 mm; tab_r points along the shell angle (joint rot), tab_l opposite; a tab's
# centre is 28 mm from the axis at mid-height minus its squeeze travel, its outer face 3 mm further out.
Z_CARRIER0, SHELL_H, R_TAB, TAB_HALF_R = 0.330, 0.022, 0.028, 0.003
APERTURE_MM = 20.955   # render_demo.py PinholeCameraCfg horizontal aperture (square pixels)
MARK_MIN = 0.0005      # draw a mark only above 0.5 mm of travel
FONTS = {"regular": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/DejaVuSans.ttf",
                     "C:/Windows/Fonts/segoeui.ttf"],
         "bold": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/DejaVuSans-Bold.ttf",
                  "C:/Windows/Fonts/segoeuib.ttf"]}
SS = 2   # supersampling of the HUD layers


def font(kind, px):
    for f in FONTS[kind]:
        if Path(f).is_file():
            return ImageFont.truetype(f, int(round(px * SS)))
    return ImageFont.load_default(size=int(round(px * SS)))


class Layer:
    """an RGBA layer drawn at SS x and pasted (downsampled) onto the frame at (x, y) in frame pixels"""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.im = Image.new("RGBA", (int(w * SS), int(h * SS)), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def s(self, *v):
        return [x * SS for x in v]

    def rect(self, x0, y0, x1, y1, fill, r=0, outline=None, width=1):
        self.d.rounded_rectangle(self.s(x0, y0, x1, y1), radius=r * SS, fill=fill, outline=outline, width=width * SS)

    def text(self, x, y, t, f, fill, anchor="la"):
        self.d.text((x * SS, y * SS), t, font=f, fill=fill, anchor=anchor)

    def tlen(self, t, f):
        return self.d.textlength(t, font=f) / SS

    def line(self, pts, fill, width=1):
        self.d.line([(x * SS, y * SS) for x, y in pts], fill=fill, width=max(1, int(round(width * SS))), joint="curve")

    def dashed(self, x0, x1, y, fill, dash=5, gap=4, width=1):
        x = x0
        while x < x1:
            self.line([(x, y), (min(x + dash, x1), y)], fill, width)
            x += dash + gap

    def paste_on(self, frame, x, y, alpha=1.0):
        small = self.im.resize((self.w, self.h), Image.LANCZOS)
        if alpha < 1.0:
            a = np.asarray(small).astype(np.float32)
            a[..., 3] *= alpha
            small = Image.fromarray(a.astype(np.uint8), "RGBA")
        frame.alpha_composite(small, (int(x), int(y)))


def rgba(c, a=255):
    return (c[0], c[1], c[2], int(a))


def pos(v):
    """v clipped at 0 without a negative zero ("-0.0 mm" from solver noise at a joint limit)"""
    v = float(v)
    return v if v > 0 else 0.0


def mix(c, a, base=None):
    """opaque colour: c at opacity a/255 over the panel colour (PIL replaces, not blends, shape pixels on RGBA)"""
    base = C["panel"] if base is None else base
    t = a / 255.0
    return tuple(int(round(base[j] * (1 - t) + c[j] * t)) for j in range(3)) + (255,)


def load_state(path):
    st = json.loads(Path(path).read_text())
    rows, p = st["rows"], st["params"]
    fam = st["family"]
    ser = dict(
        press=np.array([r["press"] for r in rows]) * 1000,
        squeeze=np.array([0.5 * (r["squeeze_l"] + r["squeeze_r"]) for r in rows]) * 1000,
        torque=np.array([r["tau_shell"] for r in rows]),
        shell=np.degrees([r["th_u"] for r in rows]),
        thread=np.degrees([r["th_in"] for r in rows]),
        lift=np.array([r["lift"] for r in rows]) * 1000,
    )
    first = lambda k: next((i for i, r in enumerate(rows) if r[k]), None)
    ev = dict(unlock=first("ev_unlock"), jam=first("ev_jam"), seal_broken=first("ev_break"),
              thread_free=first("ev_release"), lifted_off=first("success"))
    return st, fam, rows, p, ser, ev


def status(fam, r):
    gated = fam in ("M2", "M3")
    if r["success"]:
        return "opened: cap lifted off", C["ok"]
    if r["released"]:
        return "thread free: lifting", C["ok"]
    if fam == "M6" and not r["broke"]:
        return "seal holding: shell stuck", C["torque"]
    if r["jam_now"]:
        return "jammed: must not be pressed", C["bad"]
    if gated and not r["engaged"]:
        if r["ev_unlock"]:                           # the gate has opened before and locked again
            return ("released: push again to engage" if fam == "M2" else "released: squeeze again to engage"), C["warn"]
        return ("locked: push down to engage" if fam == "M2" else "locked: squeeze tabs to engage"), C["warn"]
    return "engaged: turning unscrews", C["turn"]


def event_text(fam, k, i, ser, p):
    if k in GATED_EV and fam == "M2":
        return f"pushed {ser['press'][i]:.1f} mm (gate {1000 * p['th_press']:.1f} mm): turning now unscrews"
    if k in GATED_EV:
        return f"tabs squeezed {ser['squeeze'][i]:.1f} mm (gate {1000 * p['th_squeeze']:.1f} mm): turning now unscrews"
    if k == "jam":
        return f"pressed {ser['press'][i]:.1f} mm on a cap that must not be pressed"
    if k == "seal_broken":
        return f"twisted past the breakaway torque ({p['tau_break']:.2f} N m)"
    if k == "thread_free":
        return f"unscrewed {p['turns']:.2f} turns"
    return "lifted 1 cm: the closure is open"


def timeline(n, ev, fam, slowmo, window, tail_frames, extra=()):
    """output frames: list of (row index, slow?); slow around the key event and in the extra (lo, hi) row windows"""
    key = KEY_EVENT.get(fam)
    wins = []
    if slowmo > 1 and key and ev.get(key) is not None:
        wins.append((max(0, ev[key] - window[0]), min(n - 1, ev[key] + window[1])))
    if slowmo > 1:
        wins += [(max(0, lo), min(n - 1, hi)) for lo, hi in extra]
    out = []
    for i in range(n):
        slow = any(lo <= i <= hi for lo, hi in wins)
        out += [(i, slow)] * (slowmo if slow else 1)
    out += [(n - 1, False)] * tail_frames
    return out


def banner_events(fam, rows, ev, min_rows):
    """(event, row) for the banners: the logged first events, plus, for M2 / M3, every later engagement of the gate
    that holds for at least min_rows rows (from the logged engaged flag)"""
    out = [(k, ev[k]) for k in EV_ORDER if ev.get(k) is not None]
    if fam in ("M2", "M3") and ev.get("unlock") is not None:
        eng = [bool(r["engaged"]) for r in rows]
        j = ev["unlock"] + 1
        while j < len(rows):
            if eng[j] and not eng[j - 1]:
                k = j
                while k < len(rows) and eng[k]:
                    k += 1
                if k - j >= min_rows:
                    out.append(("reengage", j))
                j = k
            else:
                j += 1
    return out


class Cam:
    """pinhole projection of the render camera (render_demo.py: set_world_poses_from_view(eye, lookat), z up,
    PinholeCameraCfg with horizontal aperture 20.955 mm, square pixels) into an output frame of size out_wh"""

    def __init__(self, eye, lookat, focal, res, out_wh):
        self.E = np.asarray(eye, float)
        f = np.asarray(lookat, float) - self.E
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, (0.0, 0.0, 1.0))
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)
        self.fx = focal / APERTURE_MM * res[0] * (out_wh[0] / res[0])
        self.c = (out_wh[0] / 2.0, out_wh[1] / 2.0)
        self.az_front = math.atan2(self.E[1], self.E[0])   # the side of the cap that faces the camera

    def __call__(self, p):
        d = np.asarray(p, float) - self.E
        z = d @ self.f
        return np.array([self.c[0] + self.fx * (d @ self.r) / z, self.c[1] - self.fx * (d @ self.u) / z])

    def facing(self, p, n):
        """cosine between the outward normal n at p and the direction to the camera"""
        v = self.E - np.asarray(p, float)
        return float(np.asarray(n, float) @ v / np.linalg.norm(v))

    @staticmethod
    def from_log(path, out_wh):
        lg = json.loads(Path(path).read_text())
        return Cam(lg["eye"], lg["lookat"], float(lg.get("focal", 18.0)), lg.get("res", (1280, 720)), out_wh)


class Marks:
    """vector marks in frame pixels, drawn anti-aliased on a layer that only covers their bounding box"""

    def __init__(self):
        self.ops = []

    def poly(self, pts, fill, outline, width):
        self.ops.append(("poly", [tuple(map(float, q)) for q in pts], fill, outline, float(width)))

    def path(self, pts, fill, width, outline, owidth):
        self.ops.append(("path", [tuple(map(float, q)) for q in pts], fill, float(width), outline, float(owidth)))

    def text(self, xy, t, fnt, fill, stroke_px=2.5):
        self.ops.append(("text", tuple(map(float, xy)), t, fnt, fill, stroke_px))

    def paste_on(self, frame):
        if not self.ops:
            return
        xs, ys = [], []
        for op in self.ops:
            if op[0] in ("poly", "path"):
                w = op[4] if op[0] == "poly" else 0.5 * op[3] + op[5]
                xs += [q[0] - w for q in op[1]] + [q[0] + w for q in op[1]]
                ys += [q[1] - w for q in op[1]] + [q[1] + w for q in op[1]]
            else:
                x0, y0, x1, y1 = op[3].getbbox(op[2], anchor="mm", stroke_width=int(round(op[5] * SS)))
                xs += [op[1][0] + x0 / SS, op[1][0] + x1 / SS]
                ys += [op[1][1] + y0 / SS, op[1][1] + y1 / SS]
        X0, Y0 = max(0, int(min(xs)) - 4), max(0, int(min(ys)) - 4)
        X1, Y1 = min(frame.width, int(max(xs)) + 5), min(frame.height, int(max(ys)) + 5)
        if X1 <= X0 or Y1 <= Y0:
            return
        L = Layer(X1 - X0, Y1 - Y0)
        sh = lambda pts: [((x - X0) * SS, (y - Y0) * SS) for x, y in pts]
        for op in self.ops:
            if op[0] == "poly":
                _, pts, fill, outline, w = op
                L.d.line(sh(pts + pts[:1]), fill=outline, width=max(1, int(round(2 * w * SS))), joint="curve")
                L.d.polygon(sh(pts), fill=fill)
            elif op[0] == "path":
                _, pts, fill, w, outline, ow = op
                L.d.line(sh(pts), fill=outline, width=max(1, int(round((w + 2 * ow) * SS))), joint="curve")
                L.d.line(sh(pts), fill=fill, width=max(1, int(round(w * SS))), joint="curve")
            else:
                _, (x, y), t, fnt, fill, st = op
                L.d.text(((x - X0) * SS, (y - Y0) * SS), t, font=fnt, fill=fill, anchor="mm",
                         stroke_width=int(round(st * SS)), stroke_fill=rgba(DARK, fill[3]))
        L.paste_on(frame, X0, Y0)


def arrow(m, tip, d, length, col, a=1.0, k=1.0):
    """a filled arrow in frame pixels whose tip is at tip, pointing along d"""
    d = np.asarray(d, float) / (np.linalg.norm(d) + 1e-9)
    n = np.array([-d[1], d[0]])
    w1, w2, hl = 5.5 * k, 13.0 * k, min(18.0 * k, 0.6 * length)
    tip = np.asarray(tip, float)
    neck = tip - d * hl
    tail = tip - d * length
    m.poly([tail + n * w1, neck + n * w1, neck + n * w2, tip, neck - n * w2, neck - n * w1, tail - n * w1],
           rgba(col, 255 * a), rgba(DARK, 235 * a), 1.6 * k)


def label_at(m, anchor, d, gap, t, fnt, col, a=1.0):
    """text centred beyond anchor along -d (the tail side of an arrow), gap px past the anchor"""
    d = np.asarray(d, float) / (np.linalg.norm(d) + 1e-9)
    x0, y0, x1, y1 = fnt.getbbox(t, anchor="mm")
    half = 0.5 * (abs(d[0]) * (x1 - x0) + abs(d[1]) * (y1 - y0)) / SS
    m.text(np.asarray(anchor, float) - d * (gap + half), t, fnt, rgba(col, 255 * a))


def cap_marks(m, fam, rows, i, p, ev, cam, fnt, k=1.0, labels=True):
    """the family's own input marked on the cap at row i (see the module docstring)"""
    r = rows[i]
    zb = Z_CARRIER0 + r["lift"] - r["press"]        # shell bottom
    if fam == "M2" and r["press"] >= MARK_MIN:
        top = np.array([0.0, 0.0, zb + SHELL_H])
        P = cam(top)
        d = P - cam(top + np.array([0.0, 0.0, 0.01]))   # screen direction of -z
        mm_ = 1000 * r["press"]
        L = (22 + 13 * mm_) * k
        tip = P - d / np.linalg.norm(d) * 2 * k
        arrow(m, tip, d, L, C["press"], 1.0, k)
        if labels:
            label_at(m, tip - d / np.linalg.norm(d) * L, d, 6 * k, f"push {mm_:.1f} mm", fnt, C["press"])
    if fam == "M3":
        for side in ("l", "r"):
            s = r["squeeze_" + side]
            if s < MARK_MIN:
                continue
            th = r["rot_raw"] + (math.pi if side == "l" else 0.0)
            rad = np.array([math.cos(th), math.sin(th), 0.0])
            face = (R_TAB + TAB_HALF_R - s) * rad + np.array([0.0, 0.0, zb + SHELL_H / 2])
            P = cam(face)
            d = cam(face - 0.004 * rad) - P             # screen direction of "inward"
            if np.linalg.norm(d) < 0.3 * 0.004 * cam.fx / ((face - cam.E) @ cam.f):
                d = np.array([cam(np.array([0.0, 0.0, face[2]]))[0] - P[0], 0.0])   # tab faces the camera: sideways
                if abs(d[0]) < 1e-6:
                    d = np.array([1.0, 0.0])
            a = 1.0 if cam.facing(face, rad) > -0.25 else 0.5   # faint when the tab is behind the cap
            mm_ = 1000 * s
            L = (22 + 15 * mm_) * k
            tip = P - d / np.linalg.norm(d) * 2 * k
            arrow(m, tip, d, L, C["squeeze"], a, k)
            if labels:
                label_at(m, tip - d / np.linalg.norm(d) * L, d, 5 * k, f"{mm_:.1f} mm", fnt, C["squeeze"], a)
    if fam == "M6":
        jb = ev.get("seal_broken")
        fade = 10                                      # rows over which the arc fades after the break
        if not r["broke"]:
            frac, a, txt = min(float(r["tau_shell"]) / p["tau_break"], 1.0), 1.0, f"twist {r['tau_shell']:.2f} N m"
            if r["tau_shell"] < 0.01:
                return
        elif jb is not None and i - jb < fade:
            frac, a, txt = 1.0, 1.0 - (i - jb) / fade, "seal broken"
        else:
            return
        zt = zb + SHELL_H + 0.003
        a0 = cam.az_front - math.radians(45)
        sweep = math.radians(90) * frac
        ang = np.linspace(a0, a0 + sweep, max(3, int(math.degrees(sweep) / 2)))
        pts = [cam(np.array([0.029 * math.cos(t), 0.029 * math.sin(t), zt])) for t in ang]
        col = C["torque"]
        m.path(pts, rgba(col, 255 * a), 8 * k, rgba(DARK, 235 * a), 1.6 * k)
        d = pts[-1] - pts[-2]                          # arrow head along the +th tangent (unscrewing direction)
        d = d / (np.linalg.norm(d) + 1e-9)
        n = np.array([-d[1], d[0]])
        e = pts[-1] - d * 2 * k
        m.poly([e + n * 13 * k, e + d * 19 * k, e - n * 13 * k], rgba(col, 255 * a), rgba(DARK, 235 * a), 1.6 * k)
        if labels:
            label_at(m, pts[0], pts[1] - pts[0], 8 * k, txt, fnt, col, a)


class FullHUD:
    """bottom band (the top of the frame is left to the hand): title, task, state and events on the left, five strip
    charts over the whole episode on the right; event banners just above the band"""
    BH, M = 168, 16

    def __init__(self, W, H, st, fam, rows, p, ser, ev):
        self.W, self.H, self.st, self.fam, self.rows, self.p, self.ser, self.ev = W, H, st, fam, rows, p, ser, ev
        self.n = len(rows)
        self.f = dict(name=font("bold", 22), task=font("regular", 14), lab=font("bold", 12.5), val=font("bold", 13.5),
                      small=font("regular", 11), stat=font("bold", 15), ev=font("regular", 12), big=font("bold", 25),
                      evsub=font("regular", 15), badge=font("bold", 20), mark=font("bold", 15))
        th_p, th_s = 1000 * p["th_press"], 1000 * p["th_squeeze"]
        tmax = max(0.3, 1.15 * float(ser["torque"].max()), 1.25 * p["tau_break"] if fam == "M6" else 0)
        free_deg = 360 * p["turns"]
        dmax = max(1.1 * free_deg, 1.05 * float(max(ser["shell"].max(), ser["thread"].max())), 90)
        lift_off = 1000 * (p["pitch"] * p["turns"] + p["lift_success"])
        lmax = max(1.15 * lift_off, 1.05 * float(ser["lift"].max()), 5)
        hy = 1000 * p["hyst"]                           # a gate engaged above th stays engaged down to th - hyst
        self.charts = [
            dict(k="press", lab="PRESS", hi=6.0, thr=th_p, rel=th_p - hy if fam == "M2" else None,
                 thr_lab=f"gate {th_p:.1f} · off {th_p - hy:.1f} mm" if fam == "M2" else f"jams {th_p:.1f} mm"),
            dict(k="squeeze", lab="SQUEEZE", hi=4.0, thr=th_s if fam == "M3" else None,
                 rel=th_s - hy if fam == "M3" else None, thr_lab=f"gate {th_s:.1f} · off {th_s - hy:.1f} mm",
                 note=None if fam == "M3" else "no function here"),
            dict(k="torque", lab="TORQUE", hi=tmax, thr=p["tau_break"] if fam == "M6" else None,
                 thr_lab=f"breakaway {p['tau_break']:.2f}"),
            dict(k="turn", lab="TURN", hi=dmax, thr=free_deg, thr_lab=f"free at {p['turns']:.2f} turns"),
            dict(k="lift", lab="LIFT", hi=lmax, thr=lift_off, thr_lab="lift-off"),
        ]
        self.band = self._band_static()

    def _band_static(self):
        """the parts of the band that never change: background, badge, family name, task sentence"""
        W, BH, M, f = self.W, self.BH, self.M, self.f
        name, task = FAMILY[self.fam]
        L = Layer(W - 2 * M, BH)
        L.rect(0, 0, W - 2 * M, BH, rgba(C["panel"], 218), r=14)
        L.rect(14, 14, 60, 60, rgba((255, 255, 255), 235), r=9)
        L.text(37, 37, self.fam, f["badge"], rgba((20, 22, 26)), anchor="mm")
        L.text(72, 13, name, f["name"], rgba(C["text"]))
        L.text(72, 41, "closure family (hidden from the policy)", f["small"], rgba(C["muted"]))
        y = 70
        for line in wrap(task, f["task"], 300):
            L.text(14, y, line, f["task"], rgba(C["text"]))
            y += 19
        return L

    def value(self, k, i):
        s = self.ser
        return {"press": f"{pos(s['press'][i]):.1f} mm", "squeeze": f"{pos(s['squeeze'][i]):.1f} mm",
                "torque": f"{pos(s['torque'][i]):.2f} N m", "lift": f"{pos(s['lift'][i]):.1f} mm",
                "turn": f"{s['shell'][i]:.0f}° / {s['thread'][i]:.0f}°"}[k]

    def draw(self, frame, i, slow, toast):
        W, H, BH, M, f = self.W, self.H, self.BH, self.M, self.f
        if getattr(self, "cam", None) is not None:
            mk = Marks()
            cap_marks(mk, self.fam, self.rows, i, self.p, self.ev, self.cam, f["mark"])
            mk.paste_on(frame)
        L = Layer(W - 2 * M, BH)
        L.im.paste(self.band.im)
        txt, col = status(self.fam, self.rows[i])
        y = 112
        L.d.ellipse(L.s(15, y + 4, 27, y + 16), fill=rgba(col))
        L.text(34, y, txt, f["stat"], rgba(col))
        L.text(14, BH - 14, f"t = {(i + 1) / 30:.2f} s", f["val"], rgba(C["text"]), anchor="ld")
        chip = f"{1 / self.slowmo:.2g}× slow motion" if slow else "real time"
        L.text(112, BH - 15, chip, f["small"], rgba(C["warn"] if slow else C["muted"]), anchor="ld")
        done = [(k, self.ev[k]) for k in EV_ORDER if self.ev.get(k) is not None and self.ev[k] <= i]
        # charts
        x_start, gap = 352, 12
        cw = (W - 2 * M - x_start - 14 - gap * (len(self.charts) - 1)) / len(self.charts)
        top, ch = 14, BH - 14 - 22 - 28
        n = self.n
        for c_i, cs in enumerate(self.charts):
            k, colr = cs["k"], C[cs["k"]]
            x0 = x_start + c_i * (cw + gap)
            x1 = x0 + cw
            key = KEY_ROW[self.fam] == k
            L.text(x0, top, cs["lab"], f["lab"], rgba(colr))
            L.text(x1, top, self.value(k, i), f["val"], rgba(C["text"]), anchor="ra")
            y0, y1 = top + 22, top + 22 + ch
            L.rect(x0, y0, x1, y1, mix((255, 255, 255), 22 if key else 12), r=5,
                   outline=mix(colr, 230) if key else None, width=2 if key else 1)
            xs = lambda j: x0 + 4 + (cw - 8) * (j / max(n - 1, 1))
            yv = lambda v: y1 - 5 - (ch - 10) * min(max(v, 0.0), cs["hi"]) / cs["hi"]
            for kk in EV_ORDER:
                j = self.ev.get(kk)
                if j is not None and j <= i:
                    L.line([(xs(j), y0 + 3), (xs(j), y1 - 3)], mix(EV[kk][1], 150, mix((255, 255, 255), 12)), 1.2)
            if cs.get("thr") is not None:
                L.dashed(x0 + 4, x1 - 4, yv(cs["thr"]), mix((255, 255, 255), 190), width=1.2)
            if cs.get("rel") is not None:                # release line of the gate's hysteresis, fainter
                L.dashed(x0 + 4, x1 - 4, yv(cs["rel"]), mix((255, 255, 255), 85, mix((255, 255, 255), 22)),
                         dash=3, gap=4, width=1.0)
            if cs.get("note"):
                L.text((x0 + x1) / 2, (y0 + y1) / 2, cs["note"], f["small"], rgba(C["muted"]), anchor="mm")
            series = [("shell", 1.5, 130), ("thread", 2.6, 255)] if k == "turn" else [(k, 2.4, 255)]
            for sk, wdt, alpha in series:
                v = self.ser[sk][: i + 1]
                pts = [(xs(j), yv(float(v[j]))) for j in range(len(v))]
                if len(pts) == 1:
                    pts = pts * 2
                L.line(pts, mix(colr, alpha), wdt)
                L.d.ellipse(L.s(pts[-1][0] - 3.5, pts[-1][1] - 3.5, pts[-1][0] + 3.5, pts[-1][1] + 3.5),
                            fill=mix(colr, alpha))
            bg = mix((255, 255, 255), 22 if key else 12)
            if cs.get("thr") is not None:                # threshold label on a chip, above the traces
                ty = yv(cs["thr"])
                above = ty - y0 > 20
                tw = L.tlen(cs["thr_lab"], f["small"])
                cy0 = ty - 17 if above else ty + 3
                L.rect(x0 + 5, cy0, x0 + 13 + tw, cy0 + 14, bg, r=3)
                L.text(x0 + 9, cy0 + 1, cs["thr_lab"], f["small"], rgba((235, 235, 235)))
            if k == "turn":
                lg = "thin: shell · thick: thread"
                tw = L.tlen(lg, f["small"])
                L.rect(x0 + 5, y0 + 4, x0 + 13 + tw, y0 + 18, bg, r=3)
                L.text(x0 + 9, y0 + 5, lg, f["small"], rgba(C["muted"]))
        # event list under the charts
        ex = x_start
        for kk, j in done:
            t = f"✓ {EV[kk][0].lower()} {(j + 1) / 30:.2f} s"
            L.text(ex, BH - 9, t, f["ev"], rgba(EV[kk][1]), anchor="ld")
            ex += L.tlen(t, f["ev"]) + 18
        L.paste_on(frame, M, H - M - BH)
        if toast:
            kk, j, alpha = toast
            head, colr = EV[kk]
            sub = event_text(self.fam, kk, j, self.ser, self.p)
            tmp = Layer(10, 10)
            w = 30 + max(tmp.tlen(head, f["big"]), tmp.tlen(sub, f["evsub"])) + 24
            T = Layer(int(w), 78)
            T.rect(0, 0, w, 78, rgba(C["panel"], 210), r=12)
            T.rect(0, 0, 9, 78, rgba(colr), r=4)
            T.text(26, 10, head, f["big"], rgba(colr))
            T.text(26, 48, sub, f["evsub"], rgba(C["text"]))
            T.paste_on(frame, M, H - M - BH - 12 - 78, alpha)


def wrap(text, fnt, width):
    tmp = Layer(10, 10)
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if tmp.tlen(t, fnt) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur] if cur else lines


class CompactHUD:
    """a teaser quadrant: one band at the bottom with the family, its state (or the latest event) and its key gauge"""

    def __init__(self, W, H, st, fam, rows, p, ser, ev):
        self.W, self.H, self.fam, self.rows, self.p, self.ser, self.ev = W, H, fam, rows, p, ser, ev
        k = W / 640
        self.k = k
        self.f = dict(t=font("bold", 15 * k), stat=font("bold", 11.5 * k), lab=font("bold", 11 * k),
                      small=font("regular", 10 * k))

    def draw(self, frame, i, slow, toast):
        k, f, fam, p, s = self.k, self.f, self.fam, self.p, self.ser
        if getattr(self, "cam", None) is not None:      # the same marks, smaller and without numbers
            mk = Marks()
            cap_marks(mk, fam, self.rows, i, p, self.ev, self.cam, None, k=max(0.7, k * 0.5 * 1.4), labels=False)
            mk.paste_on(frame)
        txt, col = status(fam, self.rows[i])
        if toast:
            txt, col = EV[toast[0]][0].lower(), EV[toast[0]][1]
        M, bh = 8 * k, 48 * k
        L = Layer(int(self.W - 2 * M), int(bh))
        L.rect(0, 0, self.W - 2 * M, bh, rgba(C["panel"], 205), r=8 * k)
        L.text(10 * k, 7 * k, f"{fam}  {FAMILY[fam][0]}", f["t"], rgba(C["text"]))
        L.d.ellipse(L.s(10 * k, 30 * k, 19 * k, 39 * k), fill=rgba(col))
        L.text(25 * k, 28 * k, txt, f["stat"], rgba(col))
        key = KEY_ROW[fam]
        if key == "press":
            lab, v, hi, thr = "PRESS", pos(s["press"][i]), 6.0, 1000 * p["th_press"]
            vt = f"{v:.1f} mm"
        elif key == "squeeze":
            lab, v, hi, thr = "SQUEEZE", pos(s["squeeze"][i]), 4.0, 1000 * p["th_squeeze"]
            vt = f"{v:.1f} mm"
        elif key == "torque":
            lab, v, hi, thr = "TWIST", pos(s["torque"][i]), 1.25 * p["tau_break"], p["tau_break"]
            vt = f"{v:.2f} N m"
        else:
            lab, v, hi, thr = "THREAD", pos(s["thread"][i]), 360 * p["turns"], None
            vt = f"{v / 360:.2f} / {p['turns']:.2f} turns"
        if key == "torque" and self.rows[i]["broke"]:
            vt = "seal broken"
        gx0, gx1 = L.w - 236 * k, L.w - 10 * k
        L.text(gx0, 7 * k, lab, f["lab"], rgba(C[key]))
        L.text(gx1, 7 * k, vt, f["lab"], rgba(C["text"]), anchor="ra")
        by0, by1 = 26 * k, 37 * k
        L.rect(gx0, by0, gx1, by1, mix((255, 255, 255), 40), r=3 * k)
        frac = min(max(v / hi, 0.0), 1.0)
        if frac > 0.01:
            L.rect(gx0, by0, gx0 + (gx1 - gx0) * frac, by1, rgba(C[key]), r=3 * k)
        if thr is not None:
            tx = gx0 + (gx1 - gx0) * min(thr / hi, 1.0)
            L.line([(tx, by0 - 3 * k), (tx, by1 + 3 * k)], rgba((255, 255, 255)), 1.6 * k)
        L.paste_on(frame, M, self.H - M - bh)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, default=None, help="PNG frames f_00000.png ... (render_demo --keep-frames)")
    ap.add_argument("--clip", type=Path, default=None, help="or read the frames from this clip")
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--poster", type=Path, default=None)
    ap.add_argument("--mode", choices=("full", "compact"), default="full")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("--slowmo", type=int, default=1, help="repeat each frame this often around the key event")
    ap.add_argument("--slow-window", type=int, nargs=2, default=(30, 20), help="frames before / after the key event")
    ap.add_argument("--slow-rows", type=int, nargs=2, action="append", default=[], metavar=("LO", "HI"),
                    help="also slow these rows (inclusive; repeatable)")
    ap.add_argument("--tail", type=float, default=1.5, help="hold the last frame this long (s)")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--toast", type=float, default=1.6, help="event banner duration (s of output)")
    ap.add_argument("--reengage-min", type=int, default=30,
                    help="M2 / M3: a later engagement of the gate gets a banner if it holds this many rows")
    ap.add_argument("--render-log", type=Path, default=None,
                    help="render_demo.py log of the clip (camera); enables the marks on the cap")
    ap.add_argument("--poster-row", type=int, default=None, help="poster = the first output frame of this row")
    ap.add_argument("--dump", type=int, nargs="*", default=None, help="only write these rows as PNGs (layout check)")
    a = ap.parse_args()
    st, fam, rows, p, ser, ev = load_state(a.state)
    n = len(rows)
    if a.frames is not None:
        files = sorted(a.frames.glob("f_*.png"))
        get = lambda i: cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2RGB)
        nf = len(files)
    else:
        cap = cv2.VideoCapture(str(a.clip))
        frames = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        get = lambda i: frames[i]
        nf = len(frames)
    if nf < n:
        raise SystemExit(f"{nf} frames for {n} state rows")
    W, H = a.size
    hud = (FullHUD if a.mode == "full" else CompactHUD)(W, H, st, fam, rows, p, ser, ev)
    hud.slowmo = a.slowmo
    hud.cam = Cam.from_log(a.render_log, (W, H)) if a.render_log is not None else None
    tl = timeline(n, ev, fam, a.slowmo, a.slow_window, int(round(a.tail * 30)), a.slow_rows)
    first_o = {}                                      # row -> its first output frame
    for o, (i, _) in enumerate(tl):
        first_o.setdefault(i, o)
    ev_out = {kk: first_o[ev[kk]] for kk in EV_ORDER if ev.get(kk) is not None}
    toast_len = int(round(a.toast * 30))
    banners = []                                      # (event, row, first output frame, end output frame)
    for kk, j in banner_events(fam, rows, ev, a.reengage_min):
        o0 = first_o[j]
        end = o0 + toast_len
        if kk in GATED_EV:                            # ends when the gate locks again
            end = min(end, next((o for o in range(o0, len(tl)) if not rows[tl[o][0]]["engaged"]
                                 and not rows[tl[o][0]]["jam_now"]), len(tl)))
        banners.append((kk, j, o0, end))

    def toast_at(o):
        best = None
        for b in banners:
            if b[2] <= o < b[3] and (best is None or b[2] >= best[2]):
                best = b
        if best is None:
            return None
        kk, j, o0, end = best
        alpha = min(1.0, (o - o0 + 1) / 5, (end - o) / 6)
        return kk, j, alpha

    def render(o):
        i, slow = tl[o]
        im = get(i)
        if im.shape[1] != W or im.shape[0] != H:
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        fr = Image.fromarray(im).convert("RGBA")
        hud.draw(fr, i, slow, toast_at(o))
        return np.asarray(fr.convert("RGB"))

    if a.dump is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        for i in a.dump:
            o = next(o for o, (j, _) in enumerate(tl) if j == i)
            cv2.imwrite(str(a.out.with_name(f"{a.out.stem}_row{i:03d}.png")), cv2.cvtColor(render(o), cv2.COLOR_RGB2BGR))
        return
    key = KEY_EVENT.get(fam)
    pk = ev.get(key) if key else None
    if a.poster_row is not None:
        po = first_o[min(max(a.poster_row, 0), n - 1)]
    else:
        po = (ev_out.get(key) + 6) if pk is not None else int(0.6 * (n - 1))   # the key event with its banner
    po = min(po, len(tl) - 1)
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(a.out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    poster = None
    for o in range(len(tl)):
        img = render(o)
        if o == po:
            poster = img
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    if a.poster is not None and poster is not None:
        cv2.imwrite(str(a.poster), cv2.cvtColor(poster, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
    info = dict(out=str(a.out), frames=len(tl), seconds=round(len(tl) / 30, 2), rows=n, events_row=ev,
                events_out_frame=ev_out, banners=[list(b) for b in banners], slowmo=a.slowmo,
                slow_rows=[list(w) for w in a.slow_rows], poster_frame=po, poster_row=tl[po][0],
                marks=hud.cam is not None, bytes=a.out.stat().st_size)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
