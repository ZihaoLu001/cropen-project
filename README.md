# CR-Open project page

Source of the project page for **Push, Squeeze or Turn? Learning to Open Child-Resistant Closures of Hidden Type
with a Dexterous Hand (CR-Open)**. Work in progress; all results are in simulation.

## Page

- `index.html`, `static/css/index.css`, `static/js/index.js`: a lean Nerfies-family page laid out like recent ETH CVG
  project pages (e.g. Elevator-VIGS): title block with a one-line TL;DR, one clip player with a row of closure
  buttons and a play/pause toggle (no native control bar, which would cover the state display), Abstract, the
  closure types, and Method (one figure and one paragraph that cites the figure's red badges 1-4). On 2026-11-18 the
  user had the clip label (seed table, "About these clips") and the preliminary-results section (physics-v1 chart,
  ablation paragraph, "Study details") removed, as none of the surveyed project pages carries such material; results
  are left to the paper (and to a later v2 results update of this page).
- Everything is local (vendored Bulma 1.0.4, Google Sans and Noto Sans, two Font Awesome SVGs; no CDN, no
  analytics), so the page loads where Google and CDN hosts are blocked. Licence notices are in `static/vendor/`; the
  footer keeps the Nerfies credit. Bulma is its build without dark mode (`bulma-no-dark-mode.min.css`), as the page
  and its figures are light-only. To keep the page's look, `static/css/index.css` sets Bulma 1.0's CSS variables
  (neutral greys, Noto Sans text, bold `<strong>`, semibold titles, 4px corners, button weight and padding, hover and
  press borders, focus glow, no colour transitions) and a few plain rules: a fixed button height, no button shadow,
  and 0.9's system font and icon spacing on the two "coming soon" buttons (Google Sans stays on the clip buttons).
- On phones (width up to 600 px) the mechanism figure switches to a 2 x 2 version (`<picture>`;
  `mechanisms_phone.svg`); the method figure keeps a readable width and scrolls
  sideways inside its own box (a corner chip marks it); the page itself never scrolls sideways.
- Changed assets carry a `?v=` query (stylesheet `v9`, script `v8`, Bulma `1.0.4`, figures `v7`, clips `v3c`).

## Clips (`static/videos/`)

Simulation clips of one trained policy: stage-1 seed 4 of the variant with posture regularization (the N1 arm),
physics v2 (255 solver iterations), recipe R2, epoch 10000; seed 4 was chosen after the three stage-1 seeds'
epoch-10000 rates were read (it is the only one of the three that opens all four types). Since 2026-11-18 the page
carries no clip label (the user's request; recorded under 状态 in the dexrl-cropen README as a deviation from the
pre-registered clip label rule); the abstract still states that squeeze-and-turn success varies widely across
seeds. Rendered with `tools/render_demo.py` (presentation media: the type is fixed per clip, the first successful
episode is kept, every attempt is listed in the `*_render_log.json` files). The state display and the marks on the
cap are drawn by `tools/overlay_mech.py` from the per-step state log of the same episode
(`tools/render_logs/m*_state.json`) and the render camera (`m*_render_log.json`); the mechanism colours (orange
tabs, blue marker ridges, teal inner band) are paint only and change no physics. `tools/sheet.py` makes the contact
sheets; `tools/render_logs/render_v3.sh` and `overlay_v3c.sh` are the runs (v3c redraws the v3b overlay from
the same frames and logs with plainer wording; `overlay_v3c_log.jsonl` matches `overlay_v3b_log.jsonl` in every field
except the output name and size).

## Figures

- `static/images/method.svg` (method overview), `mechanisms.svg` (the four closure types as section views) and
  `mechanisms_phone.svg` (the same in a 2 x 2 layout): `python tools/make_figures.py [--preview-dir DIR]`
  (matplotlib; Noto Sans text with STIX math, converted to paths). Labels are short Title-Case nouns as in related
  papers' method figures (groups Isaac Lab, Observation, Policy, Training; a legend of arrow types: Control Loop,
  Physics Step, Training Only); what each box shows follows the training code and the recipe's command line,
  listed in the script's docstring. The script refuses to write a figure in which a text crosses the border of the
  box that holds it. Type colours follow the clip overlay (push teal, squeeze orange, twist violet); pressing jams
  every type except push-and-turn, so "Jams if Pressed" is drawn on M1, M3 and M6. Render thumbnails in
  `tools/fig_assets/`: `hand_on_cap.jpg` is the crop x 500-1280, y 0-520 (above the state display) of
  `static/videos/m1-poster.jpg`, scaled to 480 x 320; `cap_closeup.jpg` is the crop x 432-856, y 232-492 of the
  clean 1280 x 720 v3 frame M1 step 0 (box 1 `/workspace/sitemedia/final_v3/m1_frames`).
- Not on the page since 2026-11-18 (kept for the paper and a later results update): `python tools/make_v1_charts.py
  --data <dir> [--preview-dir DIR]` draws `v1_success_e10000.svg` and `v1_success_e10000_phone.svg` into
  `static/images/` from the recorded physics-v1 analysis outputs (`w5_stats` Table 2 of both arms at epoch 10000;
  `w5_gd` at epoch 6000). `tools/data/v1_numbers.json` holds those numbers and the per-seed values, with the sha256
  of each source file. The chart calls the two arms
  "without / with posture regularization"; both are drawn in one panel (blue / dark grey). The seeds are drawn as a
  beeswarm (x only separates seeds of one column), means are labelled in whole percent, and the text is Noto Sans
  like the page: `tools/fonts/` holds a Latin subset of Noto Sans 2.015 Regular and Bold (SIL Open Font Licence,
  `OFL-Noto-Sans.txt`).

## Sources for the layout and wording

The 2026-11-18 rebuild follows a survey of 31 project pages (7 ETH CVG pages including Elevator-VIGS, other
ETH/UZH robotics pages, and dexterous / Isaac Gym / Isaac Lab pages such as HORA, Twisting Lids Off, DexterityGen,
DeXtreme and DextrAH-G) and of 19 method figures of sim-trained dexterous RL papers: one figure and one paragraph per
section, a one-line TL;DR under the title block, results as videos, statistics left to the paper, the simulator named once ("Isaac Lab"), and short Title-Case figure labels.

Preview locally: `python -m http.server 8765` in this folder, then open http://localhost:8765/.
