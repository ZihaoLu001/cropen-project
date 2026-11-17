# CR-Open project page

Source of the project page for **Push, Squeeze or Turn? Learning to Open Child-Resistant Closures of Hidden Type
with a Dexterous Hand (CR-Open)**. Work in progress; all results are in simulation.

## Page

- `index.html`, `static/css/index.css`, `static/js/index.js`: a lean Nerfies-family page laid out like recent ETH CVG
  project pages (e.g. Elevator-VIGS): title block, one clip player with a closure selector, a TL;DR callout,
  Abstract, the closure types, Method (one figure and one paragraph that cites the figure's red badges 1-4) and
  preliminary results (two charts with a one-line claim each). Optional detail sits in two collapsed blocks
  ("About these clips", "Study details and per-seed numbers").
- Everything is local (vendored Bulma, Google Sans and Noto Sans, two Font Awesome SVGs; no CDN, no analytics), so
  the page loads where Google and CDN hosts are blocked. Licence notices are in `static/vendor/`; the footer keeps the
  Nerfies credit.
- On phones (width up to 600 px) the figures and charts keep a readable width and scroll sideways inside their own
  box; the page itself never scrolls sideways.
- Changed assets carry a `?v=` query (page assets `v6`, clips `v3b`).

## Clips (`static/videos/`)

Simulation clips of one trained policy: stage-1 seed 4 of the variant with posture regularization (the N1 arm),
physics v2 (255 solver iterations), recipe R2, epoch 10000; seed 4 was chosen after the three stage-1 seeds'
epoch-10000 rates were read. The label under the player states this, with all three seeds' per-type rates (the
pre-registered clip label rule); one label covers the teaser and the four per-type clips because they share one
player. Rendered with `tools/render_demo.py` (presentation media: the type is fixed per clip, the first successful
episode is kept, every attempt is listed in the `*_render_log.json` files). The state display and the marks on the
cap are drawn by `tools/overlay_mech.py` from the per-step state log of the same episode
(`tools/render_logs/m*_state.json`) and the render camera (`m*_render_log.json`); the mechanism colours (orange
tabs, blue marker ridges, teal inner band) are paint only and change no physics. `tools/sheet.py` makes the contact
sheets; `tools/render_logs/render_v3.sh` and `overlay_v3b.sh` are the runs.

## Figures

- `static/images/method.svg` (method overview) and `mechanisms.svg` (the four closure types as section views):
  `python tools/make_figures.py [--preview-dir DIR]` (matplotlib; serif text converted to paths). Labels are short
  Title-Case nouns as in related papers' method figures (groups Isaac Lab, Observation, Policy, Training; a legend
  of arrow types: Control Loop, Simulator, Training Only); what each box shows follows the training code and the
  recipe's command line, listed in the script's docstring. The script refuses to write a figure in which a text
  crosses the border of the box that holds it. Type colours follow the clip overlay (push teal, squeeze orange,
  twist violet). Render thumbnails in `tools/fig_assets/`: `hand_over_bottle.jpg` is a crop of the M3 poster of
  commit 269cd91 (the same checkpoint, earlier wide camera, no mechanism paint); `fingertips_on_cap.jpg` is a crop
  of the clean v3 frame M3 step 268, and `cap_closeup.jpg` the crop x 432-856, y 232-492 of the clean 1280 x 720 v3
  frame M1 step 0 (box 1 `/workspace/sitemedia/final_v3/m{3,1}_frames`).
- `static/images/v1_success_e10000.svg`, `v1_ablation_m3_e6000.svg` and the per-seed table in `index.html` (between
  the `v1 success table` markers): `python tools/make_v1_charts.py --data <dir> [--preview-dir DIR] [--html
  index.html]` from the recorded physics-v1 analysis outputs (`w5_stats` Table 2 of both arms at epoch 10000;
  `w5_gd` at epoch 6000). `tools/data/v1_numbers.json` holds every number the charts and the table show, with the
  sha256 of each source file. The page calls the two arms "without / with posture regularization". The seeds are
  drawn as a beeswarm (x only separates seeds of one column), means are labelled with 2 decimals (the table keeps
  full precision), and the text is Noto Sans like the page: `tools/fonts/` holds a Latin subset of Noto Sans 2.015
  Regular and Bold (SIL Open Font Licence, `OFL-Noto-Sans.txt`).

## Sources for the layout and wording

The 2026-11-18 rebuild follows a survey of 31 project pages (7 ETH CVG pages including Elevator-VIGS, other
ETH/UZH robotics pages, and dexterous / Isaac Gym / Isaac Lab pages such as HORA, Twisting Lids Off, DexterityGen,
DeXtreme and DextrAH-G) and of 19 method figures of sim-trained dexterous RL papers: one figure and one paragraph per
section, a one-line TL;DR, results as videos plus at most two charts, statistics left to the paper, the simulator
named once ("Isaac Lab"), and short Title-Case figure labels.

Preview locally: `python -m http.server 8765` in this folder, then open http://localhost:8765/.
