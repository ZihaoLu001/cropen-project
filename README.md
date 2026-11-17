# CR-Open project page

Source of the project page for **Push, Squeeze or Turn? Learning to Open Child-Resistant Closures of Hidden Type
with a Dexterous Hand (CR-Open)**. Work in progress; all results are in simulation.

- `index.html`, `static/css/index.css`, `static/js/index.js`: the page (adapted from the Nerfies template, as in
  ZihaoLu001/oap-project; licence notices in `static/vendor/`).
- `static/videos/`: simulation clips of one trained policy (stage-1 seed 4 of the N1 arm, physics v2 at 255 position
  iterations, epoch 10000; chosen after the three stage-1 seeds' epoch-10000 rates were read) rendered with
  `tools/render_demo.py` (presentation media: the family is fixed per clip, the first successful episode is kept,
  every attempt is listed in the `*_render_log.json` files).
  The state display and the marks on the cap are drawn by `tools/overlay_mech.py` from the per-step state log of
  the same episode (`tools/render_logs/m*_state.json`) and the render camera (`m*_render_log.json`); the mechanism
  colours (orange tabs, blue marker ridges, teal inner band) are paint only and change no physics.
  `tools/sheet.py` makes the contact sheets; `tools/render_logs/render_v3.sh` and `overlay_v3b.sh` are the runs.
- `static/images/method.svg` (method overview) and `mechanisms.svg` (the four families as section views): made by
  `tools/make_figures.py` (matplotlib; serif text converted to paths; every label follows the training code and the
  recipe's command line, listed in the script's docstring). The script refuses to write a figure in which a text
  crosses the border of the box that holds it. Arrow styles: black, the 30 Hz control loop; amber, simulator
  internals every physics substep; dashed slate, training only. Family colours follow the clip overlay (push teal,
  squeeze orange, twist violet; checked with the dataviz palette validator, all pairs). Render thumbnails in
  `tools/fig_assets/`:
  `hand_over_bottle.jpg` is a crop of the M3 poster of commit 269cd91 (the same checkpoint, earlier wide camera,
  no mechanism paint; the page caption says so);
  `fingertips_on_cap.jpg` is a crop of the clean v3 frame M3 step 268, and `cap_closeup.jpg` the crop x 432-856,
  y 232-492 of the clean 1280 x 720 v3 frame M1 step 0, narrow enough that no fingertip shows
  (box 1 `/workspace/sitemedia/final_v3/m{3,1}_frames`).
- `static/images/v1_success_e10000.svg`, `v1_ablation_m3_e6000.svg` and the two table views in `index.html`: made by
  `tools/make_v1_charts.py --data <dir> --html index.html` from the recorded physics-v1 analysis outputs
  (`w5_stats` Table 2 of both arms at epoch 10000; `w5_gd` at epoch 6000). `tools/data/v1_numbers.json` holds every
  number the charts and tables show, with the sha256 of each source file. The seeds are drawn as a beeswarm (x only
  separates seeds of one column), means are labelled with 2 decimals (the tables keep full precision), and the text
  is Noto Sans like the page: `tools/fonts/` holds a Latin subset of Noto Sans 2.015 Regular and Bold (SIL Open Font
  Licence, `OFL-Noto-Sans.txt`).
- On phones (width up to 600 px) every figure and chart keeps a 760 px width and scrolls sideways inside its own box.

Preview locally: `python -m http.server 8765` in this folder, then open http://localhost:8765/.
