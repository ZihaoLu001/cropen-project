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
- `static/images/curve_seed3.svg`, `bars_seed3_e8000.svg`: made by `tools/make_charts.py` from the formal
  evaluations of that seed (2,048 first episodes per checkpoint, full gate thresholds).
- `static/images/method.svg`: method overview.

Preview locally: `python -m http.server 8765` in this folder, then open http://localhost:8765/.
