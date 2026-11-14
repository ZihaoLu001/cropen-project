# CR-Open project page

Source of the project page for **Push, Squeeze or Turn? Learning to Open Child-Resistant Closures of Hidden Type
with a Dexterous Hand (CR-Open)**. Work in progress; all results are in simulation.

- `index.html`, `static/css/index.css`, `static/js/index.js`: the page (adapted from the Nerfies template, as in
  ZihaoLu001/oap-project; licence notices in `static/vendor/`).
- `static/videos/`: simulation clips of one trained policy (seed 3, epoch 8000) rendered with
  `tools/render_demo.py` (presentation media: the family is fixed per clip, the first successful episode is kept,
  every attempt is listed in the `*_render_log.json` files).
- `static/images/curve_seed3.svg`, `bars_seed3_e8000.svg`: made by `tools/make_charts.py` from the formal
  evaluations of that seed (2,048 first episodes per checkpoint, full gate thresholds).
- `static/images/method.svg`: method overview.

Preview locally: `python -m http.server 8765` in this folder, then open http://localhost:8765/.
