#!/bin/bash
# v3b: HUD clips, teaser and contact sheets again from the SAME final_v3 renders (frames, state logs, render logs;
# CPU only, nothing re-rendered): marks on the cap from the render camera (--render-log), banners that end when the
# gate locks again, released status, hysteresis release line, an extra M3 slow window over a full two-tab squeeze,
# chosen poster rows. New folders hud2/ and sheets2/; hud/ and sheets/ (v3) are kept.
cd /workspace/sitemedia/final_v3
T=/workspace/sitemedia/v7/tools
O=$T/overlay_mech.py
P=/venv/main/bin/python
sha256sum $O $T/sheet.py
mkdir -p hud2 sheets2
: > overlay_v3b_log.jsonl
for f in m1 m2 m3 m6; do
  case $f in
    m1) X="--slowmo 1" ;;
    m2) X="--slowmo 3 --slow-window 30 20 --poster-row 24" ;;
    m3) X="--slowmo 3 --slow-window 30 12 --slow-rows 258 280 --poster-row 268 --crf 27" ;;
    m6) X="--slowmo 3 --slow-window 30 20 --poster-row 8" ;;
  esac
  nice -n 5 $P $O --frames ${f}_frames --state ${f}_state.json --render-log ${f}_render_log.json --out hud2/$f.mp4 --poster hud2/$f-poster.jpg $X >> overlay_v3b_log.jsonl || exit 1
  nice -n 5 $P $O --frames ${f}_frames --state ${f}_state.json --render-log ${f}_render_log.json --out hud2/${f}_q.mp4 --mode compact --size 640 360 --crf 21 --tail 1.0 >> overlay_v3b_log.jsonl || exit 1
  $P $T/sheet.py hud2/$f.mp4 ${f^^} sheets2/${f}_grid.jpg
done
cd hud2
# teaser: 2x2 (clockwise from top left M1, M2, M6, M3), each quadrant clip looped to 16 s (as v3)
ffmpeg -nostdin -y -loglevel error -stream_loop -1 -t 16 -i m1_q.mp4 -stream_loop -1 -t 16 -i m2_q.mp4 -stream_loop -1 -t 16 -i m6_q.mp4 -stream_loop -1 -t 16 -i m3_q.mp4 \
  -filter_complex "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[2:v]setpts=PTS-STARTPTS[c];[3:v]setpts=PTS-STARTPTS[d];[a][b][d][c]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0,trim=duration=16,format=yuv420p[v]" \
  -map "[v]" -c:v libx264 -crf ${TCRF:-29} -preset slow -movflags +faststart teaser.mp4 < /dev/null && \
ffmpeg -nostdin -y -loglevel error -ss 2.87 -i teaser.mp4 -frames:v 1 -q:v 3 teaser-poster.jpg < /dev/null && echo TEASER_OK
$P $T/sheet.py teaser.mp4 TEASER ../sheets2/teaser_grid.jpg
cd /workspace/sitemedia/final_v3
sha256sum hud2/*.mp4 hud2/*-poster.jpg overlay_v3b_log.jsonl sheets2/*.jpg > SHA256SUMS_v3b
ls -la hud2 sheets2
echo OVERLAY_V3B_DONE
