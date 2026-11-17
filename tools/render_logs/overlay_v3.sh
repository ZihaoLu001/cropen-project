#!/bin/bash
# HUD clips, teaser quadrants, teaser and contact sheets from the final_v3 renders (CPU only)
cd /workspace/sitemedia/final_v3
O=/workspace/sitemedia/v6/tools/overlay_mech.py
P=/venv/main/bin/python
sha256sum $O
mkdir -p hud sheets
for f in m1 m2 m3 m6; do
  slow=${SLOW:-3}; [ $f = m1 ] && slow=1
  crf=${CRF:-23}; [ $f = m3 ] && crf=${CRF3:-27}
  $P $O --frames ${f}_frames --state ${f}_state.json --out hud/$f.mp4 --poster hud/$f-poster.jpg --slowmo $slow --slow-window 30 20 --crf $crf || exit 1
  $P $O --frames ${f}_frames --state ${f}_state.json --out hud/${f}_q.mp4 --mode compact --size 640 360 --crf 21 --tail 1.0 || exit 1
  $P /workspace/sitemedia/v6/sheet.py hud/$f.mp4 ${f^^} sheets/${f}_grid.jpg
done
cd hud
# teaser: 2x2 (clockwise from top left M1, M2, M6, M3), each quadrant clip looped to 16 s (as render_v2.sh)
ffmpeg -nostdin -y -loglevel error -stream_loop -1 -t 16 -i m1_q.mp4 -stream_loop -1 -t 16 -i m2_q.mp4 -stream_loop -1 -t 16 -i m6_q.mp4 -stream_loop -1 -t 16 -i m3_q.mp4 \
  -filter_complex "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[2:v]setpts=PTS-STARTPTS[c];[3:v]setpts=PTS-STARTPTS[d];[a][b][d][c]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0,trim=duration=16,format=yuv420p[v]" \
  -map "[v]" -c:v libx264 -crf ${TCRF:-28} -preset slow -movflags +faststart teaser.mp4 < /dev/null && \
ffmpeg -nostdin -y -loglevel error -ss 3.5 -i teaser.mp4 -frames:v 1 -q:v 3 teaser-poster.jpg < /dev/null && echo TEASER_OK
$P /workspace/sitemedia/v6/sheet.py teaser.mp4 TEASER ../sheets/teaser_grid.jpg
ls -la /workspace/sitemedia/final_v3/hud /workspace/sitemedia/final_v3/sheets
echo OVERLAY_V3_DONE
