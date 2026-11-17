"""contact sheet: 8 frames of a clip (evenly spread), 4 x 2 tiles of 640x360 with the clip time (frames via ffmpeg)
usage: sheet.py clip.mp4 label out.jpg"""
import subprocess
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
clip, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", clip], capture_output=True, text=True).stdout.strip())
idx = [int(round(k * (n - 1) / 7)) for k in range(8)]
with tempfile.TemporaryDirectory() as d:
    sel = "+".join(rf"eq(n\,{i})" for i in idx)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", clip, "-vf", f"select='{sel}',scale=640:360",
                    "-vsync", "0", f"{d}/f_%02d.png"], check=True)
    tiles = []
    for k, i in enumerate(idx):
        t = cv2.imread(f"{d}/f_{k + 1:02d}.png")
        cv2.putText(t, f"{label} t={i / 30:.1f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(t)
img = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out, img.shape, n, idx)
