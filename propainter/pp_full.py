"""Clean Frame - ProPainter high-quality erase mode (experimental)

Reuses the OCR cache and glyph masks from cleanframe.py, runs video inpainting only on the frame
ranges that contain subtitles and only on the band where they appear, then composites the result
back into the original video. Run cleanframe.py once first (--srt-only is enough) to create the
OCR cache.

Usage:
  python pp_full.py video.mp4 --out output --propainter /path/to/ProPainter [--raft-iter 20]

Note: ProPainter is released under the NTU S-Lab License 1.0 - non-commercial use only.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import cleanframe as cf  # noqa: E402

ap = argparse.ArgumentParser(description="Clean Frame: ProPainter high-quality erase (experimental)")
ap.add_argument("video")
ap.add_argument("-o", "--out", default="output", help="same output directory as cleanframe.py (the OCR cache is read from it)")
ap.add_argument("--propainter", required=True, help="ProPainter repository directory (containing weights/)")
ap.add_argument("--band", default="0.70,1.0", help="same as cleanframe.py")
ap.add_argument("--raft-iter", type=int, default=20, help="optical flow iterations, fewer is faster (default: 20)")
ap.add_argument("--chunk", type=int, default=120, help="frames per chunk; lower it if GPU memory runs out (80 for 8 GB cards)")
args = ap.parse_args()

src = Path(args.video)
out_dir = Path(args.out)
W, H, fps, n = cf.probe(src)
band = cf.parse_band(args.band, H)
caches = glob.glob(str(out_dir / ".cache" / f"{glob.escape(src.stem)}.{band[0]}-{band[1]}.ocr.json"))
if not caches:
    sys.exit(f"OCR cache not found; run first: python cleanframe.py \"{src}\" -o \"{out_dir}\" --srt-only")

os.chdir(args.propainter)
sys.path.insert(0, args.propainter)
sys.path.insert(0, str(HERE))
from pp_engine import ProPainter  # noqa: E402

CHUNK, CTX, PAD = args.chunk, 10, 8
t0 = time.time()
pf = json.loads(Path(caches[0]).read_text(encoding="utf-8"))
line = cf.subtitle_line(pf)
pf = [cf.join_text(cf.filter_line(b, line)) for b in pf]
segs = cf.build_segments(pf, 5, max(2, round(0.4 * fps)))
masks = cf.glyph_masks(src, W, H, band, segs, 10, 8, 3)
frame_seg = {}
for k, s in enumerate(segs):
    for i in range(max(0, s.start - 1), s.end + 2):
        frame_seg.setdefault(i, k)
rows = np.nonzero(np.any([m.any(1) for m in masks.values()], 0))[0]
r0 = int(max(0, (rows.min() - 40) // 8 * 8))
r1 = int(min(H, r0 + ((rows.max() + 40 - r0 + 7) // 8) * 8))
print(f"{len(segs)} subtitles  band y={r0}-{r1}  masks {time.time()-t0:.0f}s", flush=True)

# Frames to inpaint -> contiguous runs (gaps <= 10 frames merged, PAD frames added at both ends)
# -> chunks of CHUNK frames, each with CTX frames of context on either side
runs = []
for i in sorted(frame_seg):
    if runs and i - runs[-1][1] <= 10:
        runs[-1][1] = i
    else:
        runs.append([i, i])
chunks = []
for a, b in runs:
    a, b = max(0, a - PAD), min(n - 1, b + PAD)
    for s in range(a, b + 1, CHUNK):
        e = min(b, s + CHUNK - 1)
        chunks.append((s, e, max(0, s - CTX), min(n - 1, e + CTX)))
need = set()
for s, e, cs, ce in chunks:
    need.update(range(cs, ce + 1))
print(f"{len(frame_seg)} frames to inpaint in {len(chunks)} chunks, {len(need)} frames processed", flush=True)

bands = {}
for i, f in enumerate(cf.read_frames(src, W, H, (r0, r1))):
    if i in need:
        bands[i] = f.copy()

pp = ProPainter("weights", raft_iter=args.raft_iter, subvideo_length=CHUNK + 2 * CTX)
empty = np.zeros((r1 - r0, W), bool)
out = {}
for ci, (s, e, cs, ce) in enumerate(chunks):
    t1 = time.time()
    ids = list(range(cs, ce + 1))
    ms = [masks[frame_seg[i]][r0:r1] if i in frame_seg else empty for i in ids]
    res = pp([bands[i] for i in ids], ms)
    for j, i in enumerate(ids):
        if s <= i <= e and i in frame_seg:
            out[i] = res[j]
    print(f"  [inpaint] {ci + 1}/{len(chunks)}  {time.time() - t1:.0f}s  total {time.time() - t0:.0f}s", flush=True)

# Composite back into the original video: only pixels within the mask dilated by 4 px are replaced
kern = np.ones((9, 9), np.uint8)
dm = {k: cv2.dilate(m[r0:r1].astype(np.uint8), kern).astype(bool) for k, m in masks.items()}
dst = out_dir / (src.stem + "_clean_pp.mp4")
wr = cf.open_writer(src, dst, W, H, fps, cf.pick_encoder("auto"), 18)
for i, f in enumerate(cf.read_frames(src, W, H)):
    if i in out:
        f = f.copy()
        b = f[r0:r1]
        m = dm[frame_seg[i]]
        b[m] = out[i][m]
    wr.stdin.write(np.ascontiguousarray(f).tobytes())
wr.stdin.close()
wr.wait()
print(f"done -> {dst}  total {time.time() - t0:.0f}s", flush=True)
