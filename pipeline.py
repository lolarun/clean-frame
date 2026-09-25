"""Per-video pipeline: OCR -> SRT -> masks -> erase (LaMa or ProPainter) -> encode."""
import json
import time
from pathlib import Path

import numpy as np

from common import log
from masks import box_masks, frame_masks, glyph_masks, white_pixels
from subtitles import frame_boxes, segments_from_ocr, write_srt
from video import detect_cuts, open_writer, probe, read_frames


def parse_band(s, H):
    """'0.7,1.0' (fractions) or '800,1080' (pixels) -> (y0, y1) in pixels, even-aligned"""
    a, b = (float(v) for v in s.split(","))
    if a <= 1 and b <= 1:
        a, b = a * H, b * H
    a, b = int(a) // 2 * 2, int(b) // 2 * 2
    return max(0, a), min(H, b)


def run_ocr(src, W, H, n, band, ocr, args):
    """Pass 1: per-frame OCR boxes of the subtitle band. If the white-pixel layout barely changed since
    the last OCR'd frame, the subtitle is the same and the result is reused (OCR is still forced every
    --ocr-interval frames)."""
    t0 = time.time()
    per_frame = []
    min_h = max(8, int(H * args.min_height))
    ref, last, n_ocr = None, -1, 0
    for i, strip in enumerate(read_frames(src, W, H, band)):
        sig = white_pixels(strip[::2, ::2])
        if ref is not None and i - last < args.ocr_interval:
            diff = np.count_nonzero(sig ^ ref)
            if diff <= max(30, 0.08 * np.count_nonzero(ref)):
                per_frame.append(per_frame[-1])
                continue
        boxes = frame_boxes(ocr(strip), band[0], W, min_h, args.min_score)
        per_frame.append(boxes)
        ref, last = sig, i
        n_ocr += 1
        if n_ocr % 100 == 0:
            el = time.time() - t0
            txt = " / ".join(b[4] for b in boxes)
            log(f"  [ocr] {i}/{n}  {i / max(el, 1e-6):.1f} fps  {txt}")
    log(f"  OCR ran on {n_ocr}/{len(per_frame)} frames")
    return per_frame


def process(src, out_dir, args, ocr, backend, encoder):
    src = Path(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H, fps, n = probe(src)
    band = parse_band(args.band, H)
    log(f"\n==> {src.name}  {W}x{H} {fps:.3f}fps  ~{n} frames  subtitle band y={band[0]}-{band[1]}")

    # Pass 1: OCR (raw results are cached as JSON so reruns skip recognition)
    t0 = time.time()
    cache = out_dir / ".cache" / f"{src.stem}.{band[0]}-{band[1]}.ocr.json"
    if cache.exists() and not args.no_cache:
        per_frame = json.loads(cache.read_text(encoding="utf-8"))
        log(f"  using OCR cache {cache}")
    else:
        per_frame = run_ocr(src, W, H, n, band, ocr, args)
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(per_frame, ensure_ascii=False), encoding="utf-8")
    line, segs = segments_from_ocr(per_frame, fps, max_gap=args.max_gap, min_dur=args.min_dur)
    if line:
        log(f"  subtitle line y~{line[0]:.0f}, text height ~{line[1]:.0f}px")
    srt = out_dir / (src.stem + ".srt")
    write_srt(segs, fps, srt)
    log(f"  {len(segs)} subtitles -> {srt}  ({time.time() - t0:.0f}s)")
    if backend is None:
        return

    # One fixed mask per subtitle keeps the result temporally stable
    t0 = time.time()
    if args.mask == "glyph":
        masks = glyph_masks(src, W, H, band, segs, args.dilate, args.grow, args.shadow)
    else:
        masks = box_masks(segs, H, W, args.dilate)
    frame_seg, masks = frame_masks(segs, masks, args.pad_frames)
    log(f"  {args.mask} masks done ({time.time() - t0:.0f}s)")

    cuts = []
    if getattr(backend, "uses_cuts", False):
        t0 = time.time()
        cuts = detect_cuts(src)
        log(f"  {len(cuts)} shot cuts ({time.time() - t0:.0f}s)")

    # Pass 2: erase + encode
    t0 = time.time()
    dst = out_dir / (src.stem + "_clean.mp4")
    wr = open_writer(src, dst, W, H, fps, encoder, args.crf)
    written = 0
    try:
        for frame in backend.erase(read_frames(src, W, H), frame_seg, masks, cuts):
            wr.stdin.write(np.ascontiguousarray(frame).tobytes())
            written += 1
            if written % 100 == 0:
                log(f"  [erase] {written}/{n}  {written / max(time.time() - t0, 1e-6):.1f} fps")
    finally:
        wr.stdin.close()
        wr.wait()
    if wr.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed (code {wr.returncode})")
    log(f"  video -> {dst}  ({time.time() - t0:.0f}s, {backend.name})")
