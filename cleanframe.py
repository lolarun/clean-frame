"""
Clean Frame - hardcoded subtitle removal + subtitle extraction (SRT)

Pipeline:
  Pass 1  decode only the subtitle band -> RapidOCR detection + recognition -> merge into
          subtitle segments -> write .srt
  Pass 2  build a fixed glyph mask per segment -> LaMa inpainting of full frames ->
          ffmpeg encode (audio stream copied)

Requires: ffmpeg/ffprobe (on PATH), onnxruntime(-gpu), rapidocr, opencv-python, numpy
"""
import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".m4v", ".wmv", ".webm"}
HERE = Path(__file__).resolve().parent
LAMA_NAME = "lama_fp32.onnx"
LAMA_URLS = [
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
]


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- ffmpeg

def _exe(name):
    local = HERE / "ffmpeg" / (name + (".exe" if os.name == "nt" else ""))
    if local.exists():
        return str(local)
    p = shutil.which(name)
    if not p:
        sys.exit(f"{name} not found: install ffmpeg and add it to PATH, or put it in {HERE / 'ffmpeg'}")
    return p


def probe(path):
    out = subprocess.run(
        [_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
         "-of", "json", str(path)],
        capture_output=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = (s.get("avg_frame_rate") or s["r_frame_rate"]).split("/")
    if float(den) == 0 or float(num) == 0:
        num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    dur = float(j["format"].get("duration", 0) or 0)
    n = int(s.get("nb_frames") or 0) or int(round(dur * fps))
    return int(s["width"]), int(s["height"]), fps, n


def read_frames(path, w, h, crop=None):
    """Yield frames as BGR ndarrays. With crop=(y0, y1) only that horizontal band is decoded."""
    cmd = [_exe("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0"]
    if crop:
        y0, y1 = crop
        cmd += ["-vf", f"crop={w}:{y1 - y0}:0:{y0}"]
        h = y1 - y0
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=int(w * h * 3 * 4))
    size = w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def pick_encoder(want):
    if want != "auto":
        return want
    try:
        r = subprocess.run(
            [_exe("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.2",
             "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=30)
        if r.returncode == 0:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def open_writer(src, dst, w, h, fps, encoder, crf):
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        venc = ["-c:v", encoder, "-preset", "p5", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        venc = ["-c:v", encoder, "-preset", "medium", "-crf", str(crf)]
    cmd = [_exe("ffmpeg"), "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
           "-i", str(src), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy",
           *venc, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ----------------------------------------------------------------------------- devices / models

def providers_for(device):
    import onnxruntime as ort
    try:
        ort.preload_dlls()  # use the pip-installed CUDA/cuDNN (onnxruntime-gpu[cuda,cudnn])
    except Exception:
        pass
    avail = ort.get_available_providers()
    if device == "auto":
        device = "cuda" if "CUDAExecutionProvider" in avail else (
            "dml" if "DmlExecutionProvider" in avail else "cpu")
    prov = {"cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
            "cpu": ["CPUExecutionProvider"]}[device]
    if prov[0] not in avail:
        sys.exit(f"Device {device} is not available; onnxruntime providers: {avail}")
    return device, prov


def lama_path():
    for p in [HERE / "models" / LAMA_NAME, Path.home() / ".cache" / "cleanframe" / LAMA_NAME]:
        if p.exists():
            return p
    dst = HERE / "models" / LAMA_NAME
    dst.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for url in LAMA_URLS:
        try:
            log(f"Downloading LaMa model: {url}")
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dst)
            return dst
        except Exception as e:
            log(f"  failed: {e}")
    sys.exit(f"Could not download the LaMa model; download {LAMA_NAME} manually into {dst.parent}")


class OCR:
    def __init__(self, device):
        from rapidocr import RapidOCR
        params = {"Det.limit_type": "max", "Det.limit_side_len": 960,
                  "Global.log_level": "error"}
        if device == "cuda":
            params["EngineConfig.onnxruntime.use_cuda"] = True
        elif device == "dml":
            params["EngineConfig.onnxruntime.use_dml"] = True
        self.engine = RapidOCR(params=params)

    def __call__(self, img):
        r = self.engine(img, use_cls=False)
        if r.boxes is None or r.txts is None:
            return []
        return [(np.asarray(b), t, float(s)) for b, t, s in zip(r.boxes, r.txts, r.scores)]


class Lama:
    SIZE = 512

    def __init__(self, providers):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(lama_path()), so, providers=providers)

    def __call__(self, img, mask):
        """img: HxWx3 BGR uint8 (square), mask: HxW bool -> inpainted BGR uint8"""
        h, w = mask.shape
        S = self.SIZE
        a = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA if h > S else cv2.INTER_CUBIC)
        a = a[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        m = cv2.resize(mask.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
        m = m.astype(np.float32)[None, None]
        out = self.sess.run(None, {"image": np.ascontiguousarray(a), "mask": m})[0][0]
        out = np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8)[:, :, ::-1]
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


# ----------------------------------------------------------------------------- subtitle segments

@dataclass
class Seg:
    start: int
    end: int
    texts: Counter = field(default_factory=Counter)
    boxes: list = field(default_factory=list)  # (x0, y0, x1, y1) in full-frame coordinates

    @property
    def text(self):
        return self.texts.most_common(1)[0][0]

    @property
    def xr(self):
        return min(b[0] for b in self.boxes), max(b[2] for b in self.boxes)


def frame_boxes(items, band_y0, W, min_h, min_score):
    """First-pass filter of one frame's OCR results -> [(x0, y0, x1, y1, text)] in full-frame coordinates"""
    keep = []
    for box, txt, score in items:
        x0, y0 = box.min(0)
        x1, y1 = box.max(0)
        cx = (x0 + x1) / 2
        if score < min_score or (y1 - y0) < min_h or not txt.strip():
            continue
        if not (0.15 * W < cx < 0.85 * W):  # subtitles are normally centred
            continue
        keep.append((int(x0), int(y0) + band_y0, int(x1), int(y1) + band_y0, txt.strip()))
    return keep


def subtitle_line(per_frame):
    """Find where subtitles usually sit from all text boxes in the video: (bottom line y-centre, text height)"""
    ys = [((b[1] + b[3]) / 2, b[3] - b[1]) for boxes in per_frame for b in boxes]
    if not ys:
        return None
    yc = np.array([y for y, _ in ys])
    hs = np.array([h for _, h in ys])
    hist = np.bincount((yc // 4).astype(int))
    mode = (np.argmax(hist) + 0.5) * 4
    h = float(np.median(hs[np.abs(yc - mode) < 8]))
    return mode, h


def filter_line(boxes, line):
    """Keep only boxes matching the subtitle line position and height (one extra line above is allowed for two-line subtitles)"""
    if line is None:
        return boxes
    mode, h = line
    keep = [b for b in boxes
            if 0.75 * h <= b[3] - b[1] <= 1.33 * h
            and mode - 2.5 * h <= (b[1] + b[3]) / 2 <= mode + 0.5 * h]
    main = [b for b in keep if abs((b[1] + b[3]) / 2 - mode) <= 0.5 * h]
    if not main:
        return []
    # An upper line must lie entirely above the subtitle line (boxes overlapping it vertically are scene texture/objects)
    top = min(b[1] for b in main)
    return main + [b for b in keep if b not in main and b[3] <= top + 0.1 * h]


def join_text(keep):
    """Sort into lines and join -> (text, [boxes])"""
    if not keep:
        return None, []
    # Group boxes whose y-centres are close into the same line
    keep.sort(key=lambda b: (b[1] + b[3]) / 2)
    lines, cur = [], [keep[0]]
    for b in keep[1:]:
        ph = cur[-1][3] - cur[-1][1]
        if abs((b[1] + b[3]) / 2 - (cur[-1][1] + cur[-1][3]) / 2) < 0.5 * ph:
            cur.append(b)
        else:
            lines.append(cur)
            cur = [b]
    lines.append(cur)
    text = "\n".join(" ".join(b[4] for b in sorted(l, key=lambda b: b[0])) for l in lines)
    return text, [b[:4] for b in keep]


def similar(a, b):
    return difflib.SequenceMatcher(None, a.replace(" ", ""), b.replace(" ", "")).ratio()


def _x_overlap(a, b):
    (ax0, ax1), (bx0, bx1) = a.xr, b.xr
    return min(ax1, bx1) - max(ax0, bx0) > 0.4 * min(ax1 - ax0, bx1 - bx0)


def build_segments(per_frame, max_gap, min_len):
    # 1) Consecutive frames with identical text (ignoring spaces) -> runs
    runs = []
    for i, (text, boxes) in enumerate(per_frame):
        if not text:
            continue
        r = runs[-1] if runs else None
        if r and i - r.end <= max_gap + 1 and r.text.replace(" ", "") == text.replace(" ", ""):
            r.end = i
            r.texts[text] += 1
            r.boxes += boxes
        else:
            runs.append(Seg(i, i, Counter({text: 1}), list(boxes)))
    # 2) Merge OCR jitter: very similar text, or a short run fairly similar to the previous one.
    #    Two long runs with different text are kept as two separate subtitles.
    short = max(min_len, 4)
    segs = []
    for r in runs:
        p = segs[-1] if segs else None
        if p and r.start - p.end <= max_gap + 1 and _x_overlap(p, r):
            s = similar(p.text, r.text)
            rl, pl = r.end - r.start + 1, p.end - p.start + 1
            if s >= 0.9 or (s >= 0.6 and min(rl, pl) < short):
                p.end = r.end
                p.texts.update(r.texts)
                p.boxes += r.boxes
                continue
        segs.append(r)
    return [s for s in segs if s.end - s.start + 1 >= min_len]


def ts(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segs, fps, path):
    with open(path, "w", encoding="utf-8") as f:
        for k, s in enumerate(segs, 1):
            f.write(f"{k}\n{ts(s.start / fps)} --> {ts((s.end + 1) / fps)}\n{s.text}\n\n")


# ----------------------------------------------------------------------------- inpainting

def inpaint_frame(frame, mask, lama):
    """Inpaint the mask with square windows from left to right; each window uses the previous window's result as context."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return frame
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    T = int(np.clip((y1 - y0) * 2.5, 512, min(H, W)))
    cy = (y0 + y1) // 2
    Y0 = int(np.clip(cy - T // 2, 0, H - T))
    out = frame.copy()
    remain = mask.copy()
    X = max(0, x0 - T // 4)
    while True:
        X = min(X, W - T)
        last = X + T >= min(W, x1 + T // 4)
        region = out[Y0:Y0 + T, X:X + T]
        m = remain[Y0:Y0 + T, X:X + T].copy()
        if not last:
            m[:, 3 * T // 4:] = False
        if m.any():
            res = lama(region, remain[Y0:Y0 + T, X:X + T])
            region[m] = res[m]
            remain[Y0:Y0 + T, X:X + T][m] = False
        if last:
            break
        X += T // 2
    return out


_plan_cache = {}


def _pack_plan(mask, S, ctx, side):
    """Compute the strip layout; the mask is constant within a segment, so cache it per mask object"""
    key = id(mask)
    hit = _plan_cache.get(key)
    if hit is not None and hit[0] is mask:
        return hit[1]
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        plan = None
    else:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        k = S // (y1 - y0 + 2 * ctx)
        if k < 2 or W < S or H < S:
            plan = "square"
        else:
            hs = S // k
            Y0 = int(np.clip((y0 + y1) // 2 - hs // 2, 0, H - hs))
            col = mask[Y0:Y0 + hs].sum(0)
            F = S - 2 * side
            cuts = [x0]
            while x1 - cuts[-1] > F:
                lo = cuts[-1] + F - 96
                cuts.append(lo + int(np.argmin(col[lo:cuts[-1] + F + 1])))
            cuts.append(x1)
            pieces = [(a, b, int(np.clip(a - side, 0, W - S))) for a, b in zip(cuts[:-1], cuts[1:])]
            plan = (k, hs, Y0, pieces)
    _plan_cache.clear()
    _plan_cache[key] = (mask, plan)
    return plan


def inpaint_packed(frame, mask, lama, ctx=24, side=64):
    """A subtitle is a long, thin band: cut it into 512-wide strips and stack them into one 512x512
    image so a single LaMa call inpaints the whole line - 2-4x faster than square windows.
    Cuts are placed at the columns with the fewest mask pixels (gaps between characters), and
    every strip carries `side` pixels of context on each side."""
    plan = _pack_plan(mask, S=lama.SIZE, ctx=ctx, side=side)
    if plan is None:
        return frame
    if plan == "square":
        return inpaint_frame(frame, mask, lama)
    k, hs, Y0, pieces = plan
    S = lama.SIZE
    out = frame.copy()
    band = out[Y0:Y0 + hs]
    mband = mask[Y0:Y0 + hs]
    for j in range(0, len(pieces), k):
        chunk = pieces[j:j + k]
        img = np.zeros((S, S, 3), np.uint8)
        m = np.zeros((S, S), bool)
        for r, (a, b, ws) in enumerate(chunk):
            img[r * hs:(r + 1) * hs] = band[:, ws:ws + S]
            m[r * hs:(r + 1) * hs] = mband[:, ws:ws + S]
        used = len(chunk) * hs
        if used < S:  # pad unused rows with edge pixels
            img[used:] = img[used - 1]
        res = lama(img, m)
        for r, (a, b, ws) in enumerate(chunk):
            sl = slice(a - ws, b - ws)
            mm = mband[:, a:b]
            band[:, a:b][mm] = res[r * hs:(r + 1) * hs, sl][mm]
    return out


class TemporalFiller:
    """Experimental temporal fill: background hidden by a subtitle is often visible in nearby frames
    (gaps between subtitles, moving people or camera). The masked area is processed in 64 px column
    blocks; candidate frames within +/-N are tried nearest first. If a candidate is not covered by a
    subtitle there and the ring of pixels around the mask matches the current frame (same shot,
    aligned), its real pixels are copied (with brightness compensation). Otherwise the previous
    output frame is tried, and whatever remains goes to LaMa."""

    def __init__(self, src, W, H, rows, frame_seg, masks, radius=50, thr=7.0, block=64):
        self.W, self.rows = W, rows
        self.frame_seg, self.masks = frame_seg, masks
        self.N, self.thr, self.block = radius, thr, block
        self.reader = read_frames(src, W, H, rows)
        self.buf = {}  # frame index -> original band
        self.next_read = 0
        self.prev_out = None  # (frame index, inpainted band)
        self._geo = {}

    def _fill_buffer(self, t):
        while self.next_read <= t + self.N:
            try:
                self.buf[self.next_read] = next(self.reader).copy()
            except StopIteration:
                self.next_read = 1 << 60
                break
            self.next_read += 1
        for j in [j for j in self.buf if j < t - self.N]:
            del self.buf[j]

    def _band_mask(self, j):
        k = self.frame_seg.get(j)
        if k is None:
            return None
        return self.masks[k][self.rows[0]:self.rows[1]]

    def _geometry(self, M):
        key = id(M)
        g = self._geo.get(key)
        if g is None or g[0] is not M:
            ring = cv2.dilate(M.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~M
            xs = np.nonzero(M.any(0))[0]
            x0, x1 = xs.min(), xs.max() + 1
            edges = np.arange(x0, x1, self.block)
            self._geo = {key: (M, (ring, x0, x1, edges))}
            g = self._geo[key]
        return g[1]

    def fill(self, t, frame, mask, lama):
        self._fill_buffer(t)
        r0, r1 = self.rows
        M = mask[r0:r1]
        if not M.any():
            return frame
        ring, x0, x1, edges = self._geometry(M)
        cur = self.buf[t][:, x0:x1].astype(np.int16)
        need = M[:, x0:x1].copy()
        out = self.buf[t][:, x0:x1].copy()
        ringc = ring[:, x0:x1]
        bl = edges - x0
        cands = []
        for d in range(1, self.N + 1):
            for j in (t - d, t + d):
                if j in self.buf:
                    cands.append((self.buf[j], self._band_mask(j)))
        if self.prev_out is not None and self.prev_out[0] == t - 1:
            cands.append((self.prev_out[1], None))
        for cand, cm in cands:
            if not need.any():
                break
            c = cand[:, x0:x1]
            valid = ~cm[:, x0:x1] if cm is not None else np.ones_like(need)
            # only look at column blocks that still need filling
            blk_need = np.add.reduceat(need.any(0).astype(np.int32), bl) > 0
            if not blk_need.any():
                break
            r = ringc & valid
            diff = np.abs(cur - c.astype(np.int16)).sum(2) * r
            cnt = np.add.reduceat(r.sum(0), bl)
            err = np.add.reduceat(diff.sum(0), bl) / np.maximum(cnt, 1) / 3
            ring_total = np.add.reduceat(ringc.sum(0), bl)
            ok = blk_need & (cnt >= 0.3 * ring_total) & (cnt >= 30) & (err < self.thr)
            for b in np.nonzero(ok)[0]:
                s = slice(bl[b], bl[b + 1] if b + 1 < len(bl) else x1 - x0)
                f = need[:, s] & valid[:, s]
                if not f.any():
                    continue
                rb = r[:, s]
                off = (cur[:, s][rb].mean(0) - c[:, s][rb].astype(np.int16).mean(0)).clip(-12, 12)
                out[:, s][f] = np.clip(c[:, s][f] + off, 0, 255).astype(np.uint8)
                need[:, s] &= ~f
        res = frame.copy()
        res[r0:r1, x0:x1] = np.where(M[:, x0:x1, None], out, res[r0:r1, x0:x1])
        if need.any():  # remaining pixels go to LaMa (already-filled pixels serve as context)
            rest = np.zeros(mask.shape, bool)
            rest[r0:r1, x0:x1] = need
            res = inpaint_packed(res, rest, lama)
        self.prev_out = (t, res[r0:r1].copy())
        return res

    def skip(self, t):
        """Frame without subtitles: just advance the buffer"""
        self._fill_buffer(t)
        self.prev_out = None


def seg_mask(seg, H, W, dilate):
    """Union of all text boxes in a segment, expanded by `dilate` pixels"""
    m = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in seg.boxes:
        m[max(0, y0 - dilate):y1 + dilate, max(0, x0 - dilate):x1 + dilate] = True
    return m


def white_pixels(img):
    """Bright, low-saturation pixels (white subtitle glyphs)"""
    img = img.astype(np.int16)
    mn, mx = img.min(2), img.max(2)
    return (mn > 170) & (mx - mn < 50)


def glyph_masks(src, W, H, band, segs, dilate, grow, shift):
    """Glyph masks: within a segment, pixels that are white in more than half of the frames are text;
    they are then dilated to cover the outline and shadow. Inpainting only the strokes instead of the
    whole text box keeps far more original pixels, so the result is sharper, and the mask is fixed
    within a segment so it does not flicker. Segments whose text is not white fall back to the box
    mask. Returns {segment index: full-frame bool mask}."""
    by0, by1 = band
    boxm = [seg_mask(s, H, W, dilate)[by0:by1] for s in segs]
    # only count pixels inside each segment's bounding rectangle
    rects = []
    for m in boxm:
        ys, xs = np.nonzero(m)
        rects.append((ys.min(), ys.max() + 1, xs.min(), xs.max() + 1))
    seg_of = {i: k for k, s in enumerate(segs) for i in range(s.start, s.end + 1)}
    counts = {}
    for i, strip in enumerate(read_frames(src, W, H, band)):
        k = seg_of.get(i)
        if k is not None:
            ry0, ry1, rx0, rx1 = rects[k]
            c = counts.setdefault(k, np.zeros(boxm[k].shape, np.uint16))
            c[ry0:ry1, rx0:rx1] += white_pixels(strip[ry0:ry1, rx0:rx1]) & boxm[k][ry0:ry1, rx0:rx1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    out = {}
    fallback = 0
    for k, s in enumerate(segs):
        m = np.zeros((H, W), bool)
        c = counts.get(k)
        white = (c >= 0.5 * (s.end - s.start + 1)).astype(np.uint8) if c is not None else None
        if white is None or white.sum() < 0.03 * boxm[k].sum():
            m[by0:by1] = boxm[k]
            fallback += 1
        else:
            g = cv2.dilate(white, kernel)
            if shift:  # subtitle shadows usually fall to the lower right
                g |= np.roll(np.roll(g, shift, 0), shift, 1)
            m[by0:by1] = (g > 0) & cv2.dilate(boxm[k].astype(np.uint8), kernel).astype(bool)
        out[k] = m
    if fallback:
        log(f"  {fallback}/{len(segs)} subtitles have no white glyphs; using box masks")
    return out


# ----------------------------------------------------------------------------- main pipeline

def parse_band(s, H):
    a, b = (float(v) for v in s.split(","))
    if a <= 1 and b <= 1:
        a, b = a * H, b * H
    a, b = int(a) // 2 * 2, int(b) // 2 * 2
    return max(0, a), min(H, b)


def process(src, out_dir, args, ocr, lama, encoder):
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
        per_frame = []
        min_h = max(8, int(H * args.min_height))
        ref, last, n_ocr = None, -1, 0
        for i, strip in enumerate(read_frames(src, W, H, band)):
            # If the white-pixel layout barely changed since the last OCR'd frame, the subtitle is the same:
            # reuse the result (OCR is still forced every N frames)
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
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(per_frame, ensure_ascii=False), encoding="utf-8")
    line = subtitle_line(per_frame)
    if line:
        log(f"  subtitle line y~{line[0]:.0f}, text height ~{line[1]:.0f}px")
    per_frame = [join_text(filter_line(b, line)) for b in per_frame]
    segs = build_segments(per_frame, max_gap=args.max_gap, min_len=max(2, round(args.min_dur * fps)))
    srt = out_dir / (src.stem + ".srt")
    write_srt(segs, fps, srt)
    log(f"  {len(segs)} subtitles -> {srt}  ({time.time() - t0:.0f}s)")
    if args.srt_only:
        return

    # Per-frame masks: one fixed mask per segment keeps the result temporally stable
    frame_seg = {}
    pad = args.pad_frames
    for k, s in enumerate(segs):
        for i in range(max(0, s.start - pad), s.end + pad + 1):
            frame_seg.setdefault(i, k)
    if args.mask == "glyph":
        t0 = time.time()
        masks = glyph_masks(src, W, H, band, segs, args.dilate, args.grow, args.shadow)
        log(f"  glyph masks done ({time.time() - t0:.0f}s)")
    else:
        masks = {}

    # Pass 2: inpaint + encode
    t0 = time.time()
    dst = out_dir / (src.stem + "_clean.mp4")
    if not masks:
        masks = {k: seg_mask(s, H, W, args.dilate) for k, s in enumerate(segs)}
    filler = None
    if args.temporal > 0 and segs:
        rows = np.nonzero(np.any([m.any(1) for m in masks.values()], 0))[0]
        r0 = max(0, (rows.min() - 16) // 2 * 2)
        r1 = min(H, (rows.max() + 18) // 2 * 2)
        filler = TemporalFiller(src, W, H, (int(r0), int(r1)), frame_seg, masks, radius=args.temporal)
    wr = open_writer(src, dst, W, H, fps, encoder, args.crf)
    try:
        for i, frame in enumerate(read_frames(src, W, H)):
            k = frame_seg.get(i)
            if k is not None:
                if filler:
                    frame = filler.fill(i, frame, masks[k], lama)
                else:
                    frame = inpaint_packed(frame, masks[k], lama)
            elif filler:
                filler.skip(i)
            wr.stdin.write(np.ascontiguousarray(frame).tobytes())
            if i % 100 == 0:
                el = time.time() - t0
                log(f"  [erase] {i}/{n}  {i / max(el, 1e-6):.1f} fps")
    finally:
        wr.stdin.close()
        wr.wait()
    if wr.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed (code {wr.returncode})")
    log(f"  video -> {dst}  ({time.time() - t0:.0f}s)")


def collect_inputs(inputs):
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        elif p.exists():
            files.append(p)
        else:
            log(f"Skipping missing path: {p}")
    return files


def build_argparser():
    ap = argparse.ArgumentParser(prog="cleanframe", description="Clean Frame: remove hardcoded subtitles and extract them as SRT")
    ap.add_argument("inputs", nargs="+", help="video files or directories")
    ap.add_argument("-o", "--out", default="output", help="output directory (default: ./output)")
    ap.add_argument("--band", default="0.70,1.0",
                    help="horizontal band containing subtitles, as fractions or pixels, e.g. 0.7,1.0 or 800,1080 "
                         "(default: bottom 30%%)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "dml", "cpu"])
    ap.add_argument("--srt-only", action="store_true", help="extract subtitles only, do not erase")
    ap.add_argument("--no-cache", action="store_true", help="ignore the OCR cache and run OCR again")
    ap.add_argument("--ocr-interval", type=int, default=10,
                    help="max frames to skip OCR while the subtitle band is unchanged (1 = OCR every frame)")
    ap.add_argument("--encoder", default="auto", help="auto / h264_nvenc / libx264 / hevc_nvenc ...")
    ap.add_argument("--crf", type=int, default=18, help="quality, lower is better (default: 18)")
    ap.add_argument("--mask", default="glyph", choices=["glyph", "box"],
                    help="glyph: erase only the strokes, sharper (default); box: erase whole text boxes")
    ap.add_argument("--temporal", type=int, default=0,
                    help="experimental: borrow real background pixels from +/-N frames (default 0 = off, LaMa only)")
    ap.add_argument("--dilate", type=int, default=10, help="text box dilation in pixels")
    ap.add_argument("--grow", type=int, default=8, help="glyph mask dilation in pixels (covers the outline)")
    ap.add_argument("--shadow", type=int, default=3, help="extra glyph mask dilation towards the lower right (covers the shadow)")
    ap.add_argument("--pad-frames", type=int, default=1, help="extra frames erased before/after each subtitle (fade in/out)")
    ap.add_argument("--max-gap", type=int, default=5, help="missed frames tolerated within one subtitle")
    ap.add_argument("--min-dur", type=float, default=0.4, help="subtitles shorter than this many seconds are treated as noise")
    ap.add_argument("--min-score", type=float, default=0.6, help="OCR confidence threshold")
    ap.add_argument("--min-height", type=float, default=0.015, help="minimum text height as a fraction of frame height")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    files = collect_inputs(args.inputs)
    if not files:
        sys.exit("No video files found")
    device, prov = providers_for(args.device)
    log(f"device: {device}  providers={prov}")
    ocr = OCR(device)
    lama = None if args.srt_only else Lama(prov)
    encoder = None if args.srt_only else pick_encoder(args.encoder)
    if encoder:
        log(f"encoder: {encoder}")
    out_dir = Path(args.out)
    failed = []
    for f in files:
        try:
            process(f, out_dir, args, ocr, lama, encoder)
        except Exception as e:
            log(f"!! failed {f}: {e}")
            failed.append(f)
    log(f"\ndone {len(files) - len(failed)}/{len(files)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
