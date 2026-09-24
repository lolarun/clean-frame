"""ffmpeg helpers: probing, frame decoding and encoding."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from common import ROOT

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".m4v", ".wmv", ".webm"}


def _exe(name):
    local = ROOT / "ffmpeg" / (name + (".exe" if os.name == "nt" else ""))
    if local.exists():
        return str(local)
    p = shutil.which(name)
    if not p:
        sys.exit(f"{name} not found: install ffmpeg and add it to PATH, or put it in {ROOT / 'ffmpeg'}")
    return p


def probe(path):
    """-> (width, height, fps, approximate frame count)"""
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
    """Yield frames as read-only BGR ndarrays. With crop=(y0, y1) only that horizontal band is decoded."""
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
    """ffmpeg process that encodes raw BGR frames from stdin and copies the audio of `src`."""
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        venc = ["-c:v", encoder, "-preset", "p5", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        venc = ["-c:v", encoder, "-preset", "medium", "-crf", str(crf)]
    cmd = [_exe("ffmpeg"), "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
           "-i", str(src), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy",
           *venc, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
