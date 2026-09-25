# Memo: ProPainter defect analysis

Investigation notes for defects found in the ProPainter output of the test clip
(`燕赤霞传之情迷兰若寺.mp4`, 1920×1080, 25 fps, 7515 frames). Dates are 2026-09-24.

---

## 1. Dark, flickering blobs at ~0:25

### Symptom

Around 0:24–0:27 (subtitle "能杀我的妖 还没出生呢", frames 621–685) the ProPainter output shows
dark, irregular blobs with straight edges where the subtitle used to be. Their shape changes every
frame, so they read as flicker. The LaMa output of the same frames is clean.

### Hypotheses and experiments

All experiments re-ran ProPainter on this subtitle only (band rows 816–1024 unless noted) and
compared frames 630 / 640 / 650.

| # | Hypothesis | Experiment | Result |
|---|---|---|---|
| 1 | fp16 overflow / precision | same chunk (frames 590–700) in fp16 vs fp32 | identical blobs → rejected |
| 2 | black letterbox dragged into the fill (bar starts at row 975) | band 816–968, letterbox excluded | blobs remain, different shape → rejected |
| 3 | subtitle shadow not covered by the mask | overlay of the glyph mask on frame 640 | mask fully covers glyphs and shadow → rejected |
| 4 | mask too small | glyph mask grow 14 / shadow 6, and full box mask | blobs get **larger**, with dark-red colour → the fill is copying content from elsewhere |
| 5 | too little context for flow completion | taller bands: 208 / 464 / 624 px, frames 605–675 | **all clean, including the 208 px reference** → the frame range matters, not the band height |
| 6 | a specific frame range is the source | frames 590–675 vs 605–700, band 208 px | 590–675 clean, 605–700 blobs → source is in frames 676–700 |

Side result of #5: peak VRAM grows with band height (7.0 / 15.6 / 20.3 GB for 71 frames at
208 / 464 / 624 px), so taller bands are expensive.

### Root cause

Frame-difference analysis found shot cuts at frames **607, 666 and 695**. The subtitle spans three
shots:

| Frames | Shot |
|---|---|
| 607–665 | foggy wide shot (where the blobs appear) |
| 666–694 | close-up with a large dark robe exactly where the subtitle is |
| 695– | red-robe close-up |

ProPainter treats every frame of a chunk as one continuous shot and propagates pixels between
them along the completed optical flow. The chunk used in the full run crossed these cuts, so dark
robe pixels from the close-ups were copied into the masked area of the foggy shot. The low-texture
fog gives almost no reliable flow of its own, which makes the wrong propagation dominate. The more
frames of the other shots a chunk contains, the worse it gets, which matches experiments 5 and 6.

### Fix

- `video.detect_cuts()`: shot cut detection on 320×180 grayscale frames. A cut is a spike in the
  mean absolute frame difference: above 12 **and** 3× the median of the 5 differences on each side.
  A plain threshold found 345 "cuts" in the 5-minute clip (fire, flashes and fast motion in the
  intro); the spike rule finds 123, 5–8 per 30 s in the dialogue part, and keeps 607 / 666 / 695.
  Runs in ~11 s for the clip.
- `ProPainterBackend._chunks()`: runs of subtitle frames are split at cuts, and padding and context
  frames are clamped to the shot. Result on the clip: 75 chunks (was 58), none crossing a cut, all
  subtitle frames covered, 5293 frames processed (was 4792).
- One-frame shots are handled by duplicating the frame (flow needs two frames).
- The pipeline only runs cut detection for backends with `uses_cuts = True` (ProPainter).

Verified offline (cut positions, chunk layout, streaming order with a fake engine) and with a full
ProPainter run on the A10 (OCR from cache): 123 cuts found in 16 s, 75 chunks, erase + encode
2139 s. The blobs at 25.2 / 25.6 / 26.0 / 26.4 s are gone, and the check points 1:53, 2:30, 2:56
and 4:05 are unchanged compared with the previous output.

---

## 2. Subtitle left unerased at 2:56 (earlier, fixed in `ec57a9e`)

### Symptom

At 2:56 most of "比如说我们有一个新产品" stayed visible in the ProPainter output; only the middle
part was erased.

### Root cause

The erased part matched the previous subtitle "你看" (two centred characters). With
`--pad-frames 1`, frame 4399 belonged to both the padding of "你看" and the start of the new
subtitle, but only received the earlier subtitle's mask. The new subtitle was therefore unmasked on
that frame; LaMa left a one-frame flash, and ProPainter treated the text as real background and
propagated it into the following frames. 14 of 99 subtitle transitions in the clip are back to back.

### Fix

`masks.frame_masks()` gives such frames the union of all covering subtitles' masks. The delivered
ProPainter file was repaired by re-running only the 11 affected chunks and compositing them onto the
original (`*_clean_pp_fixed.mp4`).

---

## Lessons

- Video inpainting must never mix shots: detect cuts before planning chunks.
- Any frame that is not masked is treated as ground truth by ProPainter and can be propagated far;
  masking errors that are invisible with LaMa (one frame) become obvious with ProPainter.
- Reproduce on a short range first, then bisect the frame range; it located the source quickly after
  the parameter-based hypotheses failed.
