#!/usr/bin/env python3
"""Build a 1080x1920 Reel from the 16:9 "Leadership-Vortrag Change" video.

Usage: python3 build_reel.py SOURCE.mp4 TRANSCRIPT.json OUT_DIR

- Speaker shots: 9:16 crop that follows the speaker's face (smoothed).
- Text slides / end cards: full width over a blurred, darkened fill.
- Burned-in word-by-word captions (ASS, Inter Black, active word in gold).
"""
import json, subprocess, sys, os
import numpy as np, cv2

SRC, TRANSCRIPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
W, H = 1920, 1080          # source
OW, OH = 1080, 1920        # output
FPS = 30000 / 1001

# (src_in, src_out, mode, opts)  -- times in seconds of the source video
PIECES = [
    # Hook: closing sites
    (49.64, 52.12, "crop", {}),
    (79.60, 86.00, "crop", {"h": 960}),
    # "Fakt: wenn wir uns nicht verändern ..."
    (32.20, 40.34, "crop", {"h": 830, "max_bottom": 830}),   # hide lower-third banner
    (40.34, 42.05, "crop", {"h": 860}),
    (42.05, 43.20, "fit", {}),                                # slide "Wer nicht transformiert ..."
    # Guiding principle
    (86.56, 93.30, "crop", {}),
    (93.30, 101.30, "fit", {}),                               # slide "Nichts bleibt bestehen ..."
    (101.30, 110.28, "crop", {"h": 930, "max_right": 1640, "max_bottom": 930}),  # keep social icons + handle out
    (110.28, 113.88, "crop", {"h": 860, "max_right": 1640, "max_bottom": 930}),
    (113.88, 118.05, "crop", {"fixed_cx": 960, "h": 800, "y0": 140}),  # audience (skip letterbox bars)
    (118.05, 125.00, "crop", {}),
    # End card
    (125.03, 132.60, "fit", {"endcard": True}),
]


def decode(t0, t1, scale=1):
    w, h = W // scale, H // scale
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", SRC, "-t", f"{t1 - t0:.3f}",
           "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.wait()


CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


def track_faces(t0, t1):
    """Return per-frame (cx, cy, fh) in source pixels, smoothed."""
    pts = []
    for i, f in enumerate(decode(t0, t1, scale=2)):
        if i % 2:
            pts.append(None); continue
        g = cv2.equalizeHist(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        faces = list(CASCADE.detectMultiScale(g, 1.1, 6, minSize=(28, 28)))
        if not faces:
            faces = list(PROFILE.detectMultiScale(g, 1.1, 6, minSize=(28, 28)))
            faces += [(g.shape[1] - x - w, y, w, h) for x, y, w, h in
                      PROFILE.detectMultiScale(cv2.flip(g, 1), 1.1, 6, minSize=(28, 28))]
        if faces:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            pts.append(((x + w / 2) * 2, (y + h / 2) * 2, h * 2))
        else:
            pts.append(None)
    n = len(pts)
    good = [i for i, p in enumerate(pts) if p]
    if not good:
        return None, n
    arr = np.array([pts[i] for i in good], float)
    # reject outliers (false positives on slides / audience)
    med = np.median(arr[:, 0])
    keep = np.abs(arr[:, 0] - med) < 350
    good = [g for g, k in zip(good, keep) if k]
    arr = arr[keep]
    out = np.stack([np.interp(np.arange(n), good, arr[:, k]) for k in range(3)], 1)
    # heavy temporal smoothing (~0.8 s) -> calm virtual camera
    sig = 0.8 * FPS
    k = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2); k /= k.sum()
    pad = len(k) // 2
    sm = np.stack([np.convolve(np.pad(out[:, j], pad, mode="edge"), k, "valid") for j in range(3)], 1)
    return sm[:n], n


def crop_frame(f, cx, cy, opts):
    ch = int(opts.get("h", H))
    cw = int(round(ch * 9 / 16))
    x0 = int(round(cx - cw / 2))
    x0 = max(0, min(x0, opts.get("max_right", W) - cw))
    y0 = opts["y0"] if "y0" in opts else int(round(cy - 0.30 * ch))
    y0 = max(0, min(y0, opts.get("max_bottom", H) - ch))
    c = f[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(c, (OW, OH), interpolation=cv2.INTER_LANCZOS4 if ch < H else cv2.INTER_CUBIC)


def fit_frame(f):
    cw = int(H * 9 / 16)
    bg = f[:, (W - cw) // 2:(W + cw) // 2]
    bg = cv2.resize(bg, (108, 192), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(bg, (0, 0), 6)
    bg = cv2.resize(bg, (OW, OH), interpolation=cv2.INTER_CUBIC)
    bg = (bg.astype(np.float32) * 0.40).astype(np.uint8)
    fg = cv2.resize(f, (OW, int(OW * H / W)), interpolation=cv2.INTER_AREA)
    y = (OH - fg.shape[0]) // 2 - 60
    bg[y:y + fg.shape[0]] = fg
    return bg


def render_video(path):
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
         "-r", "30000/1001", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
         "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    total = 0
    for t0, t1, mode, opts in PIECES:
        if mode == "crop":
            if "fixed_cx" in opts:
                track = None
            else:
                track, _ = track_faces(t0, t1)
            for i, f in enumerate(decode(t0, t1)):
                if track is not None:
                    cx, cy, _ = track[min(i, len(track) - 1)]
                else:
                    cx, cy = opts.get("fixed_cx", W / 2), H * 0.35
                enc.stdin.write(crop_frame(f, cx, cy, opts).tobytes()); total += 1
        else:
            for f in decode(t0, t1):
                enc.stdin.write(fit_frame(f).tobytes()); total += 1
        print(f"  piece {t0:.2f}-{t1:.2f} {mode}: frames so far {total}", flush=True)
    enc.stdin.close(); enc.wait()
    return total


def render_audio(path):
    parts, fl = [], []
    for i, (t0, t1, _, opts) in enumerate(PIECES):
        d = t1 - t0
        fo = 1.2 if opts.get("endcard") else 0.02
        fl.append(f"[0:a]atrim={t0:.3f}:{t1:.3f},asetpts=PTS-STARTPTS,"
                  f"afade=t=in:d=0.02,afade=t=out:st={d - fo:.3f}:d={fo}[a{i}]")
        parts.append(f"[a{i}]")
    fl.append("".join(parts) + f"concat=n={len(PIECES)}:v=0:a=1,"
              "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[out]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-filter_complex", ";".join(fl),
                    "-map", "[out]", "-c:a", "aac", "-b:a", "192k", path], check=True)


# ---------------------------------------------------------------- captions
def ass_time(t):
    t = max(0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def build_captions(path):
    words = []
    for s in json.load(open(TRANSCRIPT)):
        for w in s["words"]:
            t = w["w"].strip()
            if t == "in" and 99.7 < w["s"] < 100.0:   # ASR: "rechtfertigt den derzeitigen"
                t = "den"
            words.append((w["s"], w["e"], t))
    # map source time -> reel time
    off, mapped = 0.0, []
    for t0, t1, mode, opts in PIECES:
        if not opts.get("endcard"):
            for s, e, w in words:
                mid = (s + e) / 2
                if t0 <= mid < t1:
                    mapped.append([max(s, t0) - t0 + off, min(e, t1) - t0 + off, w])
        off += t1 - t0
    # chunk into short lines
    chunks, cur = [], []
    for wd in mapped:
        cur.append(wd)
        text = " ".join(x[2] for x in cur)
        if len(text) >= 16 or wd[2][-1] in ",.?!" or len(cur) >= 4:
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)

    GOLD = "&H0000B7F5&"   # BGR of #F5B700
    lines = []
    for ci, ch in enumerate(chunks):
        start = ch[0][0]
        end = ch[-1][1] + 0.25
        if ci + 1 < len(chunks):
            end = min(end, chunks[ci + 1][0][0])
        for wi, (ws, we, _) in enumerate(ch):
            seg_s = ws if wi else start
            seg_e = ch[wi + 1][0] if wi + 1 < len(ch) else end
            if seg_e - seg_s < 0.01:
                continue
            txt = " ".join(("{\\c" + GOLD + "}" + w[2] + "{\\c&H00FFFFFF&}") if j == wi else w[2]
                           for j, w in enumerate(ch))
            lines.append(f"Dialogue: 0,{ass_time(seg_s)},{ass_time(seg_e)},Cap,,0,0,0,,{txt}")

    hook = ("Dialogue: 1,0:00:00.00,0:00:03.60,Hook,,0,0,0,,"
            "{\\fad(150,250)}Wer nicht {\\c" + GOLD + "}transformiert{\\c&H00FFFFFF&},\\N"
            "muss restrukturieren.")
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OW}
PlayResY: {OH}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Inter Black,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,7,4,2,70,70,520,1
Style: Hook,Inter Black,78,&H00FFFFFF,&H00FFFFFF,&H50000000,&HA0000000,-1,0,0,0,100,100,0,0,3,22,0,8,60,60,190,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(path, "w") as f:
        f.write(header + hook + "\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    v, a, s = f"{OUT}/video.mp4", f"{OUT}/audio.m4a", f"{OUT}/captions.ass"
    build_captions(s)
    render_audio(a)
    n = render_video(v)
    print("frames:", n)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v, "-i", a,
                    "-vf", f"ass={s}", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
                    "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", "30000/1001",
                    "-c:a", "copy", "-shortest", "-movflags", "+faststart",
                    f"{OUT}/reel.mp4"], check=True)
    print("done:", f"{OUT}/reel.mp4")
