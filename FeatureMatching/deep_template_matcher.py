"""
deep_template_matcher.py
========================
Seed-Specific Visual Similarity Detector — IRoC-U 2026
------------------------------------------------------
WHY NOT YOLO FOR THIS TASK:
  YOLOv8 pretrained on COCO detects 80 generic categories.  Fine-tuning on
  3 images only adds a new label on top — the COCO bias remains, so it fires on
  laptops, chairs, tables, etc. It has NO concept of "only detect things that
  look like my seeds."

THIS APPROACH (deep embedding similarity):
  1. Load a pretrained MobileNetV3-Small as a feature extractor (no classifier layer).
  2. Auto-annotate seed images → crop the target region.
  3. Compute a prototype embedding = mean L2-normalised embedding of all seed crops.
  4. Per live frame (in a background thread):
       • Slide a window across the frame at 3 scales.
       • Compute cosine similarity of each window to the prototype.
       • Accept the best-matching window if similarity > threshold.
  5. Display:
       LEFT  panel — seed images with annotated crops
       RIGHT panel — live feed with similarity heatmap overlay + bounding box
       + temporal EMA smoothing so the box is stable

Works for ANY seed image — fire extinguisher, yellow panel, drone pad, anything.

Usage
-----
  python deep_template_matcher.py --seeds fire_1.jpeg fire_2.jpeg fire_3.jpeg
  python deep_template_matcher.py --seeds s1.jpg s2.jpg --camera 0
  python deep_template_matcher.py --seeds s1.jpg s2.jpg --ip_cam http://IP:8080/video?x.mjpeg
  python deep_template_matcher.py --seeds s1.jpg s2.jpg --shot_jpg http://IP:8080/shot.jpg

Controls (live window)
----------------------
  +/-   Raise / lower similarity threshold
  S     Save screenshot
  Q/ESC Quit

Requirements
------------
  pip install torch torchvision opencv-python numpy
  (PyTorch is already installed in the SuperGlue environment)
"""

import argparse
import os
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ── Similarity / display config ───────────────────────────────────────────────
SIM_THRESHOLD  = 0.70    # cosine similarity threshold (0-1). Raise if too many FP.
SIM_STEP       = 0.02
SMOOTH_ALPHA   = 0.35    # EMA weight (lower = smoother box, higher = more responsive)
ABSENT_FRAMES  = 8       # frames without detection before clearing box

# Window sizes as fraction of frame short-side (multi-scale)
WINDOW_SCALES  = [0.70, 0.50, 0.35]
WINDOW_STRIDE  = 0.45    # stride as fraction of window size (overlap = 1 - stride)

INFER_W, INFER_H = 480, 360   # frame size for sliding-window extraction
SEED_PANEL_W     = 380
LIVE_PANEL_W     = 640
PANEL_H          = 480
SHOT_TIMEOUT     = 2.0


# ── Camera backends ───────────────────────────────────────────────────────────

class ShotJpegCamera:
    def __init__(self, url, fps_limit=8.0):
        self._url, self._min_dt = url, 1.0 / fps_limit
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[INFO] Connecting to {url} ...")
        for _ in range(40):
            if self._frame is not None: print("[INFO] Connected."); return
            time.sleep(0.15)
        raise RuntimeError(f"Cannot reach {url}")

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                r   = urllib.request.urlopen(self._url, timeout=SHOT_TIMEOUT)
                raw = np.frombuffer(r.read(), dtype=np.uint8)
                f   = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if f is not None:
                    with self._lock: self._frame = f
            except Exception: pass
            time.sleep(max(0, self._min_dt - (time.time()-t0)))

    def read(self):
        with self._lock: f = self._frame
        return (True, f.copy()) if f is not None else (False, None)

    def release(self): self._stop.set()


class MJPEGCamera:
    def __init__(self, url):
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened(): raise RuntimeError(f"Cannot open: {url}")
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        for _ in range(50):
            if self._frame is not None: return
            time.sleep(0.1)

    def _loop(self):
        while not self._stop.is_set():
            ret, f = self._cap.read()
            if ret and f is not None:
                with self._lock: self._frame = f
            else: time.sleep(0.05)

    def read(self):
        with self._lock: f = self._frame
        return (True, f.copy()) if f is not None else (False, None)

    def release(self): self._stop.set(); self._cap.release()


# ── Auto-annotation (same HSV + extend logic) ─────────────────────────────────

def auto_annotate_pixels(bgr):
    """
    Returns (x1,y1,x2,y2) pixel crop of the dominant object.
    Uses HSV colour saliency → extends upward (for extinguisher top hardware),
    falls back to GrabCut, then centre-crop.
    """
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Build colour masks — red/orange/yellow for common targets
    masks = []
    for lo, hi in [
        ((0,   60, 40), (15,  255, 255)),   # red (lower hue)
        ((155, 60, 40), (180, 255, 255)),   # red (upper hue)
        ((10,  90, 50), (30,  255, 255)),   # orange
        ((25, 110, 50), (45,  255, 255)),   # yellow
        ((85,  60, 40), (140, 255, 255)),   # blue/teal (for blue targets)
    ]:
        masks.append(cv2.inRange(hsv, np.array(lo), np.array(hi)))

    combined = masks[0]
    for m in masks[1:]: combined = cv2.bitwise_or(combined, m)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = None
    if contours:
        c    = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area >= h * w * 0.005:
            x, y, bw, bh = cv2.boundingRect(c)
            # Extend upward 80% to capture metal hardware above coloured body
            up   = int(bh * 0.80)
            down = int(bh * 0.12)
            side = int(bw * 0.18)
            box  = (max(0,x-side), max(0,y-up), min(w,x+bw+side), min(h,y+bh+down))

    if box is None:
        # GrabCut fallback
        mx, my = max(2, int(w*0.06)), max(2, int(h*0.06))
        rect   = (mx, my, w-2*mx, h-2*my)
        mask   = np.zeros((h,w), np.uint8)
        bgdM   = np.zeros((1,65), np.float64)
        fgdM   = np.zeros((1,65), np.float64)
        try:
            cv2.grabCut(bgr, mask, rect, bgdM, fgdM, 5, cv2.GC_INIT_WITH_RECT)
            fg = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
            ys, xs = np.where(fg > 0)
            if len(xs) > 200:
                px = int((xs.max()-xs.min())*0.05); py_ = int((ys.max()-ys.min())*0.05)
                box = (max(0,int(xs.min())-px), max(0,int(ys.min())-py_),
                       min(w,int(xs.max())+px), min(h,int(ys.max())+py_))
        except Exception: pass

    if box is None:
        # Centre-crop fallback
        pad = 0.08
        box = (int(w*pad), int(h*pad), int(w*(1-pad)), int(h*(1-pad)))

    x1,y1,x2,y2 = box
    # Safety: ensure minimum 10% of frame dimension
    if (x2-x1) < w*0.05 or (y2-y1) < h*0.05:
        box = (int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95))
        x1,y1,x2,y2 = box

    return x1, y1, x2, y2


# ── Deep feature extractor ────────────────────────────────────────────────────

class FeatureExtractor:
    """
    MobileNetV3-Small pretrained on ImageNet, used as a feature backbone only
    (classifier head removed).  Outputs a 576-d L2-normalised embedding per crop.
    Runs on CPU — fast enough for drone real-time use.
    """
    INPUT_SIZE = 224

    def __init__(self):
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T
        except ImportError:
            print("[ERROR] PyTorch / torchvision not found.  "
                  "Run: pip install torch torchvision")
            sys.exit(1)

        self._torch = torch
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Remove the final classifier — keep up to the AdaptiveAvgPool
        self._model = torch.nn.Sequential(*list(net.children())[:-1])
        self._model.eval()

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def torch(self): return self._torch

    def embed(self, bgr_crop: np.ndarray) -> np.ndarray:
        """BGR crop → L2-normalised embedding (numpy float32 1-D array)."""
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        t   = self._transform(rgb).unsqueeze(0)
        with self._torch.no_grad():
            feat = self._model(t).squeeze().numpy()
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat.astype(np.float32)


# ── Seed prototype builder ────────────────────────────────────────────────────

class SeedPrototype:
    """
    Holds all seed images, their crops, and the averaged prototype embedding.
    """
    def __init__(self, extractor: FeatureExtractor):
        self._ext       = extractor
        self.seed_bgrs  = []                # full seed images (BGR)
        self.seed_crops = []                # cropped object regions (BGR)
        self.seed_boxes = []                # (x1,y1,x2,y2) in seed image coords
        self.seed_names = []
        self.prototype  = None             # mean L2-normalised embedding

    def load(self, paths: list):
        print("\n[INFO] Building seed prototype ...")
        embeddings = []
        for i, p in enumerate(paths):
            bgr = cv2.imread(str(p))
            if bgr is None:
                print(f"  [WARN] Cannot read {p}"); continue
            x1,y1,x2,y2 = auto_annotate_pixels(bgr)
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0: crop = bgr
            emb = self._ext.embed(crop)
            embeddings.append(emb)
            self.seed_bgrs.append(bgr)
            self.seed_crops.append(crop)
            self.seed_boxes.append((x1,y1,x2,y2))
            self.seed_names.append(Path(p).name)
            print(f"  Seed #{i+1} '{Path(p).name}'  crop=({x1},{y1})-({x2},{y2})")

        if not embeddings:
            raise RuntimeError("No valid seed images.")

        proto = np.mean(embeddings, axis=0)
        self.prototype = proto / (np.linalg.norm(proto) + 1e-8)
        print(f"  Prototype built from {len(embeddings)} seeds  "
              f"(embedding dim={len(self.prototype)})")

    def similarity(self, bgr_crop: np.ndarray) -> float:
        emb = self._ext.embed(bgr_crop)
        return float(np.dot(emb, self.prototype))


# ── Sliding window detector ───────────────────────────────────────────────────

class SlidingWindowDetector:
    """
    At each frame, scans the image at multiple scales using a sliding window
    and returns the window with highest cosine similarity to the seed prototype.
    """
    def __init__(self, prototype: SeedPrototype,
                 scales=WINDOW_SCALES, stride_frac=WINDOW_STRIDE):
        self._proto  = prototype
        self._scales = scales
        self._stride = stride_frac

    def detect(self, frame_bgr: np.ndarray, threshold: float):
        """
        Returns (x1,y1,x2,y2,sim) of best window, or None if below threshold.
        Also returns the similarity heatmap list [(box, sim)] for visualisation.
        """
        h, w = frame_bgr.shape[:2]
        short = min(h, w)

        best_sim  = -1.0
        best_box  = None
        heat_cells = []   # list of (x1,y1,x2,y2,sim) for heatmap

        for scale in self._scales:
            win_h = max(32, int(short * scale))
            win_w = max(32, int(short * scale))
            stride_x = max(8, int(win_w * self._stride))
            stride_y = max(8, int(win_h * self._stride))

            y = 0
            while y + win_h <= h:
                x = 0
                while x + win_w <= w:
                    crop = frame_bgr[y:y+win_h, x:x+win_w]
                    sim  = self._proto.similarity(crop)
                    heat_cells.append((x, y, x+win_w, y+win_h, sim))
                    if sim > best_sim:
                        best_sim = sim
                        best_box = (x, y, x+win_w, y+win_h)
                    x += stride_x
                y += stride_y

        if best_sim >= threshold:
            return best_box, best_sim, heat_cells
        return None, best_sim, heat_cells


# ── Temporal box smoother ─────────────────────────────────────────────────────

class BoxSmoother:
    def __init__(self, alpha=SMOOTH_ALPHA, max_absent=ABSENT_FRAMES):
        self._a   = alpha
        self._max = max_absent
        self._box = None
        self._abs = 0

    def update(self, box_sim):
        """box_sim = (x1,y1,x2,y2,sim) or None"""
        if box_sim is None:
            self._abs += 1
            if self._abs >= self._max: self._box = None
            return self._box
        self._abs = 0
        if self._box is None:
            self._box = box_sim
        else:
            a = self._a
            self._box = tuple(a*box_sim[i] + (1-a)*self._box[i] for i in range(5))
        return self._box


# ── Async inference worker ────────────────────────────────────────────────────

class InferenceWorker:
    def __init__(self, detector: SlidingWindowDetector, threshold_ref: list):
        self._det    = detector
        self._tref   = threshold_ref
        self._in     = None
        self._in_lk  = threading.Lock()
        self._in_ev  = threading.Event()
        self._out    = None
        self._out_lk = threading.Lock()
        self._stop   = threading.Event()
        self._ms     = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def push(self, frame):
        with self._in_lk: self._in = frame
        self._in_ev.set()

    def result(self):
        with self._out_lk: return self._out

    @property
    def infer_ms(self): return self._ms

    def _loop(self):
        while not self._stop.is_set():
            if not self._in_ev.wait(0.5): continue
            self._in_ev.clear()
            with self._in_lk: frame = self._in; self._in = None
            if frame is None: continue
            t0  = time.time()
            small = cv2.resize(frame, (INFER_W, INFER_H))
            box, sim, heat = self._det.detect(small, self._tref[0])
            # Scale box back to original frame coordinates
            sx = frame.shape[1] / INFER_W
            sy = frame.shape[0] / INFER_H
            if box is not None:
                box = (int(box[0]*sx), int(box[1]*sy),
                       int(box[2]*sx), int(box[3]*sy))
            # Scale heat cells too
            heat_scaled = [
                (int(c[0]*sx), int(c[1]*sy), int(c[2]*sx), int(c[3]*sy), c[4])
                for c in heat
            ]
            self._ms = (time.time()-t0)*1000
            with self._out_lk:
                self._out = dict(box=box, sim=sim, heat=heat_scaled,
                                 frame=small)

    def stop(self): self._stop.set()


# ── Display helpers ───────────────────────────────────────────────────────────

def bar_color(sim):
    if sim >= 0.80: return (0, 210, 70)
    if sim >= 0.65: return (0, 195, 255)
    return (0, 70, 255)


def txt(canvas, text, pos, color=(220,220,220), scale=0.50, th=1):
    cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, (0,0,0), th+2, cv2.LINE_AA)
    cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, th, cv2.LINE_AA)


def build_seed_panel(proto: SeedPrototype, panel_w, panel_h) -> np.ndarray:
    n = len(proto.seed_bgrs)
    if n == 0:
        p = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        txt(p, "No seeds", (10,30)); return p

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cw   = panel_w // cols
    ch   = panel_h // rows
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    for idx, (bgr, (x1,y1,x2,y2), name) in enumerate(
            zip(proto.seed_bgrs, proto.seed_boxes, proto.seed_names)):
        r, c = divmod(idx, cols)
        py0, px0 = r*ch, c*cw
        thumb = cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_AREA)

        sh, sw = bgr.shape[:2]
        tx1,ty1 = int(x1/sw*cw), int(y1/sh*ch)
        tx2,ty2 = int(x2/sw*cw), int(y2/sh*ch)
        cv2.rectangle(thumb,(tx1,ty1),(tx2,ty2),(0,210,70),2,cv2.LINE_AA)
        cv2.putText(thumb, f"#{idx+1} {Path(name).stem}", (3,16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.36, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(thumb, f"#{idx+1} {Path(name).stem}", (3,16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.36, (255,195,60), 1, cv2.LINE_AA)
        panel[py0:py0+ch, px0:px0+cw] = thumb
        cv2.rectangle(panel,(px0,py0),(px0+cw-1,py0+ch-1),(50,50,50),1)

    cv2.rectangle(panel,(0,panel_h-26),(panel_w,panel_h),(18,18,18),-1)
    txt(panel, f"SEED REFERENCE ({n} images)", (5,panel_h-8),
        color=(255,195,60), scale=0.40)
    return panel


def draw_heatmap_overlay(live_disp, heat_cells, disp_sx, disp_sy, alpha=0.30):
    """Blend a transparent similarity heat onto the live panel."""
    if not heat_cells: return live_disp
    max_sim = max(c[4] for c in heat_cells)
    min_sim = min(c[4] for c in heat_cells)
    rng = max(max_sim - min_sim, 0.01)

    overlay = live_disp.copy()
    for (x1,y1,x2,y2,sim) in heat_cells:
        t   = (sim - min_sim) / rng            # 0-1 normalised within this frame
        col = (int(255*(1-t)), int(80*t), int(255*t))  # cool-to-warm
        dx1,dy1 = int(x1*disp_sx), int(y1*disp_sy)
        dx2,dy2 = int(x2*disp_sx), int(y2*disp_sy)
        cv2.rectangle(overlay,(dx1,dy1),(dx2,dy2),col,-1)

    return cv2.addWeighted(overlay, alpha, live_disp, 1-alpha, 0)


# ── Main detection loop ───────────────────────────────────────────────────────

def open_camera(args):
    if args.shot_jpg: return ShotJpegCamera(args.shot_jpg)
    if args.ip_cam:   return MJPEGCamera(args.ip_cam)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened(): print(f"[ERROR] Cannot open webcam {args.camera}"); sys.exit(1)
    return cap


def run(args, proto: SeedPrototype):
    detector = SlidingWindowDetector(proto)
    tref     = [args.sim]
    worker   = InferenceWorker(detector, tref)
    smoother = BoxSmoother()
    cam      = open_camera(args)

    TOTAL_W = SEED_PANEL_W + 4 + LIVE_PANEL_W
    WIN = "Deep Template Matcher | IRoC-U 2026"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, TOTAL_W, PANEL_H)

    fps_t, disp_n, disp_fps = time.time(), 0, 0.0
    save_idx = 0
    last_canvas = None
    last_res    = None

    print(f"\n[CONTROLS]  +/-=Similarity threshold  S=Save  Q/ESC=Quit")
    print(f"[INFO] Similarity threshold start: {tref[0]:.2f}\n")

    while True:
        ret, frame = cam.read()
        if ret and frame is not None:
            worker.push(frame)

        res = worker.result()
        if res is not None: last_res = res

        # ── Live panel ───────────────────────────────────────────────────────
        if last_res is not None:
            src_frame  = frame if (ret and frame is not None) else last_res["frame"]
            live_disp  = cv2.resize(src_frame, (LIVE_PANEL_W, PANEL_H))
            disp_sx    = LIVE_PANEL_W / src_frame.shape[1]
            disp_sy    = PANEL_H      / src_frame.shape[0]

            # Heatmap overlay
            live_disp = draw_heatmap_overlay(
                live_disp, last_res["heat"], disp_sx, disp_sy, alpha=0.22)

            # Smooth the detection box
            raw_box = last_res["box"]
            raw_sim = last_res["sim"]
            if raw_box is not None:
                smooth = smoother.update((*raw_box, raw_sim))
            else:
                smooth = smoother.update(None)

            best_sim   = 0.0
            n_detected = 0

            if smooth is not None:
                x1s,y1s,x2s,y2s,cs = smooth
                dx1,dy1 = int(x1s*disp_sx), int(y1s*disp_sy)
                dx2,dy2 = int(x2s*disp_sx), int(y2s*disp_sy)
                col = bar_color(cs)
                cv2.rectangle(live_disp,(dx1,dy1),(dx2,dy2),col,3,cv2.LINE_AA)
                alen = 20
                for (px,py,ddx,ddy) in [
                    (dx1,dy1,1,1),(dx2,dy1,-1,1),(dx1,dy2,1,-1),(dx2,dy2,-1,-1)]:
                    cv2.line(live_disp,(px,py),(px+ddx*alen,py),col,3,cv2.LINE_AA)
                    cv2.line(live_disp,(px,py),(px,py+ddy*alen),col,3,cv2.LINE_AA)
                label = f"TARGET  sim={cs:.2f}"
                (lw,lh),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_DUPLEX,0.55,1)
                cv2.rectangle(live_disp,(dx1,dy1-lh-10),(dx1+lw+8,dy1),col,-1)
                cv2.putText(live_disp,label,(dx1+4,dy1-6),
                            cv2.FONT_HERSHEY_DUPLEX,0.55,(0,0,0),1,cv2.LINE_AA)
                best_sim   = cs
                n_detected = 1

            # Status bar
            cv2.rectangle(live_disp,(0,0),(LIVE_PANEL_W,32),(18,18,18),-1)
            fw = int(LIVE_PANEL_W * min(best_sim,1))
            cv2.rectangle(live_disp,(0,0),(fw,32),bar_color(best_sim),-1)
            status = "DETECTED" if n_detected else "SEARCHING"
            txt(live_disp,
                f"{status}  Sim:{raw_sim:.3f}  Thr:{tref[0]:.2f}  "
                f"FPS:{disp_fps:.1f}  {worker.infer_ms:.0f}ms",
                (5,22), color=(255,255,255))

            cv2.rectangle(live_disp,(0,PANEL_H-24),(LIVE_PANEL_W,PANEL_H),(18,18,18),-1)
            txt(live_disp,"[+/-]threshold  [S]save  [Q]quit",
                (5,PANEL_H-6),color=(140,140,140),scale=0.38)

            match = ("STRONG MATCH" if best_sim>=0.80 else
                     "WEAK MATCH"   if best_sim>=tref[0] else "NO MATCH")
            txt(live_disp, match, (5,PANEL_H-30), color=bar_color(best_sim),
                scale=0.68, th=2)
        else:
            live_disp = np.zeros((PANEL_H, LIVE_PANEL_W, 3), dtype=np.uint8)
            txt(live_disp, "Starting up...", (LIVE_PANEL_W//3, PANEL_H//2))

        # ── Seed panel ───────────────────────────────────────────────────────
        seed_panel = build_seed_panel(proto, SEED_PANEL_W, PANEL_H)

        # ── Compose ──────────────────────────────────────────────────────────
        gap    = np.full((PANEL_H, 4, 3), 22, dtype=np.uint8)
        canvas = np.hstack([seed_panel, gap, live_disp])
        cv2.imshow(WIN, canvas)
        last_canvas = canvas

        disp_n += 1
        elapsed = time.time() - fps_t
        if elapsed >= 1.0:
            disp_fps = disp_n / elapsed
            disp_n, fps_t = 0, time.time()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27): break
        elif key == ord("s") and last_canvas is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            sp = os.path.join(args.output_dir,
                              f"match_{time.strftime('%Y%m%d_%H%M%S')}_{save_idx:04d}.png")
            cv2.imwrite(sp, last_canvas)
            print(f"[SAVED] {sp}")
            save_idx += 1
        elif key in (ord("+"), ord("=")):
            tref[0] = min(0.99, tref[0] + SIM_STEP)
            print(f"[INFO] Similarity threshold → {tref[0]:.2f} (stricter)")
        elif key == ord("-"):
            tref[0] = max(0.30, tref[0] - SIM_STEP)
            print(f"[INFO] Similarity threshold → {tref[0]:.2f} (looser)")

    worker.stop()
    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Deep Template Matcher | IRoC-U 2026")
    p.add_argument("--seeds",      required=True, nargs="+",
                   help="Seed image paths (3-5 recommended)")
    p.add_argument("--sim",        type=float, default=SIM_THRESHOLD,
                   help=f"Cosine similarity threshold 0-1 (default {SIM_THRESHOLD})")
    p.add_argument("--camera",     type=int,   default=0)
    p.add_argument("--ip_cam",     default=None,
                   help="MJPEG stream URL")
    p.add_argument("--shot_jpg",   default=None,
                   help="shot.jpg polling URL")
    p.add_argument("--output_dir", default="./saved_matches")
    return p.parse_args()


def main():
    args  = parse_args()
    ext   = FeatureExtractor()
    proto = SeedPrototype(ext)
    proto.load(args.seeds)
    run(args, proto)


if __name__ == "__main__":
    main()
