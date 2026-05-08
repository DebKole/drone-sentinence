import argparse
import sys
import time
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import matplotlib.cm as cm

"""
SIFT Drones Matcher — robust to scale and rotation.
Multi-Seed mode: accepts 3-5 reference images of the same target and aggregates
their SIFT descriptors into one unified pool so matching is collectively stronger.
Uses traditional CV: SIFT + FLANN Lowe's ratio test + RANSAC homography.
"""

# ── Performance knobs ─────────────────────────────────────────────────────────
INFER_W, INFER_H     = 640, 480   # SIFT works well at this resolution
DISPLAY_W, DISPLAY_H = 480, 360   # Output window resolution per panel
SHOT_TIMEOUT         = 2.0

SIFT_FEATURES        = 3000
MATCH_CONF_THRES     = 0.65   # Lowe's ratio — lower = stricter
MATCH_CONF_STEP      = 0.05
RANSAC_REPROJ_THRESH = 5.0    # RANSAC reprojection tolerance in pixels
MIN_INLIERS          = 12     # Min RANSAC inliers for a valid detection

# ── Camera backends ───────────────────────────────────────────────────────────

class ShotJpegCamera:
    """ Polls /shot.jpg in a background thread """
    def __init__(self, url: str, fps_limit: float = 8.0):
        self._url    = url
        self._min_dt = 1.0 / fps_limit
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[INFO] Connecting to {url} ...")
        for _ in range(40):
            if self._frame is not None:
                print("[INFO] Camera connected.")
                return
            time.sleep(0.15)
        raise RuntimeError(f"Could not fetch frame from {url}\nCheck connection.")

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                resp  = urllib.request.urlopen(self._url, timeout=SHOT_TIMEOUT)
                raw   = np.frombuffer(resp.read(), dtype=np.uint8)
                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame = frame
            except Exception:
                pass
            time.sleep(max(0.0, self._min_dt - (time.time() - t0)))

    def read(self):
        with self._lock:
            f = self._frame
        return (True, f.copy()) if f is not None else (False, None)

    def release(self):
        self._stop.set()


class MJPEGCamera:
    """ Wraps cv2.VideoCapture in a background thread """
    def __init__(self, url: str):
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {url}")
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        for _ in range(50):
            if self._frame is not None:
                return
            time.sleep(0.1)

    def _loop(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.05)

    def read(self):
        with self._lock:
            f = self._frame
        return (True, f.copy()) if f is not None else (False, None)

    def release(self):
        self._stop.set()
        self._cap.release()

# ── Image utilities ───────────────────────────────────────────────────────────

def letterbox(bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w    = bgr.shape[:2]
    scale   = min(target_w / w, target_h / h)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off   = (target_w - new_w) // 2
    y_off   = (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas


def make_seed_mosaic(seed_bgrs: list, panel_w: int, panel_h: int) -> np.ndarray:
    """
    Tile seed images into a single display panel that fits panel_w × panel_h.
    Works for 1–16 images.
    """
    n = len(seed_bgrs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cell_w = panel_w // cols
    cell_h = panel_h // rows
    mosaic = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    for idx, bgr in enumerate(seed_bgrs):
        r, c = divmod(idx, cols)
        thumb = letterbox(bgr, cell_w, cell_h)
        thumb = cv2.cvtColor(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        y0, x0 = r * cell_h, c * cell_w
        mosaic[y0:y0+cell_h, x0:x0+cell_w] = thumb
        # Label each thumbnail with its index
        cv2.putText(mosaic, f"#{idx+1}", (x0+4, y0+14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, (255, 195, 60), 1, cv2.LINE_AA)
        cv2.rectangle(mosaic, (x0, y0), (x0+cell_w-1, y0+cell_h-1), (50, 50, 50), 1)
    return mosaic


# ── Visualization ─────────────────────────────────────────────────────────────

def ccolor(c: float):
    return (0, 210, 70) if c >= 0.6 else (0, 195, 255) if c >= 0.35 else (0, 70, 255)


def draw_result(seed_bgrs, best_seed_idx, live_bgr,
                all_kpts1, kpts0, kpts1, conf, mratio,
                avg_conf, n_matches, disp_fps, infer_ms,
                homography, seed_shape, n_seeds):
    """
    Left panel  — mosaic of ALL seed images (best one highlighted).
    Right panel — live feed with matching lines and projected bounding box.
    """
    mosaic = make_seed_mosaic(seed_bgrs, DISPLAY_W, DISPLAY_H)
    live   = letterbox(live_bgr, DISPLAY_W, DISPLAY_H)
    live   = cv2.cvtColor(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    sx = DISPLAY_W / INFER_W
    sy = DISPLAY_H / INFER_H

    GAP    = 6
    canvas = np.zeros((DISPLAY_H, DISPLAY_W * 2 + GAP, 3), dtype=np.uint8)
    canvas[:, :DISPLAY_W]         = mosaic
    canvas[:, DISPLAY_W+GAP:]     = live
    canvas[:, DISPLAY_W:DISPLAY_W+GAP] = (22, 22, 22)

    # Highlight best-matching seed in mosaic panel
    if best_seed_idx >= 0 and n_seeds > 0:
        cols = int(np.ceil(np.sqrt(n_seeds)))
        cell_w = DISPLAY_W // cols
        cell_h = DISPLAY_H // int(np.ceil(n_seeds / cols))
        br, bc = divmod(best_seed_idx, cols)
        bx0, by0 = bc * cell_w, br * cell_h
        cv2.rectangle(canvas, (bx0, by0), (bx0+cell_w-1, by0+cell_h-1), (0, 210, 70), 2)

    # Project bounding box from best seed onto live feed via homography
    if homography is not None and seed_shape is not None:
        try:
            h0, w0 = seed_shape[:2]
            corners = np.float32([[0,0],[w0,0],[w0,h0],[0,h0]]).reshape(-1,1,2)
            proj    = cv2.perspectiveTransform(corners, homography).reshape(-1,2)
            proj_sc = (proj * np.array([sx, sy])).astype(int)
            proj_sh = proj_sc + np.array([DISPLAY_W + GAP, 0])
            box_col = (0, 255, 80)
            cv2.polylines(canvas, [proj_sh.reshape(-1,1,2)], True, box_col, 2, cv2.LINE_AA)
            for pt in proj_sh:
                cv2.circle(canvas, tuple(pt), 4, box_col, -1, lineType=cv2.LINE_AA)
        except Exception:
            pass

    # Unmatched live keypoints as tiny white dots
    matched_live = set(map(tuple, kpts1.astype(int))) if len(kpts1) > 0 else set()
    for pt in all_kpts1:
        px = int(pt[0]*sx) + DISPLAY_W + GAP
        py = int(pt[1]*sy)
        cv2.circle(canvas, (px, py), 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # Matched lines — we only draw the live endpoints here (seed endpoints are in
    # the aggregate space, not directly visualisable on the mosaic).
    if len(conf) > 0:
        color_rgba = cm.jet(conf)
        color_bgr  = (color_rgba[:, :3] * 255).astype(int)[:, ::-1]
        for i in range(len(kpts1)):
            p1 = (int(kpts1[i][0]*sx) + DISPLAY_W + GAP, int(kpts1[i][1]*sy))
            c  = color_bgr[i].tolist()
            cv2.circle(canvas, p1, 3, c, -1, lineType=cv2.LINE_AA)

    def txt(text, pos, color=(220,220,220), scale=0.50, th=1):
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, (0,0,0), th+2, cv2.LINE_AA)
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, th, cv2.LINE_AA)

    bx = DISPLAY_W + GAP
    cv2.rectangle(canvas, (bx, 0), (bx + DISPLAY_W, 30), (18,18,18), -1)
    fw = int(DISPLAY_W * min(avg_conf, 1.0))
    cv2.rectangle(canvas, (bx, 0), (bx + fw, 30), ccolor(avg_conf), -1)

    best_label = f"Best:#{best_seed_idx+1}" if best_seed_idx >= 0 else "No Match"
    txt(f"Inliers:{n_matches}  Qual:{avg_conf:.3f}  FPS:{disp_fps:.1f}  {infer_ms:.0f}ms  {best_label}",
        (bx+4, 21), color=(255,255,255))

    txt(f"SEED IMAGES ({n_seeds} loaded)", (4, 20), color=(255,195,60))
    txt("LIVE FEED  (Multi-SIFT+FLANN+RANSAC)", (bx+4, 20), color=(60,195,255))
    txt(f"Lowe:{mratio:.2f}  MinInliers:{MIN_INLIERS}  [+/-]ratio  [S]save  [Q]quit",
        (bx+4, DISPLAY_H-7), color=(150,150,150), scale=0.38)

    label = "STRONG MATCH" if n_matches >= MIN_INLIERS and avg_conf > 0.55 else \
            "WEAK MATCH"   if n_matches >= MIN_INLIERS else "NO MATCH"
    lcol  = (0, 210, 70) if label == "STRONG MATCH" else \
            (0, 195, 255) if label == "WEAK MATCH" else (0, 70, 255)
    txt(label, (4, DISPLAY_H-7), color=lcol, scale=0.62, th=2)

    return canvas


# ── Multi-seed data container ─────────────────────────────────────────────────

class SeedBank:
    """
    Holds all seed images and their aggregated SIFT features.

    Strategy
    --------
    · Run SIFT on each seed image independently.
    · Concatenate all descriptors into one big array.
    · Keep a `seed_index` array that maps every descriptor row back to its
      originating seed image — used later to determine which seed image
      contributes the most inliers to a given match.
    """

    def __init__(self, sift):
        self._sift = sift
        self.seed_bgrs   = []   # list of BGR images (letterboxed)
        self.seed_grays  = []   # list of grayscale images
        self.all_kpts    = []   # flat list of cv2.KeyPoint objects
        self.seed_idx    = []   # int label per keypoint → which seed it came from
        self.all_desc    = None # (N, 128) float32 combined descriptor matrix
        self.n_seeds     = 0

    def load(self, paths: list):
        self.seed_bgrs   = []
        self.seed_grays  = []
        self.all_kpts    = []
        self.seed_idx    = []
        desc_list        = []

        for i, p in enumerate(paths):
            raw = cv2.imread(str(p))
            if raw is None:
                print(f"[WARN] Cannot read seed image: {p} — skipping.")
                continue
            bgr  = letterbox(raw, INFER_W, INFER_H)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            kpts, desc = self._sift.detectAndCompute(gray, None)
            if kpts is None or len(kpts) == 0:
                print(f"[WARN] No keypoints found in: {p} — skipping.")
                continue

            self.seed_bgrs.append(bgr)
            self.seed_grays.append(gray)
            self.all_kpts.extend(kpts)
            self.seed_idx.extend([len(self.seed_bgrs) - 1] * len(kpts))
            desc_list.append(desc)
            print(f"[INFO] Seed [{i+1}] '{p}' → {len(kpts)} SIFT keypoints.")

        self.n_seeds = len(self.seed_bgrs)
        if self.n_seeds == 0:
            raise RuntimeError("No valid seed images loaded. Check paths.")

        self.all_desc = np.vstack(desc_list).astype(np.float32)
        self.seed_idx = np.array(self.seed_idx, dtype=np.int32)
        print(f"[INFO] Seed bank ready: {self.n_seeds} images, "
              f"{len(self.all_kpts)} total keypoints, "
              f"descriptor matrix {self.all_desc.shape}.")

    def best_seed_for_inliers(self, kpts0_indices: np.ndarray) -> int:
        """Return the seed index that contributed the most inlier matches."""
        if len(kpts0_indices) == 0:
            return -1
        labels = self.seed_idx[kpts0_indices]
        counts = np.bincount(labels, minlength=self.n_seeds)
        return int(np.argmax(counts))


# ── Async inference worker ────────────────────────────────────────────────────

class InferenceWorker:
    def __init__(self, seed_bank: SeedBank, threshold_ref: list):
        self._sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)

        FLANN_INDEX_KDTREE = 1
        index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self._bank = seed_bank
        self._tref = threshold_ref

        self._in_frame  = None
        self._in_lock   = threading.Lock()
        self._in_event  = threading.Event()
        self._out       = None
        self._out_lock  = threading.Lock()
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._loop, daemon=True)
        self._infer_ms  = 0.0

    def start(self):
        self._thread.start()

    def push(self, frame_bgr: np.ndarray):
        with self._in_lock:
            self._in_frame = frame_bgr
        self._in_event.set()

    def result(self):
        with self._out_lock:
            return self._out

    @property
    def infer_ms(self):
        return self._infer_ms

    def _loop(self):
        while not self._stop.is_set():
            if not self._in_event.wait(timeout=0.5):
                continue
            self._in_event.clear()
            with self._in_lock:
                frame = self._in_frame
                self._in_frame = None
            if frame is None:
                continue
            t0 = time.time()
            try:
                r = self._infer(frame)
                self._infer_ms = (time.time() - t0) * 1000
                with self._out_lock:
                    self._out = r
            except Exception as e:
                print(f"[WARN] Inference: {e}")

    def _infer(self, frame_bgr):
        live_padded = letterbox(frame_bgr, INFER_W, INFER_H)
        live_gray   = cv2.cvtColor(live_padded, cv2.COLOR_BGR2GRAY)
        live_kpts, live_desc = self._sift.detectAndCompute(live_gray, None)

        ratio_thresh   = self._tref[0]
        kpts0_indices  = []   # index into self._bank.all_kpts
        kpts0_pts      = []   # 2D point from seed keypoint
        kpts1_pts      = []   # 2D point from live keypoint
        mconf          = []

        bank = self._bank
        n_seed = len(bank.all_kpts)

        if (bank.all_desc is not None and live_desc is not None
                and n_seed >= 2 and len(live_desc) >= 2):
            # Match: query = aggregated seed pool → train = live frame
            matches = self._matcher.knnMatch(bank.all_desc, live_desc, k=2)
            for i, m_n in enumerate(matches):
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    kpts0_indices.append(m.queryIdx)
                    kpts0_pts.append(bank.all_kpts[m.queryIdx].pt)
                    kpts1_pts.append(live_kpts[m.trainIdx].pt)
                    c = 1.0 - (m.distance / (n.distance + 1e-6))
                    c = max(0.0, min(1.0, c / 0.5))
                    mconf.append(c)

        kpts0_indices = np.array(kpts0_indices, dtype=np.int32)
        kpts0_pts  = np.array(kpts0_pts,  dtype=np.float32) if kpts0_pts  else np.zeros((0,2), np.float32)
        kpts1_pts  = np.array(kpts1_pts,  dtype=np.float32) if kpts1_pts  else np.zeros((0,2), np.float32)
        mconf      = np.array(mconf,      dtype=np.float32) if mconf      else np.zeros((0,),  np.float32)

        homography    = None
        best_seed_idx = -1

        # ── RANSAC geometric filtering ────────────────────────────────────────
        if len(kpts0_pts) >= 4:
            H, mask = cv2.findHomography(kpts0_pts, kpts1_pts,
                                         cv2.RANSAC, RANSAC_REPROJ_THRESH)
            if H is not None and mask is not None:
                d     = H[0,0]*H[1,1] - H[0,1]*H[1,0]
                scale = np.sqrt(abs(d))
                sane  = (d > 0) and (0.05 < scale < 20.0)

                # Determine which seed image the inlier keypoints come from so
                # we can project its correct bounding box corners.
                inlier_mask = mask.ravel() > 0

                # Pick the dominant seed based on inlier majority vote
                dominant_src = bank.best_seed_for_inliers(
                    kpts0_indices[inlier_mask] if inlier_mask.sum() > 0 else np.array([]))

                # Convexity + area sanity check on dominant seed's geometry
                if sane and dominant_src >= 0:
                    ref_gray = bank.seed_grays[dominant_src]
                    h0, w0   = ref_gray.shape[:2]
                    corners  = np.float32([[0,0],[w0,0],[w0,h0],[0,h0]]).reshape(-1,1,2)
                    proj     = cv2.perspectiveTransform(corners, H).reshape(-1,2)
                    area     = cv2.contourArea(proj.astype(np.float32))
                    frame_area = INFER_W * INFER_H
                    sane = (frame_area * 0.002 < area < frame_area * 0.98)

                if sane and inlier_mask.sum() >= MIN_INLIERS:
                    kpts0_indices = kpts0_indices[inlier_mask]
                    kpts0_pts     = kpts0_pts[inlier_mask]
                    kpts1_pts     = kpts1_pts[inlier_mask]
                    mconf         = mconf[inlier_mask]
                    homography    = H
                    best_seed_idx = bank.best_seed_for_inliers(kpts0_indices)
                else:
                    kpts0_pts = kpts1_pts = mconf = np.zeros((0,2)), np.zeros((0,2)), np.zeros((0,))
                    kpts0_pts, kpts1_pts, mconf = kpts0_pts[0], kpts1_pts[0], mconf[0]
            else:
                kpts0_pts, kpts1_pts, mconf = np.zeros((0,2)), np.zeros((0,2)), np.zeros((0,))
        else:
            kpts0_pts, kpts1_pts, mconf = np.zeros((0,2)), np.zeros((0,2)), np.zeros((0,))

        all_live_pts = np.array([kp.pt for kp in live_kpts]) if live_kpts else np.zeros((0,2))

        seed_shape = (bank.seed_grays[best_seed_idx].shape
                      if best_seed_idx >= 0 else
                      bank.seed_grays[0].shape)

        return dict(
            live_bgr      = live_padded,
            all_kpts1     = all_live_pts,
            kpts0         = kpts0_pts,
            kpts1         = kpts1_pts,
            conf          = mconf,
            n_matches     = len(kpts0_pts),
            avg_conf      = float(mconf.mean()) if len(mconf) > 0 else 0.0,
            homography    = homography,
            seed_shape    = seed_shape,
            best_seed_idx = best_seed_idx,
        )

    def stop(self):
        self._stop.set()


# ── Argument parsing & camera ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed SIFT drone matcher (IRoC-U 2026)")
    p.add_argument("--seeds",      required=True, nargs="+",
                   help="One or more seed image paths (3-5 recommended for competition)")
    p.add_argument("--camera",     type=int, default=0,
                   help="Webcam index (default 0)")
    p.add_argument("--ip_cam",     default=None,
                   help="MJPEG stream URL, e.g. http://IP:8080/video?x.mjpeg")
    p.add_argument("--shot_jpg",   default=None,
                   help="shot.jpg polling URL, e.g. http://IP:8080/shot.jpg")
    p.add_argument("--output_dir", default="./saved_matches")
    return p.parse_args()


def open_camera(args):
    if args.shot_jpg:
        return ShotJpegCamera(args.shot_jpg)
    if args.ip_cam:
        return MJPEGCamera(args.ip_cam)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {args.camera}")
        sys.exit(1)
    return cap


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(args):
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    tref = [MATCH_CONF_THRES]

    # Build the seed bank from all provided images
    bank = SeedBank(sift)
    bank.load(args.seeds)

    worker = InferenceWorker(bank, tref)
    worker.start()

    cam = open_camera(args)

    WIN = "Multi-Seed SIFT Matcher | IRoC-U 2026"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_W * 2 + 6, DISPLAY_H)

    save_idx    = 0
    fps_t       = time.time()
    disp_n      = 0
    disp_fps    = 0.0
    last_canvas = None

    splash = np.zeros((DISPLAY_H, DISPLAY_W * 2 + 6, 3), dtype=np.uint8)
    cv2.putText(splash, "Waiting for first result...",
                (DISPLAY_W // 3, DISPLAY_H // 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (160,160,160), 1, cv2.LINE_AA)

    print(f"\n[INFO] Running with {bank.n_seeds} seed image(s), "
          f"{len(bank.all_kpts)} total reference keypoints.")
    print("[CONTROLS]  S=Save  +/-=Lowe Ratio Threshold  Q/ESC=Quit\n")

    while True:
        ret, frame = cam.read()
        if ret and frame is not None:
            worker.push(frame)

        res = worker.result()
        if res is not None:
            canvas = draw_result(
                seed_bgrs     = bank.seed_bgrs,
                best_seed_idx = res["best_seed_idx"],
                live_bgr      = res["live_bgr"],
                all_kpts1     = res["all_kpts1"],
                kpts0         = res["kpts0"],
                kpts1         = res["kpts1"],
                conf          = res["conf"],
                mratio        = tref[0],
                avg_conf      = res["avg_conf"],
                n_matches     = res["n_matches"],
                disp_fps      = disp_fps,
                infer_ms      = worker.infer_ms,
                homography    = res["homography"],
                seed_shape    = res["seed_shape"],
                n_seeds       = bank.n_seeds,
            )
            last_canvas = canvas
            cv2.imshow(WIN, canvas)
        else:
            cv2.imshow(WIN, last_canvas if last_canvas is not None else splash)

        disp_n += 1
        elapsed = time.time() - fps_t
        if elapsed >= 1.0:
            disp_fps = disp_n / elapsed
            disp_n   = 0
            fps_t    = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s") and last_canvas is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir,
                                f"sift_match_{time.strftime('%Y%m%d_%H%M%S')}_{save_idx:04d}.png")
            cv2.imwrite(path, last_canvas)
            print(f"[SAVED] {path}")
            save_idx += 1
        elif key in (ord("+"), ord("=")):
            tref[0] = min(0.95, tref[0] + MATCH_CONF_STEP)
            print(f"[INFO] Lowe's Ratio → {tref[0]:.2f} (Looser)")
        elif key == ord("-"):
            tref[0] = max(0.1, tref[0] - MATCH_CONF_STEP)
            print(f"[INFO] Lowe's Ratio → {tref[0]:.2f} (Stricter)")

    worker.stop()
    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    run(parse_args())
