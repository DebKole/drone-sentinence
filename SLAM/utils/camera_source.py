"""
utils/camera_source.py

Unified camera source — handles:
  - Laptop webcam (cv2 index 0)
  - Phone IP camera (MJPEG / RTSP stream)

Usage:
    # Webcam
    cam = CameraSource(source=0)

    # Phone (install IP Webcam app on Android or EpocCam on iPhone)
    cam = CameraSource(source='http://192.168.1.5:8080/video')

    with cam:
        while True:
            ok, frame = cam.read()
"""

import cv2
import numpy as np
import os
import glob


class CameraSource:
    """
    Wraps cv2.VideoCapture with auto-reconnect and frame buffering.

    Args:
        source     : int (webcam index) or str (URL / video file path)
        width      : desired capture width  (0 = use camera default)
        height     : desired capture height (0 = use camera default)
        fps        : desired capture FPS    (0 = use camera default)
        flip       : 1 = horizontal flip (useful for front-facing webcam)
    """

    def __init__(
        self,
        source = 0,
        width:  int = 640,
        height: int = 480,
        fps:    int = 30,
        flip:   int = -1,   # -1 = no flip
    ):
        self.source = source
        self.width  = width
        self.height = height
        self.fps    = fps
        self.flip   = flip
        self._cap   = None
        self.is_dir = False
        self.is_file = False
        self.image_files = []
        self.idx = 0

    def open(self):
        """
        Open the camera / stream.

        For IP streams (phone), OpenCV ignores CAP_PROP_FRAME_WIDTH/HEIGHT —
        the stream sends whatever resolution it wants (e.g. 1920x1080).
        We always force-resize in read() to self.width x self.height so
        SLAM always sees a consistent, manageable frame size.
        """
        if isinstance(self.source, str):
            if os.path.isdir(self.source):
                self.is_dir = True
                files = []
                for ext in ('*.png', '*.jpg', '*.jpeg'):
                    files.extend(glob.glob(os.path.join(self.source, ext)))
                
                def sort_key(f):
                    basename = os.path.splitext(os.path.basename(f))[0]
                    return int(basename) if basename.isdigit() else basename
                
                self.image_files = sorted(files, key=sort_key)
                if not self.image_files:
                    raise RuntimeError(f"No images found in directory: {self.source}")
            elif self.source.startswith('http'):
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                self._cap = cv2.VideoCapture(self.source)
                self.is_file = True
        else:
            self._cap = cv2.VideoCapture(self.source)

        if self.is_dir:
            first_frame = cv2.imread(self.image_files[0])
            if first_frame is None:
                raise RuntimeError(f"Failed to read first image: {self.image_files[0]}")
            raw_h, raw_w = first_frame.shape[:2]
            print(f"[Camera] Opened directory {self.source} ({len(self.image_files)} images) -> raw {raw_w}x{raw_h}")
            
            if self.width <= 0: self.width = raw_w
            if self.height <= 0: self.height = raw_h
            
            if raw_w != self.width or raw_h != self.height:
                print(f"[Camera] Force-resizing every frame to {self.width}x{self.height}")
            return self.width, self.height

        # Try to request resolution (works for webcams, ignored by IP streams)
        if self.width  > 0: self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        if self.height > 0: self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps    > 0: self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")

        # Log actual stream resolution (may differ from requested)
        raw_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Opened {self.source} -> raw {raw_w}x{raw_h}")

        if self.width <= 0: self.width = raw_w
        if self.height <= 0: self.height = raw_h

        # We always deliver self.width x self.height to SLAM
        if raw_w != self.width or raw_h != self.height:
            print(f"[Camera] Force-resizing every frame to {self.width}x{self.height}")

        # Return the size SLAM will actually receive
        return self.width, self.height

    def read(self):
        """
        Read one frame, ALWAYS resized to self.width x self.height.
        Returns (True, frame_bgr) or (False, None).
        """
        if self.is_dir:
            if self.idx >= len(self.image_files):
                return False, None
            frame = cv2.imread(self.image_files[self.idx])
            self.idx += 1
            if frame is None:
                return False, None
            ok = True
        else:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            ok, frame = self._cap.read()
            if not ok:
                return False, None

        if self.flip >= 0:
            frame = cv2.flip(frame, self.flip)

        # Force resize — critical for IP streams that ignore resolution requests
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            if self.width > 0 and self.height > 0:
                frame = cv2.resize(frame, (self.width, self.height),
                                   interpolation=cv2.INTER_LINEAR)

        return True, frame

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None
        self.idx = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()

    @property
    def is_open(self):
        if self.is_dir:
            return True
        return self._cap is not None and self._cap.isOpened()