import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import argparse
import threading
import time
import urllib.request
import sys

# ---------------------------------------------------------
# 1. Setup the Siamese Feature Extractor
# We use a pre-trained MobileNetV2 as our backbone because 
# it's very fast (can run on drone) and extracts great features.
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Siamese Tracker")
    p.add_argument("--camera",   type=int,   default=0)
    p.add_argument("--ip_cam",   default=None, help="MJPEG stream URL")
    p.add_argument("--shot_jpg", default=None, help="shot.jpg polling URL")
    p.add_argument("--weights",  default=None, help="Path to custom .pth weights")
    return p.parse_args()

args = parse_args()

print("Loading model...")
if args.weights:
    print(f"Loading custom weights: {args.weights}")
    model = mobilenet_v2(weights=None).features
    
    # Robustly load state dict (handles different save formats)
    state_dict = torch.load(args.weights, map_location='cpu')
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
        
    # Strip common prefixes if they exist (e.g., if saved as part of a larger Siamese class)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("features.", "").replace("backbone.", "").replace("module.", "")
        new_state_dict[new_k] = v
        
    model.load_state_dict(new_state_dict, strict=False)
else:
    print("Loading default ImageNet weights...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Transforms to prepare the image for the neural network
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_feature_map(img_tensor):
    """Extracts the deep feature map from an image."""
    with torch.no_grad():
        return model(img_tensor)

# ---------------------------------------------------------
# 2. Setup Camera and Variables
# ---------------------------------------------------------
SHOT_TIMEOUT = 2.0

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

# Argparse is now at the top of the file

def open_camera(args):
    if args.shot_jpg: return ShotJpegCamera(args.shot_jpg)
    if args.ip_cam:   return MJPEGCamera(args.ip_cam)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened(): 
        print(f"[ERROR] Cannot open webcam {args.camera}")
        sys.exit(1)
    return cap

# args is parsed at the top
cap = open_camera(args)
reference_features = None
ref_img_display = None
box_size = 128 # The size required by your rules (128x128)

print("\n--- INSTRUCTIONS ---")
print("1. Point camera at an object.")
print("2. Press 'r' to capture the center square as your 'LR Reference'.")
print("3. Move the camera around. The network will try to find the reference in the frame.")
print("4. Press 'q' to quit.")
print("--------------------\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # Resize frame for faster processing (simulate HD downsampled to something manageable)
    # We will use 640x480 for the camera feed
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    
    # Calculate center box coordinates for capturing reference
    x1 = w//2 - box_size//2
    y1 = h//2 - box_size//2
    x2 = x1 + box_size
    y2 = y1 + box_size
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
        
    if key == ord('r'):
        # --- CAPTURE REFERENCE ---
        current_crop = frame[y1:y2, x1:x2].copy()
        ref_img_display = current_crop.copy()
        
        # Convert to tensor and get feature map
        ref_tensor = transform(current_crop).unsqueeze(0).to(device)
        reference_features = get_feature_map(ref_tensor)
        
        # Normalize the reference features for cosine similarity
        reference_features = F.normalize(reference_features, p=2, dim=1)
        print("Reference captured! Now searching...")

    if reference_features is None:
        # Just show the target box if we haven't captured anything yet
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Press 'r' to capture reference", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        # --- SEARCHING PHASE (Fully Convolutional Siamese Matching) ---
        # 1. Extract features for the ENTIRE current camera frame
        frame_tensor = transform(frame).unsqueeze(0).to(device)
        frame_features = get_feature_map(frame_tensor)
        frame_features = F.normalize(frame_features, p=2, dim=1)
        
        # 2. Cross-Correlation: Slide the reference features over the frame features
        # This is exactly how SiamFC works. We use conv2d where the "weights" are our reference image features.
        similarity_map = F.conv2d(frame_features, reference_features)
        
        # FIX: conv2d sums the dot products over the spatial dimensions of the filter. 
        # Since we normalized the channels, the max possible value is H*W. 
        # We must divide by the spatial area to get a true mean cosine similarity (0 to 1).
        area = reference_features.shape[2] * reference_features.shape[3]
        similarity_map = similarity_map / area
        
        # similarity_map is a 2D grid of scores. Find the max score and its location
        max_score = torch.max(similarity_map).item()
        
        if max_score > 0.65: # Threshold for a match (usually between 0.60 and 0.85)
            # Find the (y, x) coordinates of the highest score in the feature map
            max_idx = torch.argmax(similarity_map).item()
            map_h, map_w = similarity_map.shape[2:]
            max_y = max_idx // map_w
            max_x = max_idx % map_w
            
            # Map the feature map coordinates back to the original image coordinates
            # MobileNetV2 has a stride reduction of 32
            stride = 32
            
            # Calculate the bounding box for the match
            match_x1 = max_x * stride
            match_y1 = max_y * stride
            match_x2 = match_x1 + box_size
            match_y2 = match_y1 + box_size
            
            # Draw the bounding box
            cv2.rectangle(frame, (match_x1, match_y1), (match_x2, match_y2), (0, 255, 0), 3)
            cv2.putText(frame, f"Match: {max_score:.2f}", (match_x1, match_y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Searching... (Score: {max_score:.2f})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the reference image in the top-left corner
        frame[0:128, 0:128] = ref_img_display
        cv2.rectangle(frame, (0, 0), (128, 128), (255, 255, 255), 2)
        cv2.putText(frame, "Ref", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Siamese Tracker Test", frame)

cap.release()
cv2.destroyAllWindows()
