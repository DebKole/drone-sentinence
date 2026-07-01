import subprocess
import time
import sys
import os

def get_next_run_dir(base_dir="."):
    """Finds the next available run directory name (run1, run2, etc.)."""
    counter = 1
    while True:
        dir_name = os.path.join(base_dir, f"run{counter}")
        if not os.path.exists(dir_name):
            return dir_name
        counter += 1

def capture_images(interval=2):
    # Determine and create the new folder for this specific run
    run_dir = get_next_run_dir()
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Created new session folder: {run_dir}")
    print(f"Starting camera capture on /dev/video0 every {interval} seconds...")
    print("Press Ctrl+C to stop.")
    
    counter = 1
    try:
        while True:
            # Generate the sequential filename INSIDE the new run directory
            filename = os.path.join(run_dir, f"cap{counter}.jpg")
            
            # The ffmpeg command
            command = [
                "ffmpeg",
                "-y",
                "-fflags", "nobuffer",
                "-f", "v4l2",
                "-input_format", "mjpeg",
                "-video_size", "800x600",
                "-i", "/dev/video0",
                "-frames:v", "1",
                filename
            ]
            
            # Execute the command silently
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"[{time.strftime('%X')}] Saved: {filename}")
            
            counter += 1
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\nCapture stopped by user. All images saved in: ./{run_dir}/")
        sys.exit(0)

if __name__ == "__main__":
    capture_images()
