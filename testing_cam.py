import cv2
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# --- INLINE HTML TEMPLATE ---
# The HTML and CSS are stored as a Python string and rendered by Flask.
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jetson Live Stream</title>
    <style>
        body { 
            text-align: center; 
            font-family: Arial, sans-serif; 
            background-color: #222; 
            color: white;
            padding-top: 40px; 
        }
        img { 
            border: 2px solid #555; 
            border-radius: 8px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.5); 
            max-width: 100%; 
            height: auto; 
        }
    </style>
</head>
<body>
    <h1>Jetson /dev/video0 Stream</h1>
    <img src="{{ url_for('video_feed') }}" alt="Live Video Feed">
</body>
</html>
"""

def generate_frames():
    # Initialize the camera using V4L2 backend for Jetson compatibility
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # Set resolution to reduce latency
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print("Error: Could not open video device /dev/video0")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield the output frame in byte format required for MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
    camera.release()

@app.route('/')
def index():
    # Render the inline HTML string instead of an external file
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    # Return the multipart response for the video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # host='0.0.0.0' exposes the server to your local network
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
