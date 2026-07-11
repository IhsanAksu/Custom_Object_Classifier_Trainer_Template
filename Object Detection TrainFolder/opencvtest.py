import cv2
import numpy as np
import mss
from mss import MSS
from flask import Flask, Response
from ultralytics import YOLO
import threading

app = Flask(__name__)
model = YOLO(r'runs\detect\my_tool_model\weights\best.pt')

# Shared frame variable
current_frame = None

def detection_loop():
    global current_frame
    SKIP_FRAMES = 1
    frame_count = 0
    
    # Use the MSS class constructor
    with MSS() as sct:
        # Check monitors safely
        monitors = sct.monitors
        # Default to monitor 1 (primary), use 2 if available
        monitor_index = 2 if len(monitors) > 2 else 1
        monitor = monitors[monitor_index]
        
        while True:
            # Capture using the instance
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            if frame_count % SKIP_FRAMES == 0:
                results = model.predict(frame, imgsz=640, conf=0.07, verbose=False)
                # Plot results and store for the web server
                current_frame = results[0].plot()
            
            frame_count += 1

def generate():
    global current_frame
    while True:
        if current_frame is not None:
            # Encode frame as JPEG for streaming
            _, buffer = cv2.imencode('.jpg', current_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start detection in background thread
    threading.Thread(target=detection_loop, daemon=True).start()
    # Run web server
    app.run(host='0.0.0.0', port=5000, threaded=True)