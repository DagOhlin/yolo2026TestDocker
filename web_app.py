from flask import Flask, Response
import cv2
import time
import logging
from ultralytics import YOLO
import os

# 1. Setup proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
app = Flask(__name__)

logging.info("Loading OpenVINO model...")
model = YOLO("yolo26n-seg_openvino_model/")

#RTSP_URL = os.getenv('CAMERA01PASSWORD')
RTSP_URL = 0

def generate_frames():
    # Initial connection attempt
    cap = cv2.VideoCapture(RTSP_URL)
    prev_time = 0
    
    while True:
        # Check if the camera is opened properly
        if not cap.isOpened():
            logging.error("Camera connection lost or cannot be opened. Retrying in 10 seconds...")
            time.sleep(10)
            # Re-initialize the capture object
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        success, frame = cap.read()
        
        # If the stream drops while capturing, success will be False
        if not success:
            logging.warning("Failed to grab frame from stream. Reconnecting in 10 seconds...")
            # Explicitly release the capture object to prevent memory/resource leaks in Docker
            cap.release()
            time.sleep(10)
            # Re-initialize the capture object
            cap = cv2.VideoCapture(RTSP_URL)
            continue
            
        # 1. Run YOLO inference on your Intel GPU
        results = model(frame, imgsz=640, device="cpu", classes=[0, 14], verbose=False)
        
        # 2. Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        # 3. Get Latency directly from YOLO's built-in speed tracker
        inference_latency = results[0].speed['inference']
        
        # 4. Draw the colorful masks and bounding boxes on the frame
        annotated_frame = results[0].plot()
        
        # 5. Stamp the Performance Monitor onto the top-left of the frame
        monitor_text = f"FPS: {int(fps)} | Latency: {inference_latency:.1f}ms"
        cv2.putText(annotated_frame, monitor_text, (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 6. Encode and yield the frame to the web browser
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            logging.error("Failed to encode frame.")
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
        <head><title>YOLO Web Stream</title></head>
        <body style="background-color: #222; color: white; text-align: center; font-family: sans-serif;">
            <h1>Real-Time AI Performance Monitor</h1>
            <img src="/video_feed" style="border: 5px solid #555; border-radius: 10px; width: 640px;"/>
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    logging.info("Starting Web Server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)