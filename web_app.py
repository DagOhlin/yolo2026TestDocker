from flask import Flask, Response
import cv2
import time
from ultralytics import YOLO

app = Flask(__name__)

print("Loading OpenVINO model...")
model = YOLO("yolo26n-seg_openvino_model/")

def generate_frames():
    cap = cv2.VideoCapture(0) 
    
    # Initialize variables for FPS calculation
    prev_time = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # 1. Run YOLO inference on your Intel GPU
        results = model(frame, imgsz=640, device="intel:gpu", verbose=False)
        
        # 2. Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # 3. Get Latency directly from YOLO's built-in speed tracker
        # YOLO returns a dictionary with preprocess, inference, and postprocess times in milliseconds
        inference_latency = results[0].speed['inference']
        
        # 4. Draw the colorful masks and bounding boxes on the frame
        annotated_frame = results[0].plot()
        
        # 5. Stamp the Performance Monitor onto the top-left of the frame
        # Format: "FPS: 30 | Latency: 45.2ms"
        monitor_text = f"FPS: {int(fps)} | Latency: {inference_latency:.1f}ms"
        
        # cv2.putText(image, text, (x, y), font, scale, (B, G, R_color), thickness)
        cv2.putText(annotated_frame, monitor_text, (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 6. Encode and yield the frame to the web browser
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
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
    print("Starting Web Server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)