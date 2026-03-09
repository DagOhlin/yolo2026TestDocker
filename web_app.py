from flask import Flask, Response, jsonify, request, send_from_directory
import cv2
import time
import logging
import threading
from ultralytics import YOLO
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

app = Flask(__name__, static_folder='static')

logging.info("Loading OpenVINO model...")
model = YOLO("yolo26n-seg_openvino_model/")

# ── COCO class names (80 classes) ──
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush"
}

# Categories for the UI
CLASS_CATEGORIES = {
    "People": [0],
    "Vehicles": [1, 2, 3, 4, 5, 6, 7, 8],
    "Animals": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "Accessories": [24, 25, 26, 27, 28],
    "Sports": [29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
    "Kitchen": [39, 40, 41, 42, 43, 44, 45],
    "Food": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
    "Furniture": [56, 57, 58, 59, 60, 61],
    "Electronics": [62, 63, 64, 65, 66, 67],
    "Appliances": [68, 69, 70, 71, 72],
    "Other": [73, 74, 75, 76, 77, 78, 79]
}

# ── Shared mutable state ──
_raw = os.getenv('CAMERA01PASSWORD', '')
RTSP_URL = _raw.strip() if _raw else None
WEBCAM_INDEX = 0

# Default to RTSP if available (webcam may not be connected)
_default_source = "rtsp" if RTSP_URL else "webcam"

state_lock = threading.Lock()
state = {
    "source": _default_source,
    "active_classes": [0, 14], # default: person + bird
    "fps": 0,
    "latency": 0,
    "switch_requested": False, # flag to signal the stream loop to reconnect
}

cap = None  # managed inside generate_frames


def open_capture(src):
    """Open a VideoCapture with the right backend and settings."""
    if isinstance(src, str) and src.startswith("rtsp"):
        c = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        c = cv2.VideoCapture(src)
    return c


def get_capture_source():
    """Return the correct capture argument based on current state."""
    if state["source"] == "rtsp" and RTSP_URL:
        return RTSP_URL
    return WEBCAM_INDEX


def generate_frames():
    global cap
    src = get_capture_source()
    cap = open_capture(src)
    prev_time = 0

    while True:
        # Check if a source switch was requested
        with state_lock:
            switch = state["switch_requested"]
            if switch:
                state["switch_requested"] = False

        if switch:
            # Don't call cap.release() — FFmpeg's pthread_frame can
            # abort the process on release. Just drop the reference.
            cap = None
            time.sleep(1.0)  # let FFmpeg threads wind down
            src = get_capture_source()
            cap = open_capture(src)
            prev_time = 0
            logging.info(f"Switched camera source to: {state['source']} ({src})")

        if cap is None or not cap.isOpened():
            logging.error("Camera not available. Retrying in 5s...")
            time.sleep(5)
            src = get_capture_source()
            cap = open_capture(src)
            continue

        success, frame = cap.read()
        if not success:
            logging.warning("Failed to grab frame. Reconnecting in 5s...")
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            time.sleep(5)
            src = get_capture_source()
            cap = open_capture(src)
            continue

        # Get active classes (thread-safe read)
        with state_lock:
            classes = list(state["active_classes"]) if state["active_classes"] else None

        # Run YOLO inference
        results = model(frame, imgsz=640, device="cpu", classes=classes, verbose=False)

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        # Latency
        inference_latency = results[0].speed['inference']

        # Update shared metrics
        with state_lock:
            state["fps"] = round(fps)
            state["latency"] = round(inference_latency, 1)

        # Annotate
        annotated_frame = results[0].plot()
        monitor_text = f"FPS: {int(fps)} | Latency: {inference_latency:.1f}ms"
        cv2.putText(annotated_frame, monitor_text, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ── Routes ──

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/sources', methods=['GET'])
def get_sources():
    sources = ["webcam"]
    if RTSP_URL:
        sources.append("rtsp")
    with state_lock:
        current = state["source"]
    return jsonify({"sources": sources, "active": current})


@app.route('/api/source', methods=['POST'])
def set_source():
    data = request.get_json(force=True)
    new_source = data.get("source", "").lower()
    if new_source not in ("webcam", "rtsp"):
        return jsonify({"error": "Invalid source"}), 400
    if new_source == "rtsp" and not RTSP_URL:
        return jsonify({"error": "No RTSP URL configured"}), 400

    with state_lock:
        if state["source"] != new_source:
            state["source"] = new_source
            state["switch_requested"] = True

    return jsonify({"source": new_source})


@app.route('/api/classes', methods=['GET'])
def get_classes():
    with state_lock:
        active = list(state["active_classes"])
    return jsonify({
        "all_classes": COCO_CLASSES,
        "categories": CLASS_CATEGORIES,
        "active": active
    })


@app.route('/api/classes', methods=['POST'])
def set_classes():
    data = request.get_json(force=True)
    class_ids = data.get("classes", [])
    # Validate
    valid = [int(c) for c in class_ids if int(c) in COCO_CLASSES]
    with state_lock:
        state["active_classes"] = valid if valid else None
    return jsonify({"active": valid})


@app.route('/api/status', methods=['GET'])
def get_status():
    with state_lock:
        return jsonify({
            "source": state["source"],
            "fps": state["fps"],
            "latency": state["latency"],
            "active_classes": list(state["active_classes"]) if state["active_classes"] else []
        })


if __name__ == "__main__":
    logging.info("Starting Web Server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)