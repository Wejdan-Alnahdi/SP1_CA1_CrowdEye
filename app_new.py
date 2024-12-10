import cv2
import threading
from flask import Flask, jsonify
from ultralytics import YOLO

# Shared state for crowd statuses
crowd_status = {1: 'Not crowded', 2: 'Not crowded', 3: 'Not crowded'}
status_lock = threading.Lock()

def process_video(video_source, crowd_density, confidence, video_id):
    global crowd_status

    # Initialize YOLO model
    model = YOLO('traininResult/weights/best.pt')
    cap = cv2.VideoCapture(video_source)

    print(f"[Thread {video_id}] Starting processing for video source: {video_source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(video_source)
            continue

        results = model(frame, conf=confidence)
        people_found = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                people_found += 1

        # Determine crowd status
        with status_lock:
            crowd_status[video_id] = 'Crowded' if people_found >= crowd_density else 'Not crowded'



# Flask app for APIs
app = Flask(__name__)

@app.route('/video/<int:video_id>/status', methods=['GET'])
def get_status(video_id):
    if video_id not in crowd_status:
        return jsonify({"error": "Invalid video ID"}), 404

    with status_lock:
        status = crowd_status[video_id]
    return  status 
@app.route('/videos/statuses', methods=['GET'])
def get_all_statuses():
    """
    Returns the statuses of all videos in a single response.
    """
    with status_lock:
        statuses = crowd_status.copy()  # Copy to avoid race conditions

    return jsonify(statuses)


if __name__ == '__main__':
    # Parameters for each video thread
    video_params = [
        {"source": '4.mp4', "crowd_density": 50, "confidence": 0.3, "id": 1},
        {"source": "3.mp4", "crowd_density": 50, "confidence": 0.3, "id": 2},
        {"source": "2.mp4", "crowd_density": 50, "confidence": 0.3, "id": 3},
    ]

    # Start video processing threads
    for params in video_params:
        threading.Thread(
            target=process_video,
            args=(params["source"], params["crowd_density"], params["confidence"], params["id"]),
            daemon=True
        ).start()

    # Run Flask server
    app.run(host='0.0.0.0', port=5000)
