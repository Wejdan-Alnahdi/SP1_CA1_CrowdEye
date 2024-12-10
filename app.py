from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model
model = YOLO('traininResult/weights/best.pt')

# Define API route
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']

        # Read confidence and crowd density from the request
        cnf = request.form.get('confidence', 0.5)
        crowd_density = request.form.get('crowd_density', 100)

        # Convert to proper types
        try:
            cnf = float(cnf)
            if cnf < 0.1 or cnf > 0.9:
                cnf = 0.5
        except ValueError:
            cnf = 0.5
        
        try:
            crowd_density = int(crowd_density)
        except ValueError:
            crowd_density = 100

        # Convert image to OpenCV format
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Process the image with the YOLO model
        results = model(frame, conf=cnf)

        people_found = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                conf = round(box.conf[0].item(), 2)
                name = r.names[box.cls[0].item()]

                color = (0, 255, 0)
                x1, y1, x2, y2 = int(b[0].item()), int(b[1].item()), int(b[2].item()), int(b[3].item())
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, f'{str(name)} {str(conf)}', (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, color, 1)
                people_found += 1

        if people_found >= crowd_density:
            text_p = 'Crowded'
        else:
            text_p = 'Not crowded'

        # Add the crowd status text
        cv2.rectangle(frame, (0, 0), (250, 100), (0, 0, 0), -1)
        cv2.putText(frame, text_p, (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 255), 1)

        # Save processed image to a temporary file
        temp_file = 'output.jpg'
        cv2.imwrite(temp_file, frame)

        # Return the processed image as a file
        return send_file(temp_file, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define API route
@app.route('/', methods=['GET'])
def say_hi():
    return "Hello from Crowd detection API"

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

