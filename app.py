from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

app = Flask(__name__)
camera = cv2.VideoCapture(0)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Set up face detection with mediapipe
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert the frame to RGB for mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            # Process each detected face
            if results.detections:
                for detection in results.detections:
                    # Extract face bounding box from mediapipe detection
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                           int(bboxC.width * w), int(bboxC.height * h)
                    x, y, w, h = bbox

                    # Crop the face region for DeepFace analysis
                    face_region = frame[y:y+h, x:x+w]

                    try:
                        # Use DeepFace to detect emotion in the cropped face region
                        analysis = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
                        emotion = analysis[0]['dominant_emotion']
                        label = f"{emotion}"
                    except Exception as e:
                        label = "No Emotion"
                        print("Error in emotion detection:", e)

                    # Draw bounding box and label around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Encode the frame to JPEG and yield for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
