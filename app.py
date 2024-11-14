from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import face_recognition
import os

app = Flask(__name__)
camera = cv2.VideoCapture(1)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Load known faces and encodings
def load_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return face_recognition.load_image_file(file_path)

try:
    saurabh_image = load_image("Saurabh/saurabh.jpg")
    saurabh_face_encoding = face_recognition.face_encodings(saurabh_image)[0]

    abhishek_image = load_image("Abhishek/abhishek.jpg")
    abhishek_face_encoding = face_recognition.face_encodings(abhishek_image)[0]
except FileNotFoundError as e:
    print(e)
except IndexError:
    print("No face found in the image(s).")

known_face_encodings = [saurabh_face_encoding, abhishek_face_encoding]
known_face_names = ["Saurabh", "Abhishek"]

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize for faster processing and convert to RGB for Mediapipe
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            face_names = []

            # Mediapipe for emotion detection
            mediapipe_results = face_detection.process(rgb_frame)

            # Recognize faces and emotions
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                # Draw the bounding box around each face
                top, right, bottom, left = face_location
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 152, 255), 1)

                # Crop and detect emotion
                face_crop = frame[top:bottom, left:right]
                try:
                    analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                    cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    print("Emotion detection error:", e)

            # Encode the frame for streaming
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
