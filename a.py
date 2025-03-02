import cv2
import face_recognition
import numpy as np
import pickle
import dlib

# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load face encodings
print("Loading Encode File. . . ")
file = open("EncodeFile.p", 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, userID = encodeListKnownWithIds
print("Encode File Loaded")

# Face detector and landmark predictor (for eye blink detection)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download required

# Thresholds
MATCH_THRESHOLD = 0.5  
EYE_AR_THRESHOLD = 0.2  
EYE_BLINK_FRAMES = 3  

blink_counter = 0  
blinking = False  

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR) for blink detection."""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces and encode
    faceCurFrame = face_recognition.face_locations(img)
    encodeCurFrame = face_recognition.face_encodings(img, faceCurFrame)

    closest_face = None
    max_face_size = 0  

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        top, right, bottom, left = faceLoc
        face_size = bottom - top  

        if face_size > max_face_size:
            max_face_size = face_size
            closest_face = (encodeFace, (top, right, bottom, left))

    if closest_face:  
        encodeFace, (top, right, bottom, left) = closest_face

        # Anti-Spoofing: Eye Blink Detection
        face_roi = gray[top:bottom, left:right]
        faces = detector(face_roi)
        
        for face in faces:
            landmarks = predictor(face_roi, face)
            leftEye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(36, 42)])
            rightEye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(42, 48)])

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            avgEAR = (leftEAR + rightEAR) / 2.0

            if avgEAR < EYE_AR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EYE_BLINK_FRAMES:
                    blinking = True  # Real person detected
                blink_counter = 0  

        # Recognize closest real face
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        best_match_distance = faceDis[matchIndex]

        if best_match_distance < MATCH_THRESHOLD and blinking:  # Must blink to be verified
            name = userID[matchIndex]
            color = (0, 255, 0)
            text = "VERIFIED"
        else:
            name = ""
            color = (0, 0, 255)
            text = "SPOOF DETECTED"

        # Draw rectangle & label
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        if text == "VERIFIED":
            cv2.putText(img, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Show webcam feed
    cv2.imshow("Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
