import cv2
import face_recognition
import numpy as np
import pickle
import serial
import time

# Initialize video capture (camera 1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load encode file (for known faces)
print("Loading Encode File. . . ")
file = open("EncodeFile.p", 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, userID = encodeListKnownWithIds
print("Encode File Loaded")

# Face recognition threshold (lower = stricter)
MATCH_THRESHOLD = 0.4

# Open the serial connection to Arduino
ser = serial.Serial('COM8', 9600, timeout=1)  # Replace with your actual port
time.sleep(2)  # Allow Arduino to reset

while True:
    success, img = cap.read()
    
    # Resize for faster processing but keep good resolution for encoding
    imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)  # Use 50% size instead of 25%
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    closest_face = None
    max_face_size = 0  

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        top, right, bottom, left = faceLoc

        # Scale back to original size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Estimate face size
        face_size = bottom - top  

        # Identify the closest face (largest detected face)
        if face_size > max_face_size:
            max_face_size = face_size
            closest_face = (encodeFace, (top, right, bottom, left))

    if closest_face:  
        encodeFace, (top, right, bottom, left) = closest_face

        # Compare with known faces
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        best_match_distance = faceDis[matchIndex]

        # Check if it's a valid match
        if best_match_distance < MATCH_THRESHOLD:
            name = userID[matchIndex]
            color = (0, 255, 0)  # Green for verified
            text = "VERIFIED"
            print(f"{name} verified, unlocking door...")  # Debugging message
            # Send unlock command to Arduino
            ser.write(b'UNLOCK')
            time.sleep(5)  # Wait while the door is unlocked (adjust duration as needed)
        else:
            name = ""
            color = (0, 0, 255)  # Red for unknown
            text = "UNKNOWN"

        # Draw rectangle & label
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        if text == "VERIFIED":
            cv2.putText(img, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Show the webcam feed
    cv2.imshow("EntryVision", img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Close the serial connection when done
ser.close()
