import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Update if using a remote DB
db = client["EntryVision"]
collection = db["Users"]

# Fetch user data from MongoDB
knownEncodings = []
userIDs = []

for user in collection.find():
    knownEncodings.append(np.array(user["encoding"]))  # Store face encodings
    userIDs.append(user["name"])  # Store user names

print("Loaded Users from MongoDB!")

# Start Camera
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(knownEncodings, encodeFace)
        faceDis = face_recognition.face_distance(knownEncodings, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = userIDs[matchIndex]
            color = (0, 255, 0)  # Green for verified
            text = "VERIFIED"
        else:
            name = "UNKNOWN"
            color = (0, 0, 255)  # Red for unknown
            text = "UNKNOWN"

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imshow("Face Recognition", img)
    cv2.waitKey(1)
