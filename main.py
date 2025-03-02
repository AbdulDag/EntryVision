import cv2
import face_recognition
import numpy as np
import os
import pickle

cap = cv2.VideoCapture(1)

cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.jpg')

# Importing images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Load encode file
print("Loading Encode File. . . ")
file = open("EncodeFile.p", 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, userID = encodeListKnownWithIds
print("Encode File Loaded")

while True:
    success, img = cap.read()

    # Make image small because it takes a lot of computational power
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 568, 808:808 + 439] = imgModeList[0]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        # Get the bounding box coordinates of the face
        top, right, bottom, left = faceLoc

        # Compare faces and find the match
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Find the index of the best match
        matchIndex = np.argmin(faceDis)

        # Draw rectangle around the face
        cv2.rectangle(imgBackground, (left + 55, top + 162), (right + 55, bottom + 162), (0, 255, 0), 2)

        # Check if match is found
        if matches[matchIndex]:
            name = userID[matchIndex]
            color = (0, 255, 0)  # Green for verified
            text = "VERIFIED"
        else:
            name = ""
            color = (0, 0, 255)  # Red for unknown
            text = "UNKNOWN"

        # Position the status text relative to the face's position
        cv2.putText(imgBackground, text, (left + 55, top + 162 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        # Optionally display the name below the face (for verified users)
        if matches[matchIndex]:
            cv2.putText(imgBackground, name, (left + 55, bottom + 162 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Show the images
    cv2.imshow("Camera", img)
    cv2.imshow("Entry Vision", imgBackground)
    cv2.waitKey(1)
