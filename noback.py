import cv2
import face_recognition
import numpy as np
import pickle

# Initialize video capture (camera 1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# Set the resolution of the captured video
cap.set(3, 1920)  # Width  
cap.set(4, 1080)   # Height  


# Load encode file (for known faces)
print("Loading Encode File. . . ")
file = open("EncodeFile.p", 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, userID = encodeListKnownWithIds
print("Encode File Loaded")

while True:
    success, img = cap.read()

    # Make the image smaller to reduce computational load
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and get their encodings
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        # Get the bounding box coordinates of the face (scaled to original size)
        top, right, bottom, left = faceLoc

        # Scale back the coordinates to the original image size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Compare the face encoding with known encodings
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Find the index of the best match
        matchIndex = np.argmin(faceDis)

        # Draw rectangle around the face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        # If a match is found, display "VERIFIED" and the name
        if matches[matchIndex]:
            name = userID[matchIndex]
            color = (0, 255, 0)  # Green for verified
            text = "VERIFIED"
        else:
            name = ""
            color = (0, 0, 255)  # Red for unknown
            text = "UNKNOWN"

        # Display the status text near the face
        cv2.putText(img, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        
        # Optionally display the name below the face (for verified users)
        if matches[matchIndex]:
            cv2.putText(img, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Show the webcam feed with face detection
    cv2.imshow("Camera", img)

    # Wait for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()
