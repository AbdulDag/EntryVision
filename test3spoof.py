import cv2
import face_recognition
import numpy as np
import pickle

# Initialize video capture (camera 1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load encode file (for known faces)
print("Loading Encode File. . . ")
file = open("EncodeFile.p", 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, userID = encodeListKnownWithIds
print("Encode File Loaded")

# Face recognition threshold (lower = stricter, recommended 0.35-0.4 for accuracy)
MATCH_THRESHOLD = 0.4  

# Define a function to detect rectangular "boxes" (e.g., a phone screen)
def detect_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find rectangles (boxes)
    boxes = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 sides (rectangle)
        if len(approx) == 4:
            boxes.append(approx)
    return boxes

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # 25% size for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces
    faceCurFrame = face_recognition.face_locations(imgS, model="hog")
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    if len(faceCurFrame) == 0:
        # No faces detected
        continue

    # Detect potential phone-like boxes in the frame
    boxes = detect_boxes(img)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        top, right, bottom, left = faceLoc

        # Scale back to original size
        top *= 4  # Since we resized by 0.25x
        right *= 4
        bottom *= 4
        left *= 4

        # Estimate face size (width and height)
        face_width = right - left
        face_height = bottom - top

        # Check if face is inside any detected box (representing a phone)
        for box in boxes:
            # Approximate the bounding box for each detected box
            box = box.reshape((-1, 2))
            rect_x_min, rect_y_min = np.min(box, axis=0)
            rect_x_max, rect_y_max = np.max(box, axis=0)

            # Check if the face is within the bounding box of the detected phone-like object
            if rect_x_min < left < rect_x_max and rect_x_min < right < rect_x_max and rect_y_min < top < rect_y_max and rect_y_min < bottom < rect_y_max:
                text = "SPOOF DETECTED"
                color = (0, 0, 255)  # Red for spoofed face
                break
        else:
            text = "UNKNOWN"
            color = (0, 0, 255)  # Red for unknown

        # Compare with known faces using stricter threshold
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        best_match_distance = faceDis[matchIndex]

        if best_match_distance < MATCH_THRESHOLD:
            name = userID[matchIndex]
            text = "VERIFIED"
            color = (0, 255, 0)  # Green for verified
        else:
            name = ""

        # Draw rectangle & label
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        if text == "VERIFIED":
            cv2.putText(img, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Show the webcam feed
    cv2.imshow("Camera", img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
