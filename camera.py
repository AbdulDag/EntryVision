import cv2
from cvzone.FaceDetectionModule import FaceDetector

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(1)
        self.face_detector = FaceDetector()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None, None

        image, bboxs = self.face_detector.findFaces(image, draw=False)

        if bboxs:
            # Face detected
            pass
        else:
            # No face detected
            pass

        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), bboxs

    def release(self):
        self.video.release()