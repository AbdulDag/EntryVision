import cv2
import face_recognition
import pickle
import os

# Import student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)

imgList = []
userID = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    # Append the userID based on the image filename (without extension)
    userID.append(os.path.splitext(path)[0])

print(userID)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        #change bgr to rgb 
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList
print("Encoding Started . . . ")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown,userID]
print("Encoding Complete") 

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File Saved")

