#python create_face_images.py --path /Users/Desktop/datasets/mydataset/

#import necessary library

import cv2
import numpy as np
import argparse


#load haar face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to save directory")
args = vars(ap.parse_args())

def face_extractor(img):

    #if function detects face and return cropped face
    #if no face detected, if returns the input images

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is():
        return None

    #crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

#initialize webcam

cap = cv2.VideoCapture(0)


count = 0

while True:

    ret, frame = cap.read()

    if face_extractor(frame) is not None:

        count = count +1

        face = cv2.resize(face_extractor(frame), (250, 250))
        print(args["path"])

        file_name_path = args.get("path") + str(count) + '.jpg'

        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 50:
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")
