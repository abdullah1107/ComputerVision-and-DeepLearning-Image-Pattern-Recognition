
#import the necessary library
import imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os



ap = argparse.ArgumentParser()

ap.add_argument("-i", "--dataset", required = True,
help = "path to input image")

ap.add_argument("-e", "--encondings", required = True,
help = "path to serialized db of facial encondings")

ap.add_argument("-d", "--detection-method", type = str, default
= "cnn", help = "face detection model to use :either 'hog', or ""
cnn ")

args = vars(ap.parse_args())

print("quantifying faces....")
#load the known face and embedings
#data = pickle.loads(open(args["encoding"], "rb").read())
imagePaths = list(paths.list_images(args["dataset"]))

#initialize the list of known encodings and known  names
knownEncodings = []
knownNames = []


for (i, imagePath) in enumerate(imagePaths):

    #extract the person name from the image path
    print("processing images..{}/{}".format(i+1,
    len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]


    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    boxes = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

print("serializing encoding....")
data = {"encodings": knownEncodings, "names":knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
