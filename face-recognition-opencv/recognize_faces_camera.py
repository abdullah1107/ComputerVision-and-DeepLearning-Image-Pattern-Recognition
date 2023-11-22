#Runnig command
#python recognize_faces_camera.py --encodings encoding_filename.pickle

#import necessary package
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2



#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required = True,
help = "path to encodings file")
#ap.add_argument("-o", "--output", type = str,
#help = "path to output video")
ap.add_argument("-y", "--display", type = int, default =1,
help = "whether or not to dispaly output frame to screen")
ap.add_argument("-d", "--detect_method", type = str, default = "cnn", help ="face detection model")
args = vars(ap.parse_args())

#load the known faces and embeddings
print("loading encoadings.....")
data = pickle.loads(open(args["encodings"], "rb").read())

#initialize the video stream and point to output video file, then allow the camera sensor to warm up
print("starting video stream....")
vs = cv2.VideoCapture(0)

while True:

    ret, frame = vs.read()

    converted_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    converted_rgb = imutils.resize(frame, width = 400)

    r = frame.shape[1]/float(converted_rgb.shape[1])

    #detect the (x,y) coordinates of the bounding boxes corresponding to each face in the input frame, then compute
    #the facial embeddings for each face
    boxes = face_recognition.face_locations(converted_rgb, model = args["detect_method"])

    encodings = face_recognition.face_encodings(converted_rgb, boxes)

    names = []

    #loop over the facial embeddings

    for enconding in encodings:

        #attempt to match each face in the input image to our known
        #encodings

        matches = face_recognition.compare_faces(data["encodings"], enconding)

        name = "Unknown"

        #check to see if we have found a match
        if True in matches:

            matchedIdxs = [i for (i, b) in enumerate(matches) if b]

            counts = {}


            #loop over the matched indexes and maintain a count for
            #each recognized face face

            for i in matchedIdxs:

                name = data["names"][i]
                #preds = recognizer.predict_proba(name)[i]

                counts[name] = counts.get(name, 0) + 1
                #print("count proba:", counts[name])
                #print("ending")

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie python)
            # will select first entry in the dictionary

            name = max(counts, key = counts.get)
            #print("Count:", counts[name])
            print("temp_count",counts)
            #print("name:", name)
            if counts[name] > 40:
                name = name
                names.append(name)
            else:
                name = "Unknown"
                names.append(name)


        #update the list of names
        #names.append(name)



    #loop over the recognized faces
    for((top, right, bottom, left), name) in zip(boxes, names):

        #rescale the face coordinates

        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        cv2.rectangle(frame, (left, top),(right, bottom),(0, 255, 0),2)
		#y = top - 15 if top - 15 > 15 else top + 15
        if top - 15 >15:
            y = top - 15
        else:
            y = top + 15
        cv2.putText(frame, name, (left, y),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0),2)

    #check to see if we are supposed to display the output frame to the screen
    if args["display"]>0:

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
#do a bit of cleanup
vs.release()
cv2.distroyAllWindows()
vs .stop()
