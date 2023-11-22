#import the necessary  library
#python opencvwith_deeplearning.py --image images/jeema.png --prototxt bvlc_googlenet.prototxt --model bvls_googlenet.caffemodel --labels synset_words.txt

import numpy as np
import argparse
import time
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
	#help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
	help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())
#load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(True):
	ret, frame = cap.read()
	
	top_left_x = int(width / 3)
	top_left_y = int((height / 2) + (height / 4))
	bottom_right_x = int((width / 3) * 2)
	bottom_right_y = int((height / 2) - (height / 4))
    # Our operations on the frame come here
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), 255, 3)
	image = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]
	blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
	#load our serialized model from disk
	print("loading model....")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	#set the blob as input to the network and perform a forward-pass to
	#obtain our output classification
	net.setInput(blob)
	start = time.time()
	preds = net.forward()
	end = time.time()

	print("classification took {:.5} seconds".format(end - start))

	# sort the indexes of the probabilities in desending order (higher
	# probabilities first) and grab the top-5 predictions

	idxs = np.argsort(preds[0])[::-1][:5]

	#loop over the top-5 predictions and display them
	for (i, idx) in enumerate(idxs):
	    #draw the top prediction on the input image
	     if i == 0:
	         text = "Label: {}, {:.2f}%".format(classes[idx],
	              preds[0][idx] * 100)

	         cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	 			0.7, (0, 0, 255), 2)

	     #display the predicted label + associated probability to the console
	     print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
	 		classes[idx], preds[0][idx]))

	#disply the output images
	cv2.imshow("image", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
