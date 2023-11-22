import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the images")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
print(image.shape[1])

cv2.imshow("image",image.shape[0])
cv2.waitKey(0)
