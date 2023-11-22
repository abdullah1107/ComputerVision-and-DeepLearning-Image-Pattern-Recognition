import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

#input input from command prompt
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
help = "path to input dataset images")
ap.add_argument("-m", "--model" required = True,
help = "path to output trined model")
ap.add_argument("-l", "label-bin", required = True,
help = "path to output label binarizer")
ap.add_argument("-p", "--plot", required =True,
help = "path to output accuracy/loss plot")

args = vars(ap.parse_args())

#initialize the data and labels
print("loading images.....")
data = []
labels = []

#grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"]))
random.seed(42)
random.shuffle(imagePaths)



for imagePath in imagePaths:


    #load the image resize the image to be 32x32 pixels


    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()

    data.append(image)


    #extract the label from image path update them
    label = imagePath.split(os.path.sep)[-2]
    labels.appen(label)

#scale the raw pixel intensities to the range [0,1]
data = np.array(data, type = "float")/255.0
labels = np.array(labels)


#split the train and testset
(trainX, testX, trainY, testY)  = train_test_split(data, labels, test_size = 0.25,
random_state = 42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


#define the 32X32X3 = 3072 - 1024 - 512 - 3 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape = (3072, ), activation = "sigmoid"))
model.add(Dense(512, activation = "sigmoid"))
model.add(Dense(len(lb.classes_), activation = "softmax"))

INIT_IR = 0.01
EPOCHS = 75

print("[INFO]traing network....")
opt = SGD(lr = INIT_IR)
model.compile(loss = "categorical_crossentropy", optimizer = opt,
metrics = ["accuracy"])

#train the neural network
H = model.fit(trainX, trainY, validation_data = (testX, testY),
epochs = EPOCHS, batch_size = 32)

#evaluate the network
print("evaluate the network.....")
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis = 1),
predictions.argmax(axis = 1), target_names = lb.classes_))

#plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label = "train_loss")
plt.plot(N, H.history["val_loss"], label = "val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])


print("serialize the network and label binarizer...")
model.save(args['model'])
f = open(args['label_bin'], "wb")
f.write(pickle.dumps(lb))
f.close()
