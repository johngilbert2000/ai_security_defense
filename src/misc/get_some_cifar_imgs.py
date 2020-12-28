# Saves original CIFAR-10 images to a given folder (OUT_FOLDER)
# Saves image labels to a .txt file (correct.txt)

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import os

NUM_IMAGES = 100
OUT_FOLDER = "../../some_folder"
LABEL_FILE = "../../correct.txt"

CLASSES = ["airplane", "automobile", "bird", "cat","deer","dog","frog","horse","ship","truck"]

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

DTYPE = tf.float32

xtrain = tf.convert_to_tensor(x_train.astype(DTYPE) / 255.)
xtest = tf.convert_to_tensor(x_test.astype(DTYPE) / 255.)
ytrain = tf.convert_to_tensor(y_train.flatten())
ytest = tf.convert_to_tensor(y_test.flatten())

if not os.path.isdir(OUT_FOLDER):
    os.mkdir(OUT_FOLDER)

if NUM_IMAGES > len(xtest):
    NUM_IMAGES = len(xtest)

for i, photo in enumerate(xtest[:NUM_IMAGES]):
    assert photo.shape == (32,32,3)
    plt.imsave(f"{OUT_FOLDER}/{i+1}.png", np.array(photo))

correct = map(lambda idx: CLASSES[idx], ytest[:NUM_IMAGES])
    
fp = open(LABEL_FILE, 'w+')
for c in correct:
    fp.write(c)
    fp.write("\n")
fp.close()
