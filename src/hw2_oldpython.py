# Use this if your Python version < 3.6
# Note: place file in above directory first

# IMPORTS
import tensorflow as tf
import PIL
import numpy as np
from pathlib import Path
import os
import sys
from time import time

runtime = time()

# SETTINGS

if len(sys.argv) >= 2:
    INPUT_PATH = Path(sys.argv[1])
else:
    INPUT_PATH = Path('./example_folder')

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

GPUs = [d for d in tf.config.list_physical_devices() if "GPU" in d.device_type]
DEVICE = tf.device("GPU") if (len(GPUs) > 0) else tf.device("CPU")

DTYPE = tf.float32
VERBOSE = True # For a runtime without extra I/O, set VERBOSE to False

OUT_FILENAME = "predict.txt"

TF_VERSION = tf.__version__

if sys.version_info.minor <= 5:
    VERBOSE = False
    print("Use Python >= 3.5 with hw2.py for extra runtime information")

# try:
if int(TF_VERSION.split(".")[1]) > 1:
    t = time()
    safetynet = tf.keras.models.load_model("./model/safetynet.tf")
    robustnet = tf.keras.models.load_model("./model/robustnet.tf")
    safetynet.trainable = False
    robustnet.trainable = False
    
# except:
else:
    from tf2cv.model_provider import get_model # https://github.com/osmr/imgclsmob

    model_name = "sepreresnet56_cifar10" # resnet110_cifar10
    robustnet = get_model(model_name, pretrained=True)
   
    robustnet.trainable = True
    robustnet.load_weights("./model_weight/robustnet.h5")
    robustnet.trainable = False

    DIM = 32

    class Autoencoder(tf.keras.models.Model):
        "A Denoising Autoencoder"
        # Reference: https://www.tensorflow.org/tutorials/generative/cvae
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(DIM, DIM, 3)),
                tf.keras.layers.Conv2D(int(DIM/2), kernel_size=3, activation='relu', padding='same', strides=2),
                tf.keras.layers.Conv2D(int(DIM/4), kernel_size=3, activation='relu', padding='same', strides=2),
            ])

            self.decoder = tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(int(DIM/4), kernel_size=3, activation='relu', padding='same', strides=2),
                tf.keras.layers.Conv2DTranspose(int(DIM/2), kernel_size=3, activation='relu', padding='same', strides=2),
                tf.keras.layers.Conv2D(3, kernel_size=3, activation='sigmoid', padding='same')
            ])

        def call(self, x):
            return self.decoder(self.encoder(x))

    safetynet = Autoencoder()
    safetynet.build((None, 32,32,3))
    safetynet.load_weights("./model_weight/safetynet.h5")

def preprocess(imgs):
    "Normalizes images (according to https://github.com/osmr/imgclsmob/blob/master/examples/demo_tf2.py)"
    mean_rgb = tf.constant([0.485, 0.456, 0.406]) # CIFAR-10 RGB mean
    std_rgb = tf.constant([0.229, 0.224, 0.225]) # CIFAR-10 RGB standard deviation
    mean_rgb = tf.cast(mean_rgb, DTYPE)
    std_rgb = tf.cast(std_rgb, DTYPE)
    return (imgs - mean_rgb)/std_rgb

def get_pred(model, imgs, preproc=True):
    "Returns predictions"
    if preproc:
        imgs = preprocess(imgs)
    try:
        with DEVICE:
            preds = tf.nn.softmax(model(imgs))
            preds = tf.argmax(preds, axis=1)
            preds = tf.cast(preds, tf.uint8)
    except:
        preds = tf.nn.softmax(model(imgs))
        preds = tf.argmax(preds, axis=1)
        preds = tf.cast(preds, tf.uint8)
    return preds

def sort_by_end(filenames):
    "Sorts list of filenames by end digits"
    end_digits = [int(f.split("_")[-1].split(".")[0]) for f in filenames]
    return [pair[0] for pair in sorted(zip(filenames, end_digits), key=lambda pair: pair[1])]


def load_imgs(path, key="png"):
    "Load image files containing a keyphrase from a given path"
    path = Path(path)
    filenames = sort_by_end([f for f in os.listdir(str(path)) if key in f])
    imgs = []
    for f in filenames:
        img = PIL.Image.open(path/f)
        img = np.array(img)[:,:,:3]
        imgs.append(img)
    return np.array(imgs), filenames


# READ INPUT
imgs, fs = load_imgs(INPUT_PATH)

if np.any(imgs > 1):
    imgs = tf.convert_to_tensor(imgs/255., dtype=DTYPE)

clean_imgs = safetynet(imgs)

if len(fs) > 2000:
    preds = []
    for i in range(5):
        start = i*int(len(fs)/5)
        stop = (i+1)*int(len(fs)/5)
        if i == 5:
            stop = len(fs)
            
        X = clean_imgs[start:stop]
        pred = get_pred(robustnet, X)
        preds.append(pred)
    preds = np.array(preds).flatten()
else:
    preds = get_pred(robustnet, clean_imgs)
    preds = preds.numpy()


fp = open(OUT_FILENAME, "w+")

for pred in preds:
    idx = pred.item()
    fp.write(CLASSES[idx])
    fp.write("\n")
fp.close()

print("\nTotal execution time: {:.6f} s".format(time() - runtime))
