# Trains an autoencoder to clean FGSM and PGD attack images
# First run attack.py to generate attack images

# IMPORTS

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

import tf2cv, cv2, PIL, os
from tf2cv.model_provider import get_model
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Image

import warnings
warnings.filterwarnings('ignore')

# PATHS

attack_train_path = Path("./attacks_train")
attack_test_path = Path("./attacks_test")
model_path = Path("../model")
AE_train_path = Path("./AE_train")
AE_test_path = Path("./AE_test")

paths = [attack_train_path, attack_test_path, model_path, AE_train_path, AE_test_path]

for path in paths:
    if not os.path.isdir(path):
        os.mkdir(path)
        
# SETTINGS

CLASSES = ["airplane", "automobile", "bird", "cat","deer","dog","frog","horse","ship","truck"]

GPUs = [d for d in tf.config.list_physical_devices() if "GPU" in d.device_type]
DEVICE = tf.device("GPU") if (len(GPUs) > 0) else tf.device("CPU")

DTYPE = "float32" # float64

tf.keras.backend.set_floatx(DTYPE) # sets network layers to DTYPE

EPOCHS = 100
BS = 64 # autoencoder batch size
DIM = 32 # pixel dimensions, e.g. 32 for 32x32 color images


# LOAD DATA

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

xtrain = tf.convert_to_tensor(x_train.astype(DTYPE) / 255.)
xtest = tf.convert_to_tensor(x_test.astype(DTYPE) / 255.)
ytrain = tf.convert_to_tensor(y_train.flatten())
ytest = tf.convert_to_tensor(y_test.flatten())

def sort_by_end(filenames):
    "Sorts list of filenames by end digits"
    end_digits = [int(f.split("_")[-1].split(".")[0]) for f in filenames]
    return [pair[0] for pair in sorted(zip(filenames, end_digits), key=lambda pair: pair[1])]

def load_files(path, key="fgsm"):
    "Load npy files containing a keyphrase from a given path"
    assert (key == "fgsm") or (key == "pgd")
    path = Path(path)
    filenames = sort_by_end([f for f in os.listdir(path) if key in f])
    return np.vstack([np.load(path/f) for f in filenames])

fgsm_train = load_files(attack_train_path, key="fgsm")
pgd_train = load_files(attack_train_path, key="pgd")

fgsm_test = load_files(attack_test_path, key="fgsm")
pgd_test = load_files(attack_test_path, key="pgd")

all_train = np.vstack([xtrain, fgsm_train, pgd_train])
all_train_target = np.vstack([xtrain, xtrain, xtrain])

fgsm_train = tf.convert_to_tensor(fgsm_train.astype(DTYPE))
pgd_train = tf.convert_to_tensor(pgd_train.astype(DTYPE))
fgsm_test = tf.convert_to_tensor(fgsm_test.astype(DTYPE))
pgd_test = tf.convert_to_tensor(pgd_test.astype(DTYPE))

all_train = tf.convert_to_tensor(all_train.astype(DTYPE))
all_train_target = tf.convert_to_tensor(all_train_target.astype(DTYPE))


class Autoencoder(tf.keras.models.Model):
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
    
    def summary(self, line_length=None, positions=None, print_fn=None):
        self.encoder.summary(line_length, positions, print_fn)
        self.decoder.summary(line_length, positions, print_fn)

        
def main():
    
    try:
        assert fgsm_train.shape[0] == xtrain.shape[0]
        assert pgd_train.shape[0] == xtrain.shape[0]
        assert fgsm_test.shape[0] == xtest.shape[0]
        assert pgd_test.shape[0] == xtest.shape[0]
    except AssertionError:
        print("Incomplete attack data; please run attack.py to completion first")
        return
    
    autoencoder = Autoencoder()
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)

    with DEVICE:
        autoencoder.fit(all_train, all_train_target, batch_size=BS, epochs=EPOCHS, callbacks=[reduce_lr])
    
    
    print(f"\nSaving model to {model_path}/safety_AE")
    autoencoder.save(model_path/"safety_AE")

    autoencoder.summary()
    
    print(f"\nSaving AE test outputs to {AE_test_path}")
    # SAVE TEST OUTPUTS

    enc_orig = autoencoder.encoder(xtest)
    dec_orig = autoencoder.decoder(enc_orig)

    enc_fgsm = autoencoder.encoder(fgsm_test)
    dec_fgsm = autoencoder.decoder(enc_fgsm)

    enc_pgd = autoencoder.encoder(pgd_test)
    dec_pgd = autoencoder.decoder(enc_pgd)

    np.save(AE_test_path/"dec_orig_test", dec_orig)
    np.save(AE_test_path/"dec_fgsm_test", dec_fgsm)
    np.save(AE_test_path/"dec_pgd_test", dec_pgd)
    
    # SAVE TRAINING OUTPUTS

    print(f"Saving AE training outputs to {AE_train_path}")
    for i in range(5):
        start = i*10000
        stop = (i+1)*10000

        print(f"{i}: [{start}, {stop}]")

        enc_orig = autoencoder.encoder(xtrain[start:stop])
        dec_orig = autoencoder.decoder(enc_orig)

        enc_fgsm = autoencoder.encoder(fgsm_train[start:stop])
        dec_fgsm = autoencoder.decoder(enc_fgsm)

        enc_pgd = autoencoder.encoder(pgd_train[start:stop])
        dec_pgd = autoencoder.decoder(enc_pgd)

        np.save(f"{AE_train_path}/orig_train_dec_{i}", dec_orig)
        np.save(f"{AE_train_path}/fgsm_train_dec_{i}", dec_fgsm.numpy())
        np.save(f"{AE_train_path}/pgd_train_dec_{i}", dec_pgd.numpy())
        
        
if __name__ == "__main__":
    main()
