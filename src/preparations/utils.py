# IMPORTS

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

import tf2cv, cv2, PIL, os
from tf2cv.model_provider import get_model
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

CLASSES = ["airplane", "automobile", "bird", "cat","deer","dog","frog","horse","ship","truck"]


def sort_by_end(filenames):
    "Sorts list of filenames by end digits"
    end_digits = [int(f.split("_")[-1].split(".")[0]) for f in filenames]
    return [pair[0] for pair in sorted(zip(filenames, end_digits), key=lambda pair: pair[1])]

def load_files(path, key="fgsm"):
    "Load npy files containing a keyphrase from a given path"
    path = Path(path)
    filenames = sort_by_end([f for f in os.listdir(path) if key in f])
    return np.vstack([np.load(path/f) for f in filenames])

def augment(imgs):
    "an unused function"
    imgs = tf.image.rot90(imgs, np.random.randint(0,4))
    imgs = tf.image.random_flip_left_right(imgs)
    imgs = tf.image.random_flip_up_down(imgs)
    imgs = tf.image.random_hue(imgs, 0.07)
    imgs = tf.image.random_saturation(imgs, 0.7, 1.5)
    imgs = tf.image.random_brightness(imgs, 0.05)
    imgs = tf.image.random_contrast(imgs, 0.6, 1.2)
    imgs = tf.clip_by_value(imgs, 0, 1)
    return imgs

def preprocess(imgs):
    "Normalizes images (according to https://github.com/osmr/imgclsmob/blob/master/examples/demo_tf2.py)"
    mean_rgb = tf.constant([0.485, 0.456, 0.406]) # CIFAR-10 RGB mean
    std_rgb = tf.constant([0.229, 0.224, 0.225]) # CIFAR-10 RGB standard deviation
    mean_rgb = tf.cast(mean_rgb, DTYPE)
    std_rgb = tf.cast(std_rgb, DTYPE)
    return (imgs - mean_rgb)/std_rgb
#     return tf.clip_by_value((imgs - mean_rgb) / std_rgb, 0, 1)

def undo_preprocess(imgs):
    "restores preprocessed images to original state"
    mean_rgb = tf.constant([0.485, 0.456, 0.406]) # CIFAR-10 RGB mean
    std_rgb = tf.constant([0.229, 0.224, 0.225]) # CIFAR-10 RGB standard deviation
    mean_rgb = tf.cast(mean_rgb, DTYPE)
    std_rgb = tf.cast(std_rgb, DTYPE)
    return imgs*std_rgb + mean_rgb
#     return tf.clip_by_value(imgs*std_rgb + mean_rgb, 0, 1)

def evaluate(model, imgs, labels, loss_fn=tf.keras.losses.CategoricalCrossentropy(), preproc=True):
    "Returns loss and predictions"
    if preproc:
        imgs = preprocess(imgs)
    preds = tf.nn.softmax(model(imgs))
    if labels.shape[-1] != len(CLASSES):
        final_preds = tf.argmax(preds, axis=1)
        final_preds = tf.cast(final_preds, labels.dtype)
        labels = tf.one_hot(labels, depth=len(CLASSES), axis=1)
    loss = loss_fn(labels, preds)
    return loss, final_preds

def perturb(imgs, labels, model, loss_fn=tf.keras.losses.CategoricalCrossentropy()):
    "creates a perturbation (as a np.array) for a given image, label, and model"
    with tf.GradientTape() as g:
        g.watch(imgs)
        preds = tf.nn.softmax(model(imgs))
        loss = loss_fn(labels, preds)
        grad = g.gradient(loss, imgs)
        grad = grad if grad is not None else tf.zeros(imgs.shape)
    return tf.sign(grad)

class Attack:
    def __init__(self, model, loss_fn=None, epsilon=8, target=None, preproc=False):
        """
        model: tf model
        loss_fn: tf loss function or None (default: CategoricalCrossentropy)
        epsilon: maximum image noise perturbation within [0,255]
        target: int (targeted) or None (untargeted)
        """
        assert type(target) == int or target == None
        assert type(loss_fn) != str
        assert epsilon < 255 and epsilon >= 1
        
        self.epsilon = epsilon / 255.
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else tf.keras.losses.CategoricalCrossentropy()
        self.target = target
        self.preproc = preproc

    def FGSM(self, imgs, labels):
        "FGSM attack; inputs: tf tensors"
        if self.preproc:
            imgs = preprocess(imgs)
        if type(self.target) == int:
            labels = tf.constant([self.target]*imgs.shape[0])
        if labels.shape[-1] != len(CLASSES):
            labels = tf.one_hot(labels, depth=len(CLASSES), axis=1)

        noise = perturb(imgs, labels, self.model, self.loss_fn)
        noise = tf.clip_by_value(noise, -1, 1)
        adv_imgs = (imgs + self.epsilon*noise) if self.target is None else (imgs - self.epsilon*noise)
        adv_imgs = tf.clip_by_value(adv_imgs, 0, 1)
        if self.preproc:
            adv_imgs = undo_preprocess(adv_imgs)
        return adv_imgs
    
    def PGD(self, imgs, labels, learning_rate=0.01, steps=10, random_init=True):
        "PGD attack; inputs: tf tensors"
        if learning_rate >= 1 and learning_rate < 255:
            learning_rate = learning_rate / 255.
        if type(self.target) == int:
            labels = tf.constant([self.target]*imgs.shape[0])
        if labels.shape[-1] != len(CLASSES):
            labels = tf.one_hot(labels, depth=len(CLASSES), axis=1)

        if self.preproc:
            imgs = preprocess(imgs)
        x = imgs
        if random_init:
            init_noise = tf.random.uniform(imgs.shape, -self.epsilon, self.epsilon)
            init_noise = tf.cast(init_noise, x.dtype)
            x += init_noise
            x = tf.clip_by_value(x, 0, 1)

        for i in range(steps):
            noise = perturb(x, labels, self.model, self.loss_fn)
            noise = tf.clip_by_value(noise, -1, 1)

            if type(self.target) == int:
                x -= learning_rate * noise
            else:
                x += learning_rate * noise
            x = tf.clip_by_value(x, imgs - self.epsilon, imgs + self.epsilon)
            x = tf.clip_by_value(x, 0, 1)
        if self.preproc:
            x = undo_preprocess(x)
        return x
    
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

