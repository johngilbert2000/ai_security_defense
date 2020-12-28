# Generates FGSM and PGD attack images on a given model for CIFAR-10

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

paths = [attack_train_path, attack_test_path, model_path]

for path in paths:
    if not os.path.isdir(path):
        os.mkdir(path)

# SETTINGS

CLASSES = ["airplane", "automobile", "bird", "cat","deer","dog","frog","horse","ship","truck"]

GPUs = [d for d in tf.config.list_physical_devices() if "GPU" in d.device_type]
DEVICE = tf.device("GPU") if (len(GPUs) > 0) else tf.device("CPU")

DTYPE = "float32" # float64

tf.keras.backend.set_floatx(DTYPE) # sets network layers to DTYPE

EPSILON = 8 # on a 0-255 scale


# SELECT MODEL

model_name = "sepreresnet56_cifar10" # resnet110_cifar10 # "resnet20_cifar10"


# LOAD DATASET

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

xtrain = tf.convert_to_tensor(x_train.astype(DTYPE) / 255.)
xtest = tf.convert_to_tensor(x_test.astype(DTYPE) / 255.)
ytrain = tf.convert_to_tensor(y_train.flatten())
ytest = tf.convert_to_tensor(y_test.flatten())


def augment(imgs):
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
    "Restores preprocessed images to original state"
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
    "Creates a perturbation for a given image, label, and model"
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


def main():
    model = get_model(model_name, pretrained=True)
    model.trainable = False
    
    attack = Attack(model, epsilon=EPSILON)

    print(f"Attacking {model_name} with training set")

    for i in range(100):
        # (done in loop because kernel kept dying / SEGFAULT)
        start = i*500
        stop = (i+1)*500

        X = xtrain[start:stop]
        Y = ytrain[start:stop]

        fgsm_train = attack.FGSM(X, Y)
        pgd_train = attack.PGD(X, Y)

        np.save(f"{attack_train_path}/fgsm_train_{i}", fgsm_train.numpy())
        np.save(f"{attack_train_path}/pgd_train_{i}", pgd_train.numpy())
        print(f"\nSaving to {attack_train_path}/fgsm_train_{i}.npy")
        print(f"Saving to {attack_train_path}/pgd_train_{i}.npy\n")
        
        loss, preds = evaluate(model, X, Y)
        orig = np.mean(preds == Y)

        print("Accuracy")
        loss, preds = evaluate(model, fgsm_train, Y)
        new = np.mean(preds == Y)
        print(f"{i} FGSM: {orig*100}% --> {new*100}%")

        loss, preds = evaluate(model, pgd_train, Y)
        new = np.mean(preds == Y)
        print(f"{i} PGD:  {orig*100}% --> {new*100}%")


    print(f"\nAttacking {model_name} with test set")
    
    for i in range(20):
        start = i*500
        stop = (i+1)*500

        X = xtest[start:stop]
        Y = ytest[start:stop]

        fgsm_test = attack.FGSM(X, Y)
        pgd_test = attack.PGD(X, Y)

        np.save(f"{attack_test_path}/fgsm_test_{i}", fgsm_test.numpy())
        np.save(f"{attack_test_path}/pgd_test_{i}", pgd_test.numpy())
        print(f"\nSaving to {attack_test_path}/fgsm_test_{i}.npy")
        print(f"Saving to {attack_test_path}/pgd_test_{i}.npy\n")
        
        loss, preds = evaluate(model, X, Y)
        orig = np.mean(preds == Y)

        print("Accuracy")
        loss, preds = evaluate(model, fgsm_test, Y)
        new = np.mean(preds == Y)
        print(f"{i} FGSM: {orig*100}% --> {new*100}%")

        loss, preds = evaluate(model, pgd_test, Y)
        new = np.mean(preds == Y)
        print(f"{i} PGD:  {orig*100}% --> {new*100}%")
    
    print("\nAttack complete")
    

if __name__ == "__main__":
    main()
