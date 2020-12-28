# Finetunes a given model with adversarial images and autoencoder outputs


# IMPORTS

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

import tf2cv, cv2, PIL, os
from tf2cv.model_provider import get_model
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Image

import warnings
# warnings.filterwarnings('ignore')

# PATHS

attack_train_path = Path("./attacks_train")
attack_test_path = Path("./attacks_test")
model_path = Path("../model")
model_weight_path = Path("../../model_weight")
AE_train_path = Path("./AE_train")
AE_test_path = Path("./AE_test")

paths = [attack_train_path, attack_test_path, model_path, model_weight_path, AE_train_path, AE_test_path]

for path in paths:
    if not os.path.isdir(path):
        os.mkdir(path)

# SETTINGS

CLASSES = ["airplane", "automobile", "bird", "cat","deer","dog","frog","horse","ship","truck"]

GPUs = [d for d in tf.config.list_physical_devices() if "GPU" in d.device_type]
DEVICE = tf.device("GPU") if (len(GPUs) > 0) else tf.device("CPU")

DTYPE = "float32" # float64

tf.keras.backend.set_floatx(DTYPE) # sets network layers to DTYPE

BS = 64 # autoencoder batch size
DIM = 32 # pixel dimensions, e.g. 32 for 32x32 color images


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



def sort_by_end(filenames):
    "Sorts list of filenames by end digits"
    end_digits = [int(f.split("_")[-1].split(".")[0]) for f in filenames]
    return [pair[0] for pair in sorted(zip(filenames, end_digits), key=lambda pair: pair[1])]

def load_files(path, key="fgsm"):
    "Load npy files containing a keyphrase from a given path"
    path = Path(path)
    filenames = sort_by_end([f for f in os.listdir(path) if key in f])
    return np.vstack([np.load(path/f) for f in filenames])

# original adversarial examples
fgsm_train = load_files(attack_train_path, key="fgsm")
pgd_train = load_files(attack_train_path, key="pgd")

fgsm_test = load_files(attack_test_path, key="fgsm")
pgd_test = load_files(attack_test_path, key="pgd")

# autoencoder cleaned adversarial examples
fgsm_train_dec = load_files(AE_train_path, key="fgsm")
pgd_train_dec = load_files(AE_train_path, key="pgd")
orig_train_dec = load_files(AE_train_path, key="orig")

dec_orig_test = np.load(AE_test_path/"dec_orig_test.npy")
dec_fgsm_test = np.load(AE_test_path/"dec_fgsm_test.npy")
dec_pgd_test = np.load(AE_test_path/"dec_pgd_test.npy")


all_train = np.vstack([xtrain, fgsm_train, pgd_train, orig_train_dec, fgsm_train_dec, pgd_train_dec])
all_train_target = np.hstack([ytrain, ytrain, ytrain, ytrain, ytrain, ytrain])

fgsm_train_dec = tf.convert_to_tensor(fgsm_train_dec.astype(DTYPE))
pgd_train_dec = tf.convert_to_tensor(pgd_train_dec.astype(DTYPE))
orig_train_dec = tf.convert_to_tensor(orig_train_dec.astype(DTYPE))

dec_orig_test = tf.convert_to_tensor(dec_orig_test.astype(DTYPE))
dec_fgsm_test = tf.convert_to_tensor(dec_fgsm_test.astype(DTYPE))
dec_pgd_test = tf.convert_to_tensor(dec_pgd_test.astype(DTYPE))

all_train = tf.convert_to_tensor(all_train.astype(DTYPE))
all_train_target = tf.convert_to_tensor(all_train_target.astype(DTYPE))


all_train2 = np.vstack([xtrain, orig_train_dec, fgsm_train_dec, pgd_train_dec])
all_train_target2 = np.hstack([ytrain, ytrain, ytrain, ytrain])


def assess(model, AE, verbose=True):
    "Checks if finetuned model should be saved or discarded; returns np.array of accuracies"
    enc_orig = AE.encoder(xtest)
    dec_orig = AE.decoder(enc_orig)

    enc_fgsm = AE.encoder(fgsm_test)
    dec_fgsm = AE.decoder(enc_fgsm)

    enc_pgd = AE.encoder(pgd_test)
    dec_pgd = AE.decoder(enc_pgd)

    print("(Test Accuracy) With autoencoder protection:")

    loss, preds = evaluate(model, xtest, ytest)
    orig_reg = np.mean(preds == ytest)
    loss, preds = evaluate(model, dec_orig, ytest)
    new_reg = np.mean(preds == ytest)
    if verbose:
        print(f"  Regular: {orig_reg} --> {new_reg}")

    loss, preds = evaluate(model, fgsm_test, ytest)
    orig_fgsm = np.mean(preds == ytest)
    loss, preds = evaluate(model, dec_fgsm, ytest)
    new_fgsm = np.mean(preds == ytest)
    if verbose:
        print(f"  FGSM: {orig_fgsm} --> {new_fgsm}")

    loss, preds = evaluate(model, pgd_test, ytest)
    orig_pgd = np.mean(preds == ytest)
    loss, preds = evaluate(model, dec_pgd, ytest)
    new_pgd = np.mean(preds == ytest)
    if verbose:
        print(f"  PGD: {orig_pgd} --> {new_pgd}")
        
    # Ensure accuracies stay above 70%
    assert (new_reg > 0.7)
    return np.array([new_reg, new_fgsm, new_pgd])


def main():
    autoencoder = tf.keras.models.load_model(model_path/"safety_AE")
    autoencoder.trainable = False
    
    run = 1
    
    try:
        # continue fine-tuning if model already exists
        model = tf.keras.models.load_model(model_path/"robust_model")
        print(f"Loading model from {model_path}/robust_model")
    except OSError:
        # else load fresh model
        model_name = "sepreresnet56_cifar10" # resnet110_cifar10
        model = get_model(model_name, pretrained=True)
        print(f"Loading new pretrained {model_name} model")

    X = preprocess(all_train)
    Y = tf.cast(all_train_target, tf.uint8)
    assert not np.any(np.isnan(X))
    
    print("Initial evaluation")
    accs = assess(model, autoencoder)

    # Zeroth Run
    LR = 1e-4
    BS = 256
    EPOCHS = 1
    run = 0
    print(f"({run}) LR: {LR}, BS: {BS}, EPOCHS: {EPOCHS}, AE outputs + original adversarial examples")
    run += 1
    
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=LR)
    opt = tf.keras.optimizers.Adam(learning_rate=LR) 
    model.trainable = True
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy())
    
    with DEVICE:
        model.fit(X, Y, batch_size=BS, epochs=EPOCHS) #, callbacks=[reduce_lr])

    # Assess
    new_accs = assess(model, autoencoder)
    if True in (new_accs > accs):
        accs = new_accs
        model.save(model_path/"robust_model")
    else:
        print("Failed to save model; reloading model")
        try:
            # continue fine-tuning if model already exists
            model = tf.keras.models.load_model(model_path/"robust_model")
        except OSError:
            # else load fresh model
            model_name = "sepreresnet56_cifar10" # resnet110_cifar10
            model = get_model(model_name, pretrained=True)
        model.trainable = True
    
    
    # First Run
    LR = 1e-4
    BS = 256
    EPOCHS = 2 # 3
    print(f"({run}) LR: {LR}, BS: {BS}, EPOCHS: {EPOCHS}, AE outputs + original adversarial examples")
    run += 1
    
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=LR)
    opt = tf.keras.optimizers.Adam(learning_rate=LR) 
    model.trainable = True
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy())
    
    with DEVICE:
        model.fit(X, Y, batch_size=BS, epochs=EPOCHS) #, callbacks=[reduce_lr])

    # Assess
    new_accs = assess(model, autoencoder)
    if True in (new_accs > accs):
        accs = new_accs
        model.save(model_path/"robust_model")
    else:
        print("Failed to save model; reloading model")
        try:
            # continue fine-tuning if model already exists
            model = tf.keras.models.load_model(model_path/"robust_model")
        except OSError:
            # else load fresh model
            model_name = "sepreresnet56_cifar10" # resnet110_cifar10
            model = get_model(model_name, pretrained=True)
        model.trainable = True
    
    
    
    # Second Run
    EPOCHS = 1
    X = preprocess(all_train2)
    Y = tf.cast(all_train_target2, tf.uint8)
    assert not np.any(np.isnan(X))
    print(f"({run}) LR: {LR}, BS: {BS}, EPOCHS: {EPOCHS}, just AE outputs")
    run += 1

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=LR)
    opt = tf.keras.optimizers.Adam(learning_rate=LR) # learning_rate=LR
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy())
    
    with DEVICE:
        model.fit(X, Y, batch_size=BS, epochs=EPOCHS) #, callbacks=[reduce_lr])
        
    # Assess
    new_accs = assess(model, autoencoder)
    if True in (new_accs > accs):
        accs = new_accs
        model.save(model_path/"robust_model")
        print(f"Saving model to {model_path}/robust_model")
    else:
        print("Failed to save model; reloading model")
        try:
            model = tf.keras.models.load_model(model_path/"robust_model")
        except OSError:
            model_name = "sepreresnet56_cifar10" # resnet110_cifar10
            model = get_model(model_name, pretrained=True)
        model.trainable = True

        
        
    # Third Run
    LR = 1e-5
    EPOCHS = 3
    print(f"({run}) LR: {LR}, BS: {BS}, EPOCHS: {EPOCHS}, just AE outputs")
    run += 1
    
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=LR)
    opt = tf.keras.optimizers.Adam(learning_rate=LR) # LR=1e-4
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy()) 
    
    with DEVICE:
        model.fit(X, Y, batch_size=BS, epochs=EPOCHS) #, callbacks=[reduce_lr])

    # Assess
    new_accs = assess(model, autoencoder)
    if True in (new_accs > accs):
        accs = new_accs
        model.save(model_path/"robust_model")
        print(f"Saving model to {model_path}/robust_model")
    else:
        print("Failed to save model; reloading model")
        try:
            model = tf.keras.models.load_model(model_path/"robust_model")
        except OSError:
            model_name = "sepreresnet56_cifar10" # resnet110_cifar10
            model = get_model(model_name, pretrained=True)
        model.trainable = True        

if __name__ == "__main__":
    main()
    
