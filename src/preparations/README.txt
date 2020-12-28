# Description

This README pertains to how the model was prepared.

To see how to get predictions from an already prepared model,
see the README.txt in the above directory.


# 1. Attack

A pretrained model was initialized and adversarial attack images were generated with:

```
python attack.py
```

# 2. Defend

Next, a denoising autoencoder was initialized and trained
to clean adversarial images with:

```
python safetyencoder.py
```

# 3. Fine-tune

Finally, the original model was fine-tuned using both outputs from the autoencoder
as well as the adversarial attack images with:

```
python adversarial_finetune.py
```

Note: Using TF 2.3.1 may prevent errors with the above files.
