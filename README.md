# Wespe-Keras

This repository contains an unofficial implementation of WESPE paper in Keras. There are some modifications such as the use of the Identity loss which is not used in WESPE but used in CycleGAN and the use of InstaceNormalisation layer which improved the stability of the training.

Image enhancement is achieved by mapping images from the domain of phone images to the domain of DSLR images (denoted as domain A and B respectively in the code).

## Getting Started


Steps to run the training:

* Put the training and test data of domains A and B under the folders data/trainA, data/trainB, data/testA and data/testB
* run the model.py file (you can change the patch size, epochs, batch_size and other parameters in the main)

## Preliminary experiments/results

Visual results after 1 and 2 epochs (about 1.5h of training time in GTX 2080-ti) are saved in the folder "sample images"

Qualitative & quantitative results of the full training and the trained model will be released soon
