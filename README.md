# Wespe-Keras

This repository contains the re-implementation of WESPE paper in Keras. 
It is not complete yet. The total variation loss has not been incorporated yet. I am skeptical about that color Discriminator, because
most of the reconstructed test images have the correct texture but not the correct color distribution. It could be the case that I have not 
implemented the blurring process correctly (I have used the DepthWiseConv2D keras layer to implement it).

Image enhancement is achieved by mapping images from the domain of phone images to the domain of DSLR images (denoted as domain A and B respectively in the code).
The data path should contain 4 files: trainA, trainB, testA, testB in the same level.

