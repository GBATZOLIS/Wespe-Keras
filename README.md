# Wespe-Keras

This repository contains the re-implementation of WESPE paper in Keras. 

It is not complete yet. The total variation loss has not been incorporated yet. However, decent results are obtained even without the incorporation of the total variation loss which provides spatial smoothing.

The performance of the color Discriminator needs improvement, because
most of the reconstructed test images have the correct texture but not the correct color distribution. A possible solution is the increase of the weight associated with the color loss wrt to the weight associated with the texture loss. Another solution is to change the blurring kernel.

Image enhancement is achieved by mapping images from the domain of phone images to the domain of DSLR images (denoted as domain A and B respectively in the code).
The data path should contain 4 files: trainA, trainB, testA, testB on the same level.

