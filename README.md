# Wespe-Keras

This repository is an unofficial implementation of WESPE paper in Keras. The paper achieves unsupervised/weakly supervised smartphone image enhancement by mapping images from the domain of phone images to the domain of DSLR images (denoted as domain A and B respectively) using an architecture inspired by the CycleGAN (https://arxiv.org/pdf/1703.10593.pdf). 

It maps a training image a from A->B using the Forward Generator G. The image G(a) is input to two discriminators (the first decides whether the image is a real domain B or an enhanced domain A based on its color distribution, while the second decides based on its texture). Finally, the generated image G(a) is mapped back to domain A by the backward generator F. 6 different losses are used: 2 Discriminator Adversarial, 2 Generator Adversarial, a total variation loss on the enhanced image G(a) and a cycle-consistency loss on the reconstructed image F(G(a)) (some norm of a-F(G(a) is minimised). 

I have modified the model proposed by the paper because some crucial training details were not provided which made it very difficult to find the right combination of all training parameters for stable GAN training. The **main modifications** are:

* **G generator has greater capacity than the backward Generator F**. My intuition for this change was the fact that G learns a more complex mapping (LR --> HR), while F learns a less complex mapping (HR --> LR). I have also removed the batch_normalisation layers from both G and F, because they caused instability in training and serious deterioration of the performance.

* **Different Discriminator architecture based on the PatchGAN discriminator used in the CycleGAN paper**. The difference between a PatchGAN and regular GAN discriminator is that rather the regular GAN maps from a 256x256 image to a single scalar output, which signifies "real" or "fake", whereas the PatchGAN maps from 256x256 to an NxN array of outputs X, where each $X_{ij}$ signifies whether the patch ij in the image is real or fake. Which is patch ij in the input? Well, output $X_{ij}$ is just a neuron in a convnet, and we can trace back its receptive field to see which input pixels it is sensitive to. In the CycleGAN architecture, the receptive fields of the discriminator turn out to be 70x70 patches in the input image!

* **A cycle reconstruction loss in both domain A and B**. I have discovered that imposing a cycle reconstruction only in both domain A and B significantly improved the performance of the network than using a reconstruction loss only in domain A.



## Getting Started


Steps to run the training:

* Put the training and test data of domains A and B under the folders data/trainA, data/trainB, data/testA and data/testB
* run the modelwithVGGloss.py file (you can change the patch size, epochs, batch_size and other parameters in the main)

## Requirements
You don't need all the packages for the training of WESPE. However, for the full use of the entire repository you need all the packages listed below:

* keras (tensorflow backend)
* scipy
* Pillow
* openCV
* scikit-image
* Matplotlib


## Preliminary experiments/results

Visual results after 1 and 2 epochs (about 1.5h of training time in GTX 2080-ti) are saved in the folder "sample images"

Qualitative & quantitative results of the full training and the trained model will be released soon
