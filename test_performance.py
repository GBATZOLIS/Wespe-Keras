# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 20:13:03 2019

@author: Georgios
"""
import keras
import keras.backend as K
from data_loader import DataLoader
import numpy as np
from architectures import generator_network
import matplotlib.pyplot as plt
from preprocessing import NormalizeData
#this file is used to test the performance of a saved generator model

main_path = "C:\\Users\\Georgios\\Desktop\\4year project\\wespeDATA"
data_loader = DataLoader("cycleGANtrial", main_path)

imgs=data_loader.load_data(domain="A", batch_size=10, patch_dimension = (100,100), is_testing=True)
imgs_tensor = K.variable(imgs)

#load the model
generator = generator_network(imgs[0].shape, name = "Test_Generator")

model_path="C:\\Users\\Georgios\\Desktop\\4year project\\code\\Wespe-Keras\\models\\4_0.h5"
generator.load_weights(model_path)


for i in range(imgs.shape[0]):
    image=np.expand_dims(imgs[i,:,:,:], axis=0)
    fake_B_image = generator.predict(image)
    fake_B_image = NormalizeData(fake_B_image)
    plt.figure()
    plt.imshow(fake_B_image[0])