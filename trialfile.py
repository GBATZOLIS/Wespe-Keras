# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:12:50 2019

@author: Georgios
"""

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal


def generator_network(name):
        def resnet_block(n_filters, input_layer):
            # weight initialization
            init = RandomNormal(stddev=0.02)
            # first layer convolutional layer
            g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
            g = InstanceNormalization(axis=-1)(g)
            g = Activation('relu')(g)
            # second convolutional layer
            g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
            g = InstanceNormalization(axis=-1)(g)
            # concatenate merge channel-wise with input layer
            g = Add()([g, input_layer])
            return g
        
        init = RandomNormal(stddev=0.02)
        
        image=Input((64,64,3))
        
        y = Lambda(lambda x: 2.0*x - 1.0, output_shape=lambda x:x)(image)
        
        g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(y)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # d128
        g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # d256
        g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        
        for _ in range(3):
            g = resnet_block(256, g)
        # u128
        g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # u64
        g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # c7s1-3
        g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        out_image = Activation('tanh')(g)
        out_image = Lambda(lambda x: 0.5*x + 0.5, output_shape=lambda x:x)(out_image)
        # define model
        model = Model(inputs = image, outputs = out_image, name=name)
        
        return model

import numpy as np

x=np.array([1,30, 210, 23210, 3, 20])
y=np.array([1,3])
print(x[y])
