from __future__ import print_function, division
import scipy

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# define layer
#layer = InstanceNormalization(axis=-1)

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, GlobalAveragePooling2D,Flatten, BatchNormalization, LeakyReLU, Lambda, DepthwiseConv2D
from keras.activations import relu,tanh,sigmoid
from keras.initializers import glorot_normal
from keras.models import Model
from preprocessing import gauss_kernel, rgb2gray, NormalizeData
from architectures import resblock

from loss_functions import  total_variation, binary_crossentropy, vgg_loss
from keras.applications.vgg19 import VGG19
from test_performance import evaluator

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

class WespeGAN():
    def __init__(self, patch_size=(100,100)):
        # Input shape
        self.img_rows = patch_size[0]
        self.img_cols = patch_size[1]
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        #self.main_path = "C:\\Users\\Georgios\\Desktop\\4year project\\wespeDATA"
        #self.dataset_name = "cycleGANtrial"
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))
        
        #configure perceptual loss 
        self.content_layer = 'block1_conv2'
        
        #set the blurring settings
        self.kernel_size=21
        self.std = 3
        self.blur_kernel_weights = gauss_kernel(self.kernel_size, self.std, self.channels)
        self.texture_weights = np.expand_dims(np.expand_dims(np.expand_dims(np.array([0.2989, 0.5870, 0.1140]), axis=0), axis=0), axis=-1)
        #print(self.texture_weights.shape)
        
        #set the optimiser
        optimizer = Adam(0.0001, beta_1=0.5)
        
        # Build and compile the discriminators
        
        self.D_color = self.discriminator_network(name="Color_Discriminator", preprocess = "blur")
        self.D_texture = self.discriminator_network(name="Texture_Discriminator", preprocess = "gray")
        
        
        self.D_color.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])
        
        self.D_texture.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.G = self.generator_network(name = "Forward_Generator_G")
        self.F = self.generator_network(name = "Backward_Generator_F")

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.G(img_A)
        identity_B = self.G(img_B)
        #fake_A = self.g_BA(img_B)
        
        # Translate images back to original domain
        reconstr_A = self.F(fake_B)
        #reconstr_B = self.g_AB(fake_A)
        

        # For the combined model we will only train the generators
        self.D_color.trainable = False
        self.D_texture.trainable = False

        # Discriminators determines validity of translated images
        valid_A_color = self.D_color(fake_B)
        valid_A_texture = self.D_texture(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A_color, valid_A_texture, reconstr_A, identity_B, fake_B])
        
        
        
        self.combined.compile(loss=[binary_crossentropy, binary_crossentropy, 'mae', 'mae', total_variation],
                            loss_weights=[10, 5, 5, 2, 0.1],
                            optimizer=optimizer)
        
        print(self.combined.summary())
        
        

    def generator_network(self, name):
        
        image=Input(self.img_shape)
        b1_in = Conv2D(64, (9,9), strides = 1, padding = 'SAME', name = 'CONV_1', activation = 'relu', kernel_initializer = glorot_normal())(image)
        #b1_in = relu()(b1_in)
        # residual blocks
        b1_out = resblock(b1_in, 1)
        b2_out = resblock(b1_out, 2)
        b3_out = resblock(b2_out, 3)
        b4_out = resblock(b3_out, 4)
        
        # conv. layers after residual blocks
        temp = Conv2D(64, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_2', kernel_initializer=glorot_normal())(b4_out)
        #temp = BatchNormalization()(temp)
        temp = Activation('relu')(temp)
        
        temp = Conv2D(64, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_3', kernel_initializer=glorot_normal())(b4_out)
        #temp = BatchNormalization()(temp)
        temp = Activation('relu')(temp)
        
        temp = Conv2D(64, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_4', kernel_initializer=glorot_normal())(b4_out)
        #temp = BatchNormalization()(temp)
        temp = Activation('relu')(temp)
        
        temp = Conv2D(3, (9,9) , strides = 1, padding = 'SAME', name = 'CONV_5', kernel_initializer=glorot_normal())(b4_out)
        #temp = Activation('tanh')(temp)
        
        return Model(inputs=image, outputs=temp, name=name)


    def discriminator_network(self, name, preprocess = 'gray'):
        
        image = Input(self.img_shape)
        
        
        if preprocess == 'gray':
            #convert to grayscale image
            print("Discriminator-texture")
            
            #output_shape=(image.shape[0], image.shape[1], 1)
            gray_layer=Conv2D(1, (1,1), strides = 1, padding = "SAME", use_bias=False, name="Gray_layer")
            image_processed=gray_layer(image)
            gray_layer.set_weights([self.texture_weights])
            gray_layer.trainable = False
            
            #image_processed=Lambda(rgb2gray, output_shape = output_gray_shape)(image)
            #print(image_processed.shape)
            #image_processed = rgb_to_grayscale(image)
            
        elif preprocess == 'blur':
            print("Discriminator-color (blur)")
            
            g_layer = DepthwiseConv2D(self.kernel_size, use_bias=False, padding='same')
            image_processed = g_layer(image)
            
            g_layer.set_weights([self.blur_kernel_weights])
            g_layer.trainable = False

            
        else:
            print("Discriminator-color (none)")
            image_processed = image
            
        # conv layer 1 
        temp = Conv2D(48, (11,11), strides = 4, padding = 'SAME', name = 'CONV_1', kernel_initializer = glorot_normal())(image_processed)
        temp = InstanceNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        # conv layer 2
        temp = Conv2D(96, (5,5), strides = 2, padding = 'SAME', name = 'CONV_2', kernel_initializer = glorot_normal())(temp)
        temp = InstanceNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        # conv layer 3
        temp = Conv2D(192, (3,3), strides = 1, padding = 'SAME', name = 'CONV_3', kernel_initializer = glorot_normal())(temp)
        temp = InstanceNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        # conv layer 4
        temp = Conv2D(192, (3,3), strides = 1, padding = 'SAME', name = 'CONV_4', kernel_initializer = glorot_normal())(temp)
        temp = InstanceNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        # conv layer 5
        temp = Conv2D(96, (3,3), strides = 2, padding = 'SAME', name = 'CONV_5', kernel_initializer = glorot_normal())(temp)
        temp = InstanceNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        # FC layer 1
        fc_in = Flatten()(temp)
        
        fc_out = Dense(1024)(fc_in)
        fc_out = LeakyReLU(alpha=0.2)(fc_out)
        
        # FC layer 2
        logits = Dense(1)(fc_out)
        #probability = sigmoid(logits)
        
        return Model(inputs=image, outputs=logits, name=name)
    
    
    

    def train(self, epochs, batch_size=1, sample_interval=50):
        #every sample_interval batches, the model is saved and sample images are generated and saved
        
        
        start_time = datetime.datetime.now()
        
        
        
        try:
            
            # Adversarial loss ground truths
            valid = np.ones((batch_size,1))
            fake = np.zeros((batch_size,1))
            
            #instantiate the evaluator
            performance_evaluator = evaluator(model=self.G, img_shape=self.img_shape)
    
            for epoch in range(epochs):
                for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                    
                    
                        
                        # ----------------------
                        #  Train Discriminators
                        # ----------------------
        
                        # Translate images to opposite domain
                        fake_B = self.G.predict(imgs_A)
                        #fake_A = self.g_BA.predict(imgs_B)
        
                        # Train the discriminators (original images = real / translated = Fake)
                        dcolor_loss_real = self.D_color.train_on_batch(imgs_B, valid)
                        dcolor_loss_fake = self.D_color.train_on_batch(fake_B, fake)
                        dcolor_loss = 0.5 * np.add(dcolor_loss_real, dcolor_loss_fake)
        
                        dtexture_loss_real = self.D_texture.train_on_batch(imgs_B, valid)
                        dtexture_loss_fake = self.D_texture.train_on_batch(fake_B, fake)
                        dtexture_loss = 0.5 * np.add(dtexture_loss_real, dtexture_loss_fake)
        
                        # Total disciminator loss
                        d_loss = 0.5 * np.add(dcolor_loss, dtexture_loss)
                        #d_loss = dcolor_loss
        
                        # ------------------
                        #  Train Generators
                        # ------------------
        
                        # Train the generators
                        g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid,
                                                                imgs_A, imgs_B, imgs_A])
        
                        elapsed_time = datetime.datetime.now() - start_time
        
                        # Plot the progress
                        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, ID:%05f,  TV: %05f] time: %s " \
                                                                                % ( epoch, epochs,
                                                                                    batch_i, self.data_loader.n_batches,
                                                                                    d_loss[0], 100*d_loss[1],
                                                                                    g_loss[0],
                                                                                    np.mean(g_loss[1:3]),
                                                                                    g_loss[3],
                                                                                    g_loss[4],
                                                                                    g_loss[5],
                                                                                    elapsed_time))
        
                        # If at save interval => save generated image samples
                        if batch_i % sample_interval == 0:
                            print("Epoch: {} --- Batch: {} ---- saved".format(epoch, batch_i))
                            
                            performance_evaluator.model = self.G
                            performance_evaluator.epoch = epoch
                            performance_evaluator.num_batch = batch_i
                            
                            performance_evaluator.perceptual_test(5) #generation of perceptual results
                            
                            mean_sample_ssim = performance_evaluator.objective_test(400) #metric based evaluation on test data
                            
                            performance_evaluator.ssim_vals.append(np.abs(np.around(mean_sample_ssim, decimals=3)))
                            
                            self.G.save("models/{}_{}.h5".format(epoch, batch_i))
                            
                            r"""
                            if batch_i % (3*sample_interval) == sample_interval:
                                db_ssim = performance_evaluator.objective_test()
                                print("Test data mean SSIM -----%05f------ in (DB)" %(db_ssim))
                            """
                        if epoch<1:
                            if batch_i % (2*sample_interval) == 0 and batch_i!=0:
                                fig = plt.figure()
                                ax = fig.add_subplot(1, 1, 1)
                                num_values_saved=len(performance_evaluator.ssim_vals)
                                ax.plot(np.arange(1, num_values_saved+1)*sample_interval, np.array(performance_evaluator.ssim_vals), color='blue')
                                ax.plot(np.arange(1, num_values_saved+1)*sample_interval, np.ones(num_values_saved)*0.9, color = 'red')
                                plt.title("mean sample SSIM vs number of batches")
                                fig.savefig("progress/ssim_curve.png")
                        
                        
        except KeyboardInterrupt:
            end_time=datetime.datetime.now()
            print("Training was interrupted after %s" %(end_time-start_time))
            print("Training interruption details: epochs: {} --- batches: {}/{}".format(epoch, batch_i, self.data_loader.n_batches))
            
            total_mean_ssim = performance_evaluator.objective_test()
            
            #print(performance_evaluator.ssim_vals)
            plt.figure()
            num_values_saved=len(performance_evaluator.ssim_vals)
            plt.plot(np.arange(1, num_values_saved+1)*sample_interval, np.array(performance_evaluator.ssim_vals))
            plt.title("mean sample SSIM vs number of batches")
            plt.show()
            
            self.G.save("models/KeyboardInterrupt_{}_{}%.h5".format(epoch, int(batch_i/self.data_loader.n_batches*100)))
            
            print("Report has been generated. Model has been saved.")
                        
        

if __name__ == '__main__':
    patch_size=(100, 100)
    epochs=50
    batch_size=5
    sample_interval = 50 #after sample_interval batches save the model and generate sample images
    
    gan = WespeGAN(patch_size=patch_size)
    gan.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)