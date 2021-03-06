# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:03:33 2019

@author: Georgios
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:12:49 2019

@author: Georgios
"""

import scipy

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras_contrib.losses.dssim import DSSIMObjective


# define layer
#layer = InstanceNormalization(axis=-1)
from glob import glob
import imageio #gif creation

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add
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
from keras.initializers import glorot_normal, RandomNormal
from keras.models import Model
from preprocessing import gauss_kernel, rgb2gray, NormalizeData
#from architectures import resblock

from loss_functions import  total_variation, binary_crossentropy, vgg_loss, ssim
#from keras_radam import RAdam
from keras.applications.vgg19 import VGG19
from test_performance import evaluator

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

import warnings
warnings.filterwarnings("ignore")

from ROIextraction import cropper


class FeedBackGAN():
    def __init__(self, patch_size=(100,100)):
        # Input shape
        self.img_rows = patch_size[0]
        self.img_cols = patch_size[1]
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # Calculate output shape of D (PatchGAN)
        patch = int(np.ceil(self.img_shape[0] / 2**4))
        self.disc_patch = (patch, patch, 1)
        
        #Feedback Settings
        #test image paths
        self.test_image_dir="data/test phone images/"
        self.test_image_paths=glob(self.test_image_dir+"*")
        self.MSE_weights=np.ones(128)
        self.sum_of_weights=np.sum(self.MSE_weights)
        
        #details for gif creation featuring the progress of the training.
        #self.gif_batch_size=5
        #self.gif_frames_per_sample_interval=3
        #self.gif_images = [[] for i in range(self.gif_batch_size)]
        
        #manual logs
        #this will be changed using tensorboard
        self.log_TrainingPoint=[]
        self.log_D_colorloss=[]
        self.log_D_textureloss=[]
        self.log_G_colorloss=[]
        self.log_G_textureloss=[]
        self.log_ReconstructionLossA=[]
        self.log_ReconstructionLossB=[]
        self.log_TotalVariance=[]
        self.log_sample_ssim_time_point=[]
        self.log_sample_ssim=[]
            
        
        # Configure data loader
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))
        
        #set the blurring and texture discriminator settings
        self.kernel_size=23
        self.std = 3
        self.blur_kernel_weights = gauss_kernel(self.kernel_size, self.std, self.channels)
        self.texture_weights = np.expand_dims(np.expand_dims(np.expand_dims(np.array([0.2989, 0.5870, 0.1140]), axis=0), axis=0), axis=-1)
        #print(self.texture_weights.shape)
        
        #set the optimiser
        optimizer = Adam(0.0002, beta_1=0.5)
        #optimizer = RAdam()
        
       
        
        
        # Build and compile the discriminators
        
        self.D_color = self.discriminator_network(name="Color_Discriminator", preprocess = "blur")
        self.D_texture = self.discriminator_network(name="Texture_Discriminator", preprocess = "gray")
        
        
        self.D_color.compile(loss='mse', loss_weights=[0.5], optimizer=optimizer, metrics=['accuracy'])
        
        self.D_texture.compile(loss='mse', loss_weights=[0.5], optimizer=optimizer, metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.G = self.generator_network(filters = 128, name = "Forward_Generator_G")
        self.F = self.generator_network(filters = 64, name = "Backward_Generator_F")
        
        #instantiate the VGG model
        self.vgg_model = VGG19(weights='imagenet', include_top=False, input_shape = self.img_shape)
        self.layer_name="block2_conv2" #128 filters in the feature map
        self.inter_VGG_model = Model(inputs=self.vgg_model.input, outputs=self.vgg_model.get_layer(self.layer_name).output)
        
        self.inter_VGG_model.trainable=False
        
        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        #img_A_vgg = vgg_model(img_A)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.G(img_A)
        #identity_B = self.G(img_B)
        fake_A = self.F(img_B)
        
        # Translate images back to original domain
        reconstr_A = self.F(fake_B)
        reconstr_A_vgg = self.inter_VGG_model(reconstr_A)
        
        reconstr_B = self.G(fake_A)
        reconstr_B_vgg = self.inter_VGG_model(reconstr_B)
        

        # For the combined model we will only train the generators
        self.D_color.trainable = False
        self.D_texture.trainable = False

        # Discriminators determines validity of translated images
        valid_A_color = self.D_color(fake_B)
        valid_A_texture = self.D_texture(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A,img_B],
                              outputs=[valid_A_color, valid_A_texture, reconstr_A_vgg, reconstr_B_vgg, fake_B])
        
        
        #ssim_loss = DSSIMObjective()
        self.combined.compile(loss=['mse', 'mse', self.weighted_MSE, self.weighted_MSE, total_variation],
                            loss_weights=[0.5, 0.5, 1, 1, 0.5],
                            optimizer=optimizer)
        
        print(self.combined.summary())
        
    
    def weighted_MSE(self, y_true, y_pred):
        result=0
        #filters=shape[3]
        
        for i in range(128):
            result+=self.MSE_weights[i]*K.mean(K.square(y_true[:,:,:,i]-y_pred[:,:,:,i]))
        
        result=result/self.sum_of_weights
        
        return result
    
    def generator_network(self, filters, name):
        
        def resblock(feature_in, filters, num):
            
            init = RandomNormal(stddev=0.02)
            
            temp =  Conv2D(filters, (3, 3), strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_1' %num), kernel_initializer = init)(feature_in)
            temp = LeakyReLU(alpha=0.2)(temp)
            temp =  Conv2D(filters, (3, 3), strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_2' %num), kernel_initializer = init)(temp)
            temp = LeakyReLU(alpha=0.2)(temp)
            
            return Add()([temp, feature_in])
        
        init = RandomNormal(stddev=0.02)
        
        image=Input(self.img_shape)
        x = Lambda(lambda x: 2.0*x - 1.0, output_shape=lambda x:x)(image)
        b1_in = Conv2D(filters, (9,9), strides = 1, padding = 'SAME', name = 'CONV_1', activation = 'relu', kernel_initializer = init)(x)
        #b1_in = LeakyReLU(alpha=0.2)(b1_in)
        b1_in = Activation('relu')(b1_in)
        #b1_in = relu()(b1_in)
        # residual blocks
        b1_out = resblock(b1_in, filters, 1)
        b2_out = resblock(b1_out, filters, 2)
        b3_out = resblock(b2_out, filters, 3)
        b4_out = resblock(b3_out, filters, 4)
        
        # conv. layers after residual blocks
        temp = Conv2D(filters, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_2', kernel_initializer=init)(b4_out)
        #temp = BatchNormalization(axis=-1)(temp)
        temp = Activation('relu')(temp)
        #temp = LeakyReLU(alpha=0.2)(temp)
        
        temp = Conv2D(filters, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_3', kernel_initializer=init)(temp)
        #temp = BatchNormalization(axis=-1)(temp)
        temp = Activation('relu')(temp)
        #temp = LeakyReLU(alpha=0.2)(temp)
        
        temp = Conv2D(filters, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_4', kernel_initializer=init)(temp)
        #temp = BatchNormalization(axis=-1)(temp)
        temp = Activation('relu')(temp)
        #temp = LeakyReLU(alpha=0.2)(temp)
        
        temp = Conv2D(3, (7,7) , strides = 1, padding = 'SAME', name = 'CONV_5', kernel_initializer=init)(temp)
        #temp = Activation('sigmoid')(temp)
        #temp = Lambda(lambda x: K.clip(x, 0, 1), output_shape=lambda x:x)(temp)
        
        temp = Activation('tanh')(temp)
        temp = Lambda(lambda x: 0.5*x + 0.5, output_shape=lambda x:x)(temp)
        
        return Model(inputs=image, outputs=temp, name=name)


    def discriminator_network(self, name, preprocess = 'gray'):
        #The main modification from the original approach is the use of the InstanceNormalisation layer
        
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
        
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        #in_image = Input(shape=image_shape)
        # C64
        d = Lambda(lambda x: 2.0*x - 1.0, output_shape=lambda x:x)(image_processed)
        
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        #d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        # define model
        return Model(inputs = image, outputs = patch_out, name = name)
    
    
    def logger(self,):
        fig, axs = plt.subplots(2, 2, figsize=(6,8))
        
        ax = axs[0,0]
        ax.plot(self.log_TrainingPoint, self.log_D_colorloss, label="D_color")
        ax.plot(self.log_TrainingPoint, self.log_D_textureloss, label="D_texture")
        ax.legend()
        ax.set_title("Discriminator Adv losses")
        
        ax = axs[0,1]
        ax.plot(self.log_TrainingPoint, self.log_G_colorloss, label="G_color")
        ax.plot(self.log_TrainingPoint, self.log_G_textureloss, label="G_texture")
        ax.legend()
        ax.set_title("Generator Adv losses")
        
        ax = axs[1,0]
        ax.plot(self.log_TrainingPoint, self.log_ReconstructionLossA, label="reconstruction A" )
        ax.plot(self.log_TrainingPoint, self.log_ReconstructionLossB, label="reconstruction B" )
        ax.set_title("Cycle-Content loss")
        
        ax = axs[1,1]
        ax.plot(self.log_TrainingPoint, self.log_TotalVariance)
        ax.set_title("Total Variation loss")
        
        fig.savefig("progress/log.png")
        
        fig, axs = plt.subplots(1,1)
        ax=axs
        ax.plot(self.log_sample_ssim_time_point, self.log_sample_ssim)
        ax.set_title("sample SSIM value")
        fig.savefig("progress/sample_ssim.png")
        
        plt.close('all')
        
        
    

    def train(self, epochs, batch_size=1, sample_interval=50):
        #every sample_interval batches, the model is saved and sample images are generated and saved
        
        
        start_time = datetime.datetime.now()
        
        try:
            
            #gif_batch = self.data_loader.load_data(domain="A", batch_size = self.gif_batch_size, is_testing = True)
            
            # Adversarial loss ground truths
            #valid = np.ones((batch_size,1))
            #fake = np.zeros((batch_size,1))
            
            #Adversarial ground truth for patches corresponding to certain receptive fields
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)
            
            
            #instantiate the evaluator
            performance_evaluator = evaluator(model=self.G, img_shape=self.img_shape)
            
            #evaluate the baseline SSIM value (mean SSIM between the phone dataset and canon dataset)
            #baseline_SSIM = performance_evaluator.objective_test(baseline=True)
            #print("Baseline SSIM value: %05f" % (baseline_SSIM))
    
            for epoch in range(epochs):
                for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                    
                    
                        
                        # ----------------------
                        #  Train Discriminators
                        # ----------------------
        
                        # Translate images to opposite domain
                        fake_B = self.G.predict(imgs_A)
                        
                        """
                        #get self.gif_frames_per_sample_interval fake gif frames in sample_interval batches
                        if batch_i % int(sample_interval/self.gif_frames_per_sample_interval)==0:
                            fake_gif_batch = self.G.predict(gif_batch)
                            for i in range(self.gif_batch_size):
                                self.gif_images[i].append(fake_gif_batch[i])
                        """
        
                        # Train the discriminators (original images = real / translated = Fake)
                        dcolor_loss_real = self.D_color.train_on_batch(imgs_B, valid)
                        dcolor_loss_fake = self.D_color.train_on_batch(fake_B, fake)
                        dcolor_loss = 0.5 * np.add(dcolor_loss_real, dcolor_loss_fake)
                        self.log_D_colorloss.append(dcolor_loss[0])
                        
        
                        dtexture_loss_real = self.D_texture.train_on_batch(imgs_B, valid)
                        dtexture_loss_fake = self.D_texture.train_on_batch(fake_B, fake)
                        dtexture_loss = 0.5 * np.add(dtexture_loss_real, dtexture_loss_fake)
                        self.log_D_textureloss.append(dtexture_loss[0])
        
                        # Total disciminator loss
                        d_loss = 0.5 * np.add(dcolor_loss, dtexture_loss)
                        #d_loss = dcolor_loss
        
                        # ------------------
                        #  Train Generators
                        # ------------------
        
                        # Train the generators
                        imgs_A_vgg = self.inter_VGG_model.predict(imgs_A)
                        imgs_B_vgg = self.inter_VGG_model.predict(imgs_B)
                        
                        g_loss = self.combined.train_on_batch([imgs_A,imgs_B], [valid, valid,
                                                                imgs_A_vgg, imgs_B_vgg, imgs_A])
                        
                        self.log_G_colorloss.append(g_loss[1])
                        self.log_G_textureloss.append(g_loss[2])
                        self.log_ReconstructionLossA.append(g_loss[3])
                        self.log_ReconstructionLossB.append(g_loss[4])
                        self.log_TotalVariance.append(g_loss[5])
                        training_time_point = epoch+batch_i/self.data_loader.n_batches
                        self.log_TrainingPoint.append(np.around(training_time_point,3))
    
        
                        elapsed_time = datetime.datetime.now() - start_time
        
                        # Plot the progress
                        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f,  TV: %05f] time: %s " \
                                                                                % ( epoch, epochs,
                                                                                    batch_i, self.data_loader.n_batches,
                                                                                    d_loss[0], 100*d_loss[-1],
                                                                                    g_loss[0],
                                                                                    np.mean(g_loss[1:3]),
                                                                                    np.mean(g_loss[3:5]),
                                                                                    g_loss[5],
                                                                                    elapsed_time))
        
                        # If at save interval => save generated image samples
                        if batch_i % sample_interval == 0:
                            
                            """update the attributes of the performance_evaluator class"""
                            performance_evaluator.model = self.G
                            performance_evaluator.epoch = epoch
                            performance_evaluator.num_batch = batch_i
                            
                            """save the model"""
                            model_name="{}_{}.h5".format(epoch, batch_i)
                            self.G.save("models/"+model_name)
                            print("Epoch: {} --- Batch: {} ---- model saved".format(epoch, batch_i))
                            
                            """generation of perceptual results"""
                            #performance_evaluator.perceptual_test(5) 
                            
                            """SSIM based evaluation on a batch of test data"""
                            #calculate mean SSIM on approximately 10% of the test data
                            mean_sample_ssim = performance_evaluator.objective_test(500)
                            print("Sample mean SSIM ---------%05f--------- " %(mean_sample_ssim))
                            log_sample_ssim_time_point = epoch+batch_i/self.data_loader.n_batches
                            self.log_sample_ssim_time_point.append(np.around(log_sample_ssim_time_point,3))
                            self.log_sample_ssim.append(mean_sample_ssim)
                            
                            """logger"""
                            self.logger()
                            
                            
                        
                        """
                        #save the gifs every two sample intervals
                        if batch_i % (10*sample_interval) == 0 and batch_i!=0:
                            #save the gif images every 5 sample intervals for inspection
                             
                            #generator predicts values just outside [0,1] in the beginning of the training. Clip it to [0,1]
                            gif_images = np.clip(np.array(self.gif_images), 0, 1)*255.
                            
                            #avoid data type conversion warning
                            gif_images = gif_images.astype('uint8') 
                            
                            #save the generated gifs
                            for i in range(self.gif_batch_size):
                                imageio.mimsave('progress/gif_image_{}.gif'.format(i), gif_images[i])
                        """
                        
                        if batch_i % int(self.data_loader.n_batches/2) == 0 and not(epoch==0 and batch_i==0):
                            """update the SSIM evolution graph saved in the file progress"""
                            
                            #update the attributes of the performance_evaluator class
                            performance_evaluator.model = self.G
                            performance_evaluator.epoch = epoch
                            performance_evaluator.num_batch = batch_i
                            
                            #calculate the mean SSIM on test data
                            total_mean_ssim = performance_evaluator.objective_test(2000)
                            print("Mean SSIM (entire test dataset) ---------%05f--------- " %(total_mean_ssim))
                            
                            #save the value
                            performance_evaluator.ssim_vals.append(np.abs(np.around(total_mean_ssim, decimals=3)))
                            
                            #save the time point of the training
                            training_time_point = epoch+batch_i/self.data_loader.n_batches
                            performance_evaluator.training_points.append(np.around(training_time_point, 2))
                            
                            #update the SSIM evolution graph using the new point
                            fig = plt.figure()
                            #ax = fig.add_subplot(1, 1, 1)
                            num_values_saved = len(performance_evaluator.ssim_vals)
                            plt.plot(np.array(performance_evaluator.training_points), np.array(performance_evaluator.ssim_vals), color='blue', label="SSIM")
                            plt.plot(np.array(performance_evaluator.training_points), np.ones(num_values_saved)*0.9, color = 'red', label="target SSIM")
                            plt.plot(np.array(performance_evaluator.training_points), np.ones(num_values_saved)*0.750454, color = 'green', label="baseline SSIM")
                            plt.title("mean SSIM vs training epochs")
                            plt.legend()
                            fig.savefig("progress/ssim_curve.png")
                            plt.close('all')
             
                """call the cropper utility on epoch end"""
                #1.) generate the perceptual results
                #2.) select bad regions
                #3.) crop the batches
                #4.) Measure the activations
                #5.) Update the weighted MSE loss parameters
                #6.) Move to the next epoch
                
                #generate perceptual results using the test_performace class
                new_eval = evaluator()
                
                for i in range(10):
                    new_eval.enhance_image(self.test_image_paths[i], model_name, reference=False)
                    
                #instantiate the cropper class located in the ROI extraction file
                new_cropper=cropper()
                #prompt the user to select the regions of the generated images that they think are not enhanced properly
                new_cropper.generate_ROI_regions(enhanced_path="generated_images/")
                
                #check whether feedback was given
                feedback=False
                for key in new_cropper.valid_rectangles:
                    if new_cropper.valid_rectangles[key]:
                        feedback=True
                    
                if feedback==False:
                    #use the current mse weights if there is no feedback given
                    continue
                else:
                    #if feedback from the human supervisor was given, then:
                    
                    #extract the selected patches from the original images
                    new_cropper.extract_patches(raw_path=self.test_image_dir)
                    
                    #use those patches to find the activations of the selected layer of the VGG19 network
                    normalised_activations = new_cropper.inspect_activation() #get the normalised activations: vector range [0,1]
                    
                    #parameters alpha and beta determine how much the weights are influenced by the human feeback
                    #decreasing beta and increasing alpha increases the responsiveness to human feedback
                    #large alpha values destablise the training
                    
                    alpha=0.8
                    beta=0.2
                    #update the new MSE weights using the activations of the inadequately reconstructed patches
                    self.MSE_weights = alpha * self.MSE_weights + beta * normalised_activations
                    self.sum_of_weights = np.sum(self.MSE_weights) #get the sum of the MSE weights to normalise the weighted MSE loss properly
                    
            
            
            
                        
        except KeyboardInterrupt:
            print("Training has been interrupted")
        


patch_size=(100, 100)
epochs=15
batch_size=8
sample_interval =100 #after sample_interval batches save the model and generate sample images
    
gan = FeedBackGAN(patch_size=patch_size)
gan.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)