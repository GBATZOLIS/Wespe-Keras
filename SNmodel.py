from __future__ import print_function, division
import scipy

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras_contrib.losses.dssim import DSSIMObjective
from SN import ConvSN2D, DenseSN

# define layer
#layer = InstanceNormalization(axis=-1)

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
from keras.initializers import RandomNormal, Orthogonal
from keras.models import Model
from preprocessing import gauss_kernel, rgb2gray, NormalizeData

from loss_functions import  total_variation, binary_crossentropy, vgg_loss, ssim, hinge_D_loss, hinge_G_loss
#from keras_radam import RAdam
from keras.applications.vgg19 import VGG19
from test_performance import evaluator

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

import warnings
warnings.filterwarnings("ignore")


class WespeGAN():
    def __init__(self, patch_size=(100,100)):
        # Input shape
        self.img_rows = patch_size[0]
        self.img_cols = patch_size[1]
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        
        #details for gif creation featuring the progress of the training.
        self.gif_batch_size=10
        self.gif_frames_per_sample_interval=5
        self.gif_images = [[] for i in range(self.gif_batch_size)]
        
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
        
        self.log_sample_ssim=[]
        self.log_sample_ssim_training_point=[]
        
        # Configure data loader
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))
        
        #set the blurring and texture discriminator settings
        self.kernel_size=21
        self.std = 3
        self.blur_kernel_weights = gauss_kernel(self.kernel_size, self.std, self.channels)
        self.texture_weights = np.expand_dims(np.expand_dims(np.expand_dims(np.array([0.2989, 0.5870, 0.1140]), axis=0), axis=0), axis=-1)
        #print(self.texture_weights.shape)
        
        #set the optimiser
        #optimizer = Adam(0.0002, beta_1=0.5)
        optimizerD = Adam(0.0002, beta_1=0.5)
        #optimizerG = Adam(0.0001, beta_1=0.5)
        #optimizer = RAdam()
        
        # Build and compile the discriminators
        
        self.D_color = self.discriminator_network(name="Color_Discriminator", preprocess = "blur")
        self.D_texture = self.discriminator_network(name="Texture_Discriminator", preprocess = "gray")
        
        
        self.D_color.compile(loss='mae', loss_weights=[0.5], optimizer=optimizerD, metrics=['accuracy'])
        
        self.D_texture.compile(loss='mae', loss_weights=[0.5], optimizer=optimizerD, metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        
        self.vgg_model = VGG19(weights='imagenet', include_top=False, input_shape = self.img_shape)
        self.layer_name="block2_conv2"
        self.inter_VGG_model = Model(inputs=self.vgg_model.input, outputs=self.vgg_model.get_layer(self.layer_name).output)
        self.inter_VGG_model.trainable=False
        
        # Build the generators
        self.G = self.generator_network(filters=64, name = "Forward_Generator_G")
        self.F = self.generator_network(filters=64, name = "Backward_Generator_F")

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.G(img_A)
        fake_A = self.F(img_B)
        #identity_B = self.G(img_B)
        #fake_A = self.g_BA(img_B)
        
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
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A_color, valid_A_texture, reconstr_A_vgg, reconstr_B_vgg, fake_B])
        
        
        #ssim_loss = DSSIMObjective()
        self.combined.compile(loss=['mae', 'mae', 'mae', 'mae', total_variation],
                            loss_weights=[0.5, 0.5, 1, 1, 0.5],
                            optimizer=optimizerD)
        
        print(self.combined.summary())
        
    
    def generator_network(self, filters, name):
        
        def resblock(feature_in, filters, num):
            
            #init = RandomNormal(stddev=0.02)
            init = Orthogonal(gain=1)
            
            temp =  ConvSN2D(filters, (3, 3), strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_1' %num), kernel_initializer = init)(feature_in)
            #temp = BatchNormalization(axis=-1)(temp)
            temp = LeakyReLU(alpha=0.2)(temp)
            
            temp =  ConvSN2D(filters, (3, 3), strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_2' %num), kernel_initializer = init)(temp)
            #temp = BatchNormalization(axis=-1)(temp)
            temp = LeakyReLU(alpha=0.2)(temp)
            
            return Add()([temp, feature_in])
        
        #init = RandomNormal(stddev=0.02)
        init = Orthogonal(gain=1)
        
        image=Input(self.img_shape)
        y = Lambda(lambda x: 2.0*x - 1.0, output_shape=lambda x:x)(image)
        
        b1_in = ConvSN2D(filters, (9,9), strides = 1, padding = 'SAME', name = 'CONV_1', activation = 'relu', kernel_initializer = init)(y)
        b1_in = LeakyReLU(alpha=0.2)(b1_in)
        # residual blocks
        b1_out = resblock(b1_in, filters, 1)
        b2_out = resblock(b1_out, filters, 2)
        b3_out = resblock(b2_out, filters, 3)
        b4_out = resblock(b3_out, filters, 4)
        
        # conv. layers after residual blocks
        temp = ConvSN2D(filters, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_2', kernel_initializer=init)(b4_out)
        #temp = BatchNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        temp = ConvSN2D(filters, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_3', kernel_initializer=init)(temp)
        #temp = BatchNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        temp = ConvSN2D(filters, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_4', kernel_initializer=init)(temp)
        #temp = BatchNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        temp = Conv2D(3, (9,9) , strides = 1, padding = 'SAME', name = 'CONV_5', kernel_initializer=init)(temp)
        temp = Activation('tanh')(temp)
        
        temp = Lambda(lambda x: 0.5*x + 0.5, output_shape=lambda x:x)(temp)
        
        return Model(inputs=image, outputs=temp, name=name)


    def discriminator_network(self, name, preprocess = 'gray'):
        #The main modification from the original approach is the use of the InstanceNormalisation layer
        
        #init = RandomNormal(stddev=0.02)
        init = Orthogonal(gain=1)
        
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
        d = Lambda(lambda x: 2.0*x - 1.0, output_shape=lambda x:x)(image_processed)
        temp = ConvSN2D(48, (11,11), strides = 4, padding = 'SAME', name = 'CONV_1', kernel_initializer = init)(d)
        #temp = BatchNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        # conv layer 2
        temp = ConvSN2D(128, (5,5), strides = 2, padding = 'SAME', name = 'CONV_2', kernel_initializer = init)(temp)
        #temp = BatchNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        # conv layer 3
        temp = ConvSN2D(192, (3,3), strides = 1, padding = 'SAME', name = 'CONV_3', kernel_initializer = init)(temp)
        #temp = BatchNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        # conv layer 4
        temp = ConvSN2D(192, (3,3), strides = 1, padding = 'SAME', name = 'CONV_4', kernel_initializer = init)(temp)
        #temp = BatchNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        # conv layer 5
        temp = ConvSN2D(128, (3,3), strides = 2, padding = 'SAME', name = 'CONV_5', kernel_initializer = init)(temp)
        #temp = BatchNormalization(axis=-1)(temp)
        temp = LeakyReLU(alpha=0.2)(temp)
        
        # FC layer 1
        fc_in = Flatten()(temp)
        
        fc_out = Dense(512)(fc_in)
        fc_out = LeakyReLU(alpha=0.2)(fc_out)
        
        # FC layer 2
        logits = Dense(1)(fc_out)
        #prob = Activation('sigmoid')(logits)
        #probability = sigmoid(logits)
        
        return Model(inputs=image, outputs=logits, name=name)
    
    
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
        ax.plot(self.log_TrainingPoint, self.log_ReconstructionLossA, label="domain A")
        ax.plot(self.log_TrainingPoint, self.log_ReconstructionLossB, label="domain B")
        ax.set_title("Cycle-Content loss")
        
        ax = axs[1,1]
        ax.plot(self.log_TrainingPoint, self.log_TotalVariance)
        ax.set_title("Total Variation loss")
        
        fig.savefig("progress/log.png")
        
        fig, axs = plt.subplots(1,1)
        ax=axs
        training_points=len(self.log_sample_ssim_training_point)
        ax.plot(self.log_sample_ssim_training_point, self.log_sample_ssim, label="sample ssim")
        ax.plot(self.log_sample_ssim_training_point, [0.750454]*training_points, label="baseline")
        ax.plot(self.log_sample_ssim_training_point, [0.9]*training_points, label="target")
        ax.legend()
        ax.set_title("sample SSIM value")
        fig.savefig("progress/sample_ssim.png")
        
        plt.close('all')
        
        
        
        
    

    def train(self, epochs, batch_size=1, sample_interval=50):
        #every sample_interval batches, the model is saved and sample images are generated and saved
        
        
        start_time = datetime.datetime.now()
        
        try:
            # Adversarial loss ground truths
            valid = np.ones((batch_size,1))
            fake = np.zeros((batch_size,1))
            
            #instantiate the evaluator
            performance_evaluator = evaluator(model=self.G, img_shape=self.img_shape)
            
            #evaluate the baseline SSIM value (mean SSIM between the phone dataset and canon dataset)
            #baseline_SSIM = performance_evaluator.objective_test(baseline=True)
            #print("Baseline SSIM value: %05f" % (baseline_SSIM))
            
            Dsteps=2
            for epoch in range(epochs):
                for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                    
                    
                        
                        # ----------------------
                        #  Train Discriminators
                        # ----------------------
                        
                        dcolor_avg_loss=0
                        dtexture_avg_loss=0
                        for i in range(Dsteps):
                            # Translate images to opposite domain
                            disc_imgs_A = imgs_A[i*int(batch_size/Dsteps): (i+1)*int(batch_size/Dsteps)]
                            disc_imgs_B = imgs_B[i*int(batch_size/Dsteps): (i+1)*int(batch_size/Dsteps)]
                            disc_valid =  valid[i*int(batch_size/Dsteps): (i+1)*int(batch_size/Dsteps)]
                            disc_fake = fake[i*int(batch_size/Dsteps): (i+1)*int(batch_size/Dsteps)]
                            
                            disc_fake_B = self.G.predict(disc_imgs_A)
            
                            # Train the discriminators (original images = real / translated = Fake)
                            dcolor_loss_real = self.D_color.train_on_batch(disc_imgs_B, disc_valid)
                            dcolor_loss_fake = self.D_color.train_on_batch(disc_fake_B, disc_fake)
                            dcolor_loss = 0.5 * np.add(dcolor_loss_real, dcolor_loss_fake)
                            dcolor_avg_loss+=dcolor_loss
                            
                            
            
                            dtexture_loss_real = self.D_texture.train_on_batch(disc_imgs_B, disc_valid)
                            dtexture_loss_fake = self.D_texture.train_on_batch(disc_fake_B, disc_fake)
                            dtexture_loss = 0.5 * np.add(dtexture_loss_real, dtexture_loss_fake)
                            dtexture_avg_loss+=dtexture_loss
                        
                        
                        dcolor_avg_loss = dcolor_avg_loss/Dsteps
                        dtexture_avg_loss = dtexture_avg_loss/Dsteps
                        
                        self.log_D_colorloss.append(dcolor_avg_loss[0])
                        self.log_D_textureloss.append(dtexture_avg_loss[0])
                        
                        # Total disciminator loss
                        d_loss = 0.5 * np.add(dcolor_avg_loss, dtexture_avg_loss)
                        
        
                        # ------------------
                        #  Train Generators
                        # ------------------
                        
                        gen_imgs_A=imgs_A[: int(batch_size/Dsteps)]
                        gen_imgs_B=imgs_B[: int(batch_size/Dsteps)]
                        gen_valid = valid[: int(batch_size/Dsteps)]
                        imgs_A_vgg = self.inter_VGG_model.predict(gen_imgs_A)
                        imgs_B_vgg = self.inter_VGG_model.predict(gen_imgs_B)
                        
                        # Train the generators
                        g_loss = self.combined.train_on_batch([gen_imgs_A, gen_imgs_B], [gen_valid, gen_valid,
                                                                imgs_A_vgg, imgs_B_vgg, gen_imgs_A])
                        
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
                            self.G.save("models/{}_{}.h5".format(epoch, batch_i))
                            print("Epoch: {} --- Batch: {} ---- model saved".format(epoch, batch_i))
                            
                            """generation of perceptual results"""
                            #performance_evaluator.perceptual_test(5) 
                            
                            """SSIM based evaluation on a batch of test data"""
                            #calculate mean SSIM on approximately 10% of the test data
                            mean_sample_ssim = performance_evaluator.objective_test(500)
                            self.log_sample_ssim.append(mean_sample_ssim)
                            print("Sample mean SSIM ---------%05f--------- " %(mean_sample_ssim))
                            log_sample_ssim_time_point = epoch+batch_i/self.data_loader.n_batches
                            self.log_sample_ssim_training_point.append(np.around(log_sample_ssim_time_point,3))
                            
                            """logger"""
                            self.logger()
                        
                        if batch_i % int(self.data_loader.n_batches/4) == 0 and not(epoch==0 and batch_i==0):
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
                
                        
        except KeyboardInterrupt:
            
            end_time=datetime.datetime.now()
            print("Training was interrupted after %s" %(end_time-start_time))
            print("Training interruption details: epochs: {} --- batches: {}/{}".format(epoch, batch_i, self.data_loader.n_batches))
            print("Wait for the training final report to be generated.")
            
                        
        


patch_size=(100, 100)
epochs=10
batch_size=32
sample_interval = 50 #after sample_interval batches save the model and generate sample images
    
gan = WespeGAN(patch_size=patch_size)
gan.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)