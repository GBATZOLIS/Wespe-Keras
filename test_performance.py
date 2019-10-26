# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 20:13:03 2019

@author: Georgios
"""
import keras
import keras.backend as K


from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import *
from keras.models import Model

from data_loader import DataLoader
import numpy as np
#from architectures import generator_network
import matplotlib.pyplot as plt
from preprocessing import NormalizeData
from glob import glob
import os
from skimage.measure import compare_ssim as ssim
#from skvideo.measure import niqe
import cv2 as cv

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

def generator_network(img_shape, filters, name):
        
        def resblock(feature_in, filters, num):
            
            temp =  Conv2D(filters, (3, 3), strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_1' %num))(feature_in)
            temp = LeakyReLU(alpha=0.2)(temp)
            temp =  Conv2D(filters, (3, 3), strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_2' %num))(temp)
            temp = LeakyReLU(alpha=0.2)(temp)
            
            return Add()([temp, feature_in])
        
        
        image=Input(img_shape)
        x = Lambda(lambda x: 2.0*x - 1.0, output_shape=lambda x:x)(image)
        b1_in = Conv2D(filters, (9,9), strides = 1, padding = 'SAME', name = 'CONV_1', activation = 'relu')(x)
        #b1_in = LeakyReLU(alpha=0.2)(b1_in)
        b1_in = Activation('relu')(b1_in)
        #b1_in = relu()(b1_in)
        # residual blocks
        b1_out = resblock(b1_in, filters, 1)
        b2_out = resblock(b1_out, filters, 2)
        b3_out = resblock(b2_out, filters, 3)
        b4_out = resblock(b3_out, filters, 4)
        
        # conv. layers after residual blocks
        temp = Conv2D(filters, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_2')(b4_out)
        temp = Activation('relu')(temp)
        
        temp = Conv2D(filters, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_3')(temp)
        temp = Activation('relu')(temp)
        
        temp = Conv2D(filters, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_4')(temp)
        temp = Activation('relu')(temp)
        
        temp = Conv2D(3, (7,7) , strides = 1, padding = 'SAME', name = 'CONV_5')(temp)
        
        temp = Activation('tanh')(temp)
        temp = Lambda(lambda x: 0.5*x + 0.5, output_shape=lambda x:x)(temp)
        
        return Model(inputs=image, outputs=temp, name=name)


class evaluator(object):
    
    def __init__(self, img_shape=(100, 100, 3), model=None, model_name=None, epoch=None, num_batch=None):
        
        self.data_loader = DataLoader()
        self.img_shape = img_shape
        
        if model_name:
            self.model_name = model_name
            self.model = generator_network(self.img_shape, name="Test_Gen") #this has to be the same as the generator in the model file
            self.model.load_weights("models/%s" % (model_name))
        else:
            self.model_name = model_name
            self.model = model
            self.epoch = epoch
            self.num_batch = num_batch
            
            self.training_points=[] #training time locations where mean SSIM value on test data has been calculated
            self.ssim_vals = [] #calculated SSIM values on test data
    
    def perceptual_test(self, batch_size):
        phone_imgs, dslr_imgs = self.data_loader.load_paired_data(batch_size=batch_size)
        
        fake_dslr_images = self.model.predict(phone_imgs)
        
        i=0
        for phone, fake_dslr, real_dslr in zip(phone_imgs, fake_dslr_images, dslr_imgs):
            #phone = NormalizeData(phone)
            phone = np.expand_dims(phone, axis=0)
            np.clip(fake_dslr, 0, 1, out=fake_dslr)
            fake_dslr = np.expand_dims(fake_dslr, axis=0)
            #real_dslr = NormalizeData(real_dslr)
            real_dslr = np.expand_dims(real_dslr, axis=0)
            
            
            all_imgs = np.concatenate([phone, fake_dslr, real_dslr])
            titles = ['phone', 'fake DSLR ', 'real DSLR']
            fig, axs = plt.subplots(1, 3, figsize=(6,8))
            
            j=0
            for ax in axs.flat:
                ax.imshow(all_imgs[j])
                ax.set_title(titles[j])
                j += 1
            
            if self.model_name:
                fig.savefig("generated_images/test%s.png" % (i))
            else:
                fig.savefig("generated_images/%d_%d_%d.png" % (self.epoch, self.num_batch, i))
            
            i+=1
        
        plt.close('all')
        print("Perceptual results have been generated")
    
    def objective_test(self, batch_size=None, baseline=False, quantiser=True):
        def quantise(image):
            image=255*image
            image=np.around(image)
            image=image/255
            return image.astype('float32')
        
        phone_imgs, dslr_imgs = self.data_loader.load_paired_data(batch_size=batch_size)
        dslr_imgs = dslr_imgs.astype('float32') #necessary typecasting
        
        if baseline:
            fake_dslr_images=phone_imgs
        else:
            fake_dslr_images = self.model.predict(phone_imgs)
        
        if quantiser:
            fake_dslr_images=quantise(fake_dslr_images)
            
        
        batch_size=phone_imgs.shape[0]
        total_ssim=0
        for i in range(batch_size):
            total_ssim+=ssim(fake_dslr_images[i,:,:,:], dslr_imgs[i,:,:,:], multichannel=True)
        
        mean_ssim = total_ssim/batch_size
        return mean_ssim
    
    def no_reference_test(self, model_name, batch_size=None, baseline=False):
        image_test_path=glob("C:\\Users\\Georgios\\Desktop\\4year project\\wespeDATA\\dped\\dped\\iphone\\test_data\\full_size_test_images\\*")
        phone_imgs=[]
        for path in image_test_path:
            image = plt.imread(path).astype(np.float)
            phone_imgs.append(image)
        phone_imgs=np.array(phone_imgs)
        
        img_shape = phone_imgs[0].shape
        self.model_name = model_name
        self.model = generator_network(img_shape, name="Test Generator") #this has to be the same as the generator in the model file
        self.model.load_weights("models/%s" % (model_name))
        
        Y_phone_imgs=[]
        for i in range(phone_imgs.shape[0]):
            Y =  0.299*phone_imgs[i,:,:,0] + 0.587*phone_imgs[i,:,:,1] + 0.114*phone_imgs[i,:,:,2]
            Y_phone_imgs.append(Y)
        
        Y_phone_imgs=np.array(Y_phone_imgs)
        niqe_val = niqe(Y_phone_imgs)
        return niqe_val
    
    def enhance_image(self, img_path, model_name, reference=True):
        
        phone_image = self.data_loader.load_img(img_path) #load image
        
        phone_image = phone_image[0]
        phone_image = phone_image[0:1500, 0:1500, :] 
        img_shape = phone_image.shape #get dimensions to build the suitable model
        
        
        self.model_name = model_name
        self.model = generator_network(img_shape, filters=128, name="Test Generator") #this has to be the same as the generator in the model file
        self.model.load_weights("models/%s" % (model_name))
        
        
            
        fake_dslr_image = self.model.predict(np.expand_dims(phone_image, axis=0))
        print(np.amax(fake_dslr_image))
        print(np.amin(fake_dslr_image))
        #width=phone_image[0].shape[0]
        #height = phone_image[0].shape[1]
        if reference:
            fig, axs = plt.subplots(1, 2)
            #fig.set_size_inches(10, 8)
            ax = axs[0]
            ax.imshow(phone_image)
            ax.set_title("phone image")
            
            ax = axs[1]
            ax.imshow(fake_dslr_image[0])
            ax.set_title("enahnced phone image")
            #filename=os.path.basename(img_path)
            #fig.savefig("sample images\\"+filename, dpi = )
            #plt.show()
            #plt.close()
        
        else:
            filename=os.path.basename(img_path)
            filename=filename.split(".")[0]+".png"
            file_save="generated_images/"+filename
            plt.imsave(file_save, fake_dslr_image[0])
        
           
      

new_eval = evaluator()
image_paths=glob("C:\\Users\\Test-PC\\Desktop\\Github\\dped\\iphone\\test_data\\full_size_test_images\\*")
#print(image_paths)
model_name="4_0.h5"
for i in range(len(image_paths)):
    new_eval.enhance_image(image_paths[i], model_name, reference=True)



