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
from skimage.measure import compare_ssim as ssim

class evaluator(object):
    
    def __init__(self, img_shape=(100, 100, 3), model=None, model_name=None, epoch=None, num_batch=None):
        
        self.data_loader = DataLoader()
        self.img_shape = img_shape
        
        if model_name:
            self.model_name = model_name
            self.model = generator_network(self.img_shape, name="Test_Gen")
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
    
    def objective_test(self, batch_size=None):
        phone_imgs, dslr_imgs = self.data_loader.load_paired_data(batch_size=batch_size)
        dslr_imgs = dslr_imgs.astype('float32') 
        #print(dslr_imgs.dtype)
        
        fake_dslr_images = self.model.predict(phone_imgs)
        #print(fake_dslr_images.dtype)
        
        batch_size=phone_imgs.shape[0]
        total_ssim=0
        for i in range(batch_size):
            total_ssim+=ssim(fake_dslr_images[i,:,:,:], dslr_imgs[i,:,:,:], multichannel=True)
        
        mean_ssim = total_ssim/batch_size
        print("Sample mean SSIM ---------%05f--------- " %(mean_ssim))
        
        #db_ssim = 10*np.log10(mean_ssim)
        return mean_ssim
        
            

#new_eval = evaluator(model_name="0_850.h5")
#print(new_eval.objective_test(1000))
    