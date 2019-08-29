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
#from skimage.measure import compare_ssim as ssim

class evaluator(object):
    
    def __init__(self, model_name, img_shape=(100, 100, 3)):
        
        self.data_loader = DataLoader()
        self.model_name = model_name
        self.generator = generator_network(img_shape, name="Test_Gen")
        self.generator.load_weights("models/%s" % (model_name))
    
    def perceptual_test(self, batch_size):
        phone_imgs, dslr_imgs = self.data_loader.load_paired_data(batch_size=batch_size)
        
        fake_dslr_images = self.generator.predict(phone_imgs)
        
        i=0
        for phone, fake_dslr, real_dslr in zip(phone_imgs, fake_dslr_images, dslr_imgs):
            phone = NormalizeData(phone)
            phone = np.expand_dims(phone, axis=0)
            fake_dslr = NormalizeData(fake_dslr)
            fake_dslr = np.expand_dims(fake_dslr, axis=0)
            real_dslr = NormalizeData(real_dslr)
            real_dslr = np.expand_dims(real_dslr, axis=0)
            
            
            all_imgs = np.concatenate([phone, fake_dslr, real_dslr])
            titles = ['phone', 'fake DSLR ', 'real DSLR']
            fig, axs = plt.subplots(1, 3, figsize=(6,8))
            
            j=0
            for ax in axs.flat:
                ax.imshow(all_imgs[j])
                ax.set_title(titles[j])
                j += 1
            
            fig.savefig("generated_images/test%s.png" % (i))
            
            i+=1
            
        print("Perceptual results have been generated")
            

new_eval = evaluator(model_name="0_400.h5")
new_eval.perceptual_test(10)
    