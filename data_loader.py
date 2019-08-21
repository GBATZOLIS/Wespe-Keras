# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:24:53 2019

@author: Georgios
"""
  
import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, dataset_name, main_path, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.main_path = main_path
        
    
    def get_random_patch(self, img, patch_dimension):
        if img.shape[0]==patch_dimension[0] and img.shape[1]==patch_dimension[1]:
            return img
        
        else:
            image_shape=img.shape
            image_length = img.shape[0]
            image_width = img.shape[1]
            patch_length = patch_dimension[0]
            patch_width = patch_dimension[1]
            
            if (image_length >= patch_length) and (image_width >= patch_width):
                x_max=image_shape[0]-patch_dimension[0]
                y_max=image_shape[1]-patch_dimension[1]
                x_index=np.random.randint(x_max)
                y_index=np.random.randint(y_max)
            else:
                print("Error. Not valid patch dimensions")
            
            return img[x_index:x_index+patch_dimension[0], y_index:y_index+patch_dimension[1], :]
        
    def load_data(self, domain, patch_dimension=None, batch_size=1, is_testing=False):
        data_type = r"train%s" % domain if not is_testing else "test%s" % domain
        path = glob(r'%s\\%s\\%s\\*' % (self.main_path, self.dataset_name, data_type))
        batch_images = np.random.choice(path, size=batch_size)
        
        if patch_dimension==None:
            #if the patch dimension is not specified, use the training dimensions
            patch_dimension = self.img_res
            
        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            img = self.get_random_patch(img, patch_dimension)   
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob(r'%s\\%s\\%sA\\*' % (self.main_path, self.dataset_name, data_type))
        path_B = glob(r'%s\\%s\\%sB\\*' % (self.main_path, self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                #img_A = scipy.misc.imresize(img_A, self.img_res)
                #img_B = scipy.misc.imresize(img_B, self.img_res)
                if (img_A.shape[0]>self.img_res[0]) or (img_A.shape[1]>self.img_res[1]):
                    img_A=self.get_random_patch(img_A, patch_dimension = self.img_res)
                    img_B=self.get_random_patch(img_B, patch_dimension = self.img_res)

                #if not is_testing and np.random.random() > 0.5:
                #        img_A = np.fliplr(img_A)
                #        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return plt.imread(path).astype(np.float)

