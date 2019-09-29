# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:12:50 2019

@author: Georgios
"""

#This file is used to extract the regions where the users things the network did not perform well

# import the necessary packages
#import argparse
import cv2
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.models import Model
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

class cropper(object):
    
    def __init__(self):
        self.valid_rectangles={}
        self.refPt = []
        self.cropping = False
        self.current_filename=""
        self.image=0
        self.all_square_boxes=None
    
    def createSquareBoxes(self, rect_box, size):
        m,n,channels=rect_box.shape
        
        stepsX=int(m/size)
        stepsY=int(n/size)    
        
        cropped_boxes=[]
        for i in range(stepsX):
            for j in range(stepsY):
                crop_box=rect_box[size*i:size*(i+1), size*j:size*(j+1), :]
                cropped_boxes.append(crop_box)
        
        
        if m-stepsX*size>0:
            for j in range(stepsY):
                crop_box=rect_box[-size:, size*j:size*(j+1), :]
                cropped_boxes.append(crop_box)
        
        if n-stepsY*size>0:
            for i in range(stepsX):
                crop_box=rect_box[size*i:size*(i+1), -size:, :]
                cropped_boxes.append(crop_box)
        
        
        return cropped_boxes
    
    def click_and_crop(self, event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping, current_filename, valid_rectangles
        
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        
        if event == cv2.EVENT_LBUTTONDOWN:
            #refPt.append((x,y)) 
            self.refPt = [(x, y)]
            self.valid_rectangles[self.current_filename].append((x,y))
            self.cropping = True
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.refPt.append((x, y))
            self.cropping = False
            
            self.valid_rectangles[self.current_filename].append((x,y))
            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", self.image)
    
    
    def generate_ROI_regions(self, enhanced_path="generated_images/"):
        image_paths=glob(enhanced_path+"*.jpg")
        for image_path in image_paths:
            self.current_filename=os.path.basename(image_path)
            self.valid_rectangles[self.current_filename]=[]
            self.image = cv2.imread(image_path)
            #image = cv2.imread(args["image"])
            self.clone = self.image.copy()
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", self.click_and_crop)
            cv2.imshow("image", self.image)
            cv2.waitKey(0)
            
            # close all open windows
            cv2.destroyAllWindows()
        
    def extract_patches(self, raw_path="C:\\Users\\Georgios\\Desktop\\4year project\\wespeDATA\\dped\\dped\\iphone\\test_data\\full_size_test_images\\", patch_size=100):
        boxes=[]
        if self.valid_rectangles:
            for key,val in self.valid_rectangles.items():
                image = plt.imread(raw_path+key, format="RGB")
                print(image.shape)
                if len(val)>0:
                    for i in range(0,len(val),2):
                        x1=val[i]
                        x2=val[i+1]
                        #print(x1)
                        #print(x2)
                        cropped_box = image[min(x1[0],x2[0]):max(x1[0],x2[0]), min(x1[1],x2[1]):max(x1[1],x2[1]), :]
                        #plt.figure()
                        #plt.imshow(cropped_box)
                        m,n,channels=cropped_box.shape
                        if m>patch_size and n>patch_size:
                            boxes.append(cropped_box)
                        else:
                            continue
                
                else:
                    continue
            
            all_square_boxes=[]
            for rect_box in boxes:
                cropped_boxes=self.createSquareBoxes(rect_box, patch_size)
                for cropped_box in cropped_boxes:
                    all_square_boxes.append(cropped_box)
            all_square_boxes=np.array(all_square_boxes)
            all_square_boxes=all_square_boxes[:,:,:,:3]
            self.all_square_boxes=all_square_boxes
            
            #return all_square_boxes
        
        else:
            return "Cropping operation must proceed the patch extraction"
    
    def save(self):
        if self.all_square_boxes.any():
            square_boxes=self.all_square_boxes
            i=1
            for box in square_boxes:
                plt.imsave("C:\\Users\\Georgios\\Desktop\\enhanced big images\\experiment ROI\\100by100patches\\cropped_patch_%d.png" % (i), box)
                i+=1
    
    def inspect_activation(self, model="VGG19", layer_name="block2_conv2", patch_size=100):
        vgg_model = VGG19(weights='imagenet', include_top=False, input_shape = (patch_size,patch_size,3))
        inter_VGG_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(layer_name).output)
        #print(self.all_square_boxes.shape)
        #print(type(self.all_square_boxes))
        featureMaps=inter_VGG_model.predict(self.all_square_boxes)
        #print(featureMaps.shape)
        avgFilterNorms=np.zeros(featureMaps.shape[-1])
        for featureMap in featureMaps:
            filterNorms=np.zeros(featureMaps.shape[-1])
            for i in range(featureMaps.shape[-1]):
                norm=np.linalg.norm(featureMap[:,:,i], ord='fro')
                filterNorms[i]=norm
            avgFilterNorms+=filterNorms
        
        avgFilterNorms=avgFilterNorms/featureMaps.shape[0]
        
        plt.figure()
        x=np.ones(len(avgFilterNorms))+0.1*avgFilterNorms/np.amax(avgFilterNorms)
        plt.plot(x/np.amax(x))
        return x/np.amax(x)
        
    
        
    


cropper1=cropper()
cropper1.generate_ROI_regions()
cropper1.extract_patches()
cropper1.save()
_=cropper1.inspect_activation()











        
    
    


