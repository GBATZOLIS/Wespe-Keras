# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:12:50 2019

@author: Georgios
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from preprocessing import *

path="data/trainA/18.JPG"
x = plt.imread(path).astype(np.float)
x = rgb2tanhRange(x)
print(np.amin(x), np.amax(x))

x = tanhRange2rgb(x)
print(np.amin(x), np.amax(x))