# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:12:50 2019

@author: Georgios
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

path="data/trainA/18.JPG"
x = cv2.imread(path)
print(np.amin(x), np.amax(x))