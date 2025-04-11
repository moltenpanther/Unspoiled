#!/usr/bin/env python
# coding: utf-8

# ## Unspoiled - AI Model
# #### Ben Cobb

# Spring 2024

## Imports and Libraries
import cv2 as cv
import os
import time
import datetime
import requests
from PIL import Image
from ultralytics import YOLO

print("IMPORTS DONE!")
## Global Variables

# Change this to where you want to save your images!
# If we want to save them at all?
modelPath = "/Users/molte/OneDrive/Desktop/UAFS/~Spring 2024/Capstone/Unspoiled/AIModel.yaml"

deptNames = ["COLD", "PRODUCE", "PACKAGED"]
classes = ["MILK", "EGGCARTON", "CREAMER", "APPLE", "BANANA", "PEAR", "COUGHDROPS", "CHEEZIT", "SODA"]
'''
classIDs:
0 = "milk"
1 = "eggcarton"
2 = "creamer"
3 = "apple"
4 = "banana"
5 = "pear"
6 = "coughdrops"
7 = "cheezit"
8 = "soda"
'''

cmap = "gray"


## Methods
# Takes a photo
def doStuff():
    print("hello")


## Model Things

# Sets up the model
model = YOLO('yolov6n.yaml')

# Shows the model's info (obviously)
print(model.info())

# Path to our dataset (will be moved up to Variables eventually, here for testing)
yamlPath = "/Users/molte/YOLOv6/data/dataset.yaml"

# Trains the model (epochs small right now just for testing)
results = model.train(data=yamlPath, epochs=2, imgsz=640, batch_size=-1)
print(results)

import os
print(os.getcwd())



# Run inference with the YOLOv6n model on the 'bus.jpg' image
results = model("/Users/molte/YOLOv6/data/dataset/images/test/banana40080-15.png")

