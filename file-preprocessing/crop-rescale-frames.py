'''
crop-rescale-frames

The purpose of this script is to try and reduce the size of the images that we are processing as
we are developing our models. It will make the data more accessible and easier to work with in 
general for testing an such if we do some preprocessing of the images. This preprocessing
will include things like:

- cropping the sides of the endoscopic view
- rescaling the images
'''

import sys
import subprocess
import glob
from math import floor, ceil
import shutil
import re
import os
import datetime
import time

import matplotlib
import matplotlib.pyplot as plt

import cv2

SOURCE_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
TRIAL_ID = sys.argv[3] # In case we want to combine all of the images into one directory
IMG_SIZE = 200

DATA_DIR = os.path.join(SOURCE_DIR, TRIAL_ID)

all_frames = [i for i in os.listdir(DATA_DIR) if i != '.DS_Store']
all_frames.sort()


start_time = time.time()
if not os.path.isdir(OUTPUT_DIR):
	os.mkdir(OUTPUT_DIR)

# Iterate through all of the images and process them
frame_counter = 0
for frame in all_frames:
	if frame_counter % 1000 == 0:
		print('Processed: ' + str(frame_counter) + '/' + str(len(all_frames)) + ' (' + str(ceil(time.time()-start_time)) + ')')
	frame_counter += 1

	image = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, frame)), cv2.COLOR_RGB2BGR)

	# Get the image shape and determine how much to crop by
	image_shape = image.shape
	amount_to_crop_x = (image_shape[1]-image_shape[0])/3 # This appears to be good from testing for 1280 x 720 images

	cropped_image = image[:, floor(amount_to_crop_x):ceil(image_shape[1] - amount_to_crop_x), :]
	resized_image = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE))

	# Save the final image
	matplotlib.image.imsave(os.path.join(OUTPUT_DIR, TRIAL_ID + '_' + frame), resized_image)


print(f"Cropping and scaling complete. Original dimensions were {str(image.shape)}. New dimensions are {str(resized_image.shape)}. Crop was from {str((floor(amount_to_crop_x), ceil(image_shape[1] - amount_to_crop_x)))}")

