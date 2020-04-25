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

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import imageio

source_directory_of_frames = sys.argv[1]
output_directory_for_rescaled_image = sys.argv[2]
frame_prefix = sys.argv[3] # In case we want to combine all of the images into one directory

all_frames = os.listdir(source_directory_of_frames)

start_time = time.time()
if not os.path.isdir(output_directory_for_rescaled_image):
	os.mkdir(output_directory_for_rescaled_image)

# Iterate through all of the images and process them
frame_counter = 0
for frame in all_frames:
	if frame == '.DS_Store':
		continue
	if frame_counter % 1000 == 0:
		print('Processed: ' + str(frame_counter) + '/' + str(len(all_frames)) + ' (' + str(ceil(time.time()-start_time)) + ')')
	frame_counter += 1

	image = imageio.imread(os.path.join(source_directory_of_frames, frame))

	# Get the image shape and determine how much to crop by
	image_shape = image.shape
	amount_to_crop_x = (image_shape[1]-image_shape[0])/3 # This appears to be good from testing for 1280 x 720 images

	cropped_image = image[:, floor(amount_to_crop_x):ceil(image_shape[1] - amount_to_crop_x), :]

	# Now we rescale the cropped image for our final image that we will be using
	sf = 0.33
	rescaled_image = resize(
		cropped_image, 
		(int(cropped_image.shape[0] * sf), int(cropped_image.shape[1]*sf))
	)

	# Save the final image
	matplotlib.image.imsave(os.path.join(output_directory_for_rescaled_image, frame_prefix + '_' + frame), rescaled_image)
