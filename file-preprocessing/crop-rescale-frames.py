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
import tqdm

import matplotlib
import matplotlib.pyplot as plt

import cv2

from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('source', '', 'path to directory with frames')
flags.DEFINE_string('output', '','path to output directory')
flags.DEFINE_string('trial', 'XXX', 'trial id')
flags.DEFINE_integer('buffer', 30, 'adds a buffer to the cropping')

FLAGS(sys.argv)

SOURCE_DIR = FLAGS.source
OUTPUT_DIR = FLAGS.output
TRIAL_ID = FLAGS.trial # In case we want to combine all of the images into one directory

DATA_DIR = os.path.join(SOURCE_DIR, TRIAL_ID)

all_frames = [i for i in os.listdir(DATA_DIR) if i != '.DS_Store']
all_frames.sort()

if not os.path.isdir(OUTPUT_DIR):
	os.mkdir(OUTPUT_DIR)

# Iterate through all of the images and process them
for frame in tqdm.tqdm(all_frames):
	image = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, frame)), cv2.COLOR_RGB2BGR)

	# Get the image shape and determine how much to crop by
	image_shape = image.shape
	square_dim = min([image.shape[0], image.shape[1]])

	# Crop the edges and then square
	left_crop = image_shape[1]/2-1 - image_shape[0]/2 - FLAGS.buffer
	right_crop = image_shape[1]/2-1 + image_shape[0]/2 + FLAGS.buffer
	new_image = image[:, floor(left_crop):ceil(right_crop), :]
	new_image = cv2.resize(new_image, (square_dim, square_dim))
	matplotlib.image.imsave(os.path.join(OUTPUT_DIR, TRIAL_ID + '_' + frame), new_image)

print(f"Cropping and scaling complete. Original dimensions were {str(image.shape)}. New dimensions are {str(new_image.shape)}.")

