'''
This script generates a tsv mapping frame to a binary classification of whether or not the
endoscope is "redded-out" in that frame.
'''

import sys
import zipfile
import scipy as sc
import numpy as np
import imageio
import os
import shutil
import csv
import re
import time

from os.path import isfile, join

### SETUP 
# Here, we want to have the path to the directory that 
# contains all of the frames for a single video in zip files
source_dir = sys.argv[1]
tsv_output_name = sys.argv[2]

print('Generating redding out output for ' + source_dir)


# The threshold we are using for now is 5 (for channel 2). 
channel_2_threshold = 5
channel_2_range_threshold = 5


# Get all of the zip files with the images for this video
zip_of_images = [join(source_dir, f) for f in os.listdir(source_dir) if isfile(join(source_dir, f)) and re.search('zip$', f)]

# This part takes a while...
all_frames = None

frame_values = []
average_channel_values = []
t1 = time.time()

# Iterate through all the zip files 
for zi in range(len(zip_of_images[:])):
	z = zip_of_images[zi]

	print('Processing zip ' + str(zi+1) + '/' + str(len(zip_of_images)) + '...')

	video_zip=zipfile.ZipFile(z)
	all_frames = video_zip.namelist()

	# For each frame in this zip, take the mean value for each channel
	for f in all_frames[:]:
		img = imageio.imread(video_zip.open(f))
		average_channel_values.append(np.mean(np.mean(img, axis=1), axis=0)) # Take the mean value of the pixels

	print('Finished process: ' + str(time.time() - t1))

# Convert the channel values to arrays for manipulation later on
channel_1 = [i[0] for i in average_channel_values]
channel_2 = [i[1] for i in average_channel_values]
channel_3 = [i[2] for i in average_channel_values]

# Get the names of the frames for mapping later on
frame_names = []
for zi in range(len(zip_of_images[:])):
	z = zip_of_images[zi]
	video_zip=zipfile.ZipFile(z)

	frame_names = frame_names + video_zip.namelist()

# Identify the frames that pass the threshold for channel 2
bleeding_channel_2 = [i for i in range(len(channel_2)) if channel_2[i] < channel_2_threshold]

# Using the crude criteria above, we will have false positives (we may also have false negatives)
# but at this time I have not looked into false negatives as much. One way to limit the false
# positives is to require that redding out has a duration of at least n frames. In the snippet
# below, we apply this filtering: we first get the range of frames that are redded out and then
# only keep the cases where the length of the range is above some threshold value.
ranges_of_bleeding = []
current_range = [-1, -1]
for j in bleeding_channel_2[:]:
	if current_range[0] == -1:
	    current_range = [j, j]
	elif (current_range[1] + 1) == j:
	    current_range[1] = j
	else:
	    ranges_of_bleeding.append((current_range[0], current_range[1]))
	    current_range = [j, j]

threshold_of_range = 5
redded_out_frames_prune = []
for rb in ranges_of_bleeding:
	if (rb[1] - rb[0] > channel_2_range_threshold):
	    redded_out_frames_prune = redded_out_frames_prune + list(range(rb[0], rb[1]+1))

# Convert the indicies to frame names and save the tsv
redded_out_frames = [frame_names[i] for i in redded_out_frames_prune]

print(str(len(redded_out_frames)) + ' frames passed redding out criteria')

# Write the output to a file
with open(join(source_dir, tsv_output_name), 'wt') as out_file:
	tsv_writer = csv.writer(out_file, delimiter='\t')
	tsv_writer.writerow(['frame', 'redded_out'])

	for fn in frame_names:
		if fn in redded_out_frames:
			tsv_writer.writerow([fn, 1])
		else:
			tsv_writer.writerow([fn, 0])






