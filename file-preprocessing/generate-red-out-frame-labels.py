'''
This script generates a tsv mapping frame to a binary classification of whether or not the
endoscope is "redded-out" in that frame.

IDEAS: take the average across 9 boxes and then take the average of that

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
import pickle
import pandas as pd

from math import floor, ceil
from os.path import isfile, join

### SETUP 
# Here, we want to have the path to the directory that 
# contains all of the frames for a single video in zip files
zip_of_images = sys.argv[1]
tsv_output_name = sys.argv[2]
model_file_name = sys.argv[3]
model_header_file_name = sys.argv[4]

# TODO: clean this up because this will be annoying later.
# Give specific time stamps that we want to assess. We convert to frames and then only care about those frames
# This is in order to save a ton of time on the annotating (by only annotating frames of interest)
# start_time_seconds = int(sys.argv[5])
# end_time_seconds = int(sys.argv[6])
# batch_size = int(sys.argv[7])
# frame_rate = 30


print('Generating redding out output for ' + zip_of_images)

# The model we are using to call red out
loaded_model = pickle.load(open(model_file_name, 'rb'))

# Given that the order of headers might change over time, we run these next lines
# so that the model prediction uses the right order of the x values for the predictions
# TODO: could probably be avoided if we use pandas dataframes...
header_order = pd.read_csv(model_header_file_name, header=None)
model_header_order_iterate = [i for i in header_order[0]]
model_header_order_index = {}
for i in range(len(model_header_order_iterate)):
    model_header_order_index[model_header_order_iterate[i]] = i

# This part takes a while...
all_frames = None

frame_values = []
average_channel_values = []
t1 = time.time()

dimension_of_squares = 3

# We have to open one image to get the dimensions for all of them
v1z=zipfile.ZipFile(zip_of_images)
print(v1z.namelist()[1])
image_1 = imageio.imread(v1z.open(v1z.namelist()[1]))
dimensions_of_images = image_1.shape

dimension_of_squares = 3
x_dim, y_dim = dimensions_of_images[0]/dimension_of_squares, dimensions_of_images[1]/dimension_of_squares

cuts_to_make = []
for k in range(dimensions_of_images[2]):
    for i in range(dimension_of_squares):
        for j in range(dimension_of_squares):
            cuts_to_make.append((
                (int(x_dim * i), int(min(x_dim * (i + 1), dimensions_of_images[0]))),
                (int(y_dim * j), int(min(y_dim * (j + 1), dimensions_of_images[1]))),
                k
            ))


# We want to split the image into squares and then calculate the mean of the 
# pixel values for each square
t1 = time.time()
boxed_values = []
# This part takes a while...
all_frames = None

frame_values = []
average_channel_values = []

# relevant_sf = floor(start_time_seconds * frame_rate / batch_size)
# relevant_ef = ceil(end_time_seconds * frame_rate / batch_size)

# print(relevant_sf)
# print(relevant_ef)
# Iterate through all the zip files 

# # Skip ranges of fluff
# if zi not in range(relevant_sf, relevant_ef):
#     continue

video_zip=zipfile.ZipFile(zip_of_images)
all_frames = video_zip.namelist()

# For each frame in this zip, take the mean value for each channel
for f in all_frames[:]:
    if not re.search('jpeg$', f):
        print('Skipping non-image file')
        continue
    img = imageio.imread(video_zip.open(f))
    # average_channel_values.append(np.mean(np.mean(img, axis=1), axis=0)) # Take the mean value of the pixels
    # average_channel_values.append(np.mean(img[:,:,1])) # We are using channel 2, so only taking the mean here cuts the run time in half
    boxed_values_tmp = []
    for rr in cuts_to_make:
        boxed_values_tmp.append(np.mean(img[rr[0][0]:rr[0][1],rr[1][0]:rr[1][1],rr[2]]))
    boxed_values.append(boxed_values_tmp)
    
print('Finished process: ' + str(time.time() - t1))

# Generate the header
output_file_header = ['frame']
# for r in range(len(average_channel_values[0])):
#     output_file_header.append('channel_' + str(r))

for c in cuts_to_make:
    output_file_header.append('range_x' + str(c[0][0]) + '-' + str(c[0][1]) + 'y' + str(c[1][0]) + '-' + str(c[1][1]) + 'z' + str(c[2]))

output_file_header.append('red_out_prob')

# Create an index array for the headers
header_dict = {}
for i in range(len(output_file_header)):
    header_dict[output_file_header[i]] = i

# Get the names of the frames for mapping later on
frame_names = []
video_zip=zipfile.ZipFile(zip_of_images)
frame_names = frame_names + video_zip.namelist()
frame_names = [f for f in frame_names if re.search('jpeg$', f)]

# Write the output to a file
with open(join(tsv_output_name), 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(output_file_header)

    for i in range(len(frame_names)):
        fn = frame_names[i]
        row_info = [fn]
        
        # for j in range(len(average_channel_values[i])):
        #     row_info.append(average_channel_values[i][j])
        
        # This is the row info that the model needs    
        m_r_i = [None] * len(model_header_order_iterate)
        for j in range(len(boxed_values[i])):
            # The column that we are currently adding
            col_adding = output_file_header[len(row_info)]
            
            row_info.append(boxed_values[i][j])   

            # If out model uses this column, then we want to add it to the m_r_i
            # array for the prediction
            if col_adding in model_header_order_index:         
                m_r_i[model_header_order_index[col_adding]] = row_info[header_dict[col_adding]]

        # Predict the value and add to row_info
        row_info.append(loaded_model.predict_proba(np.array([m_r_i]))[0,1])
        tsv_writer.writerow(row_info)






