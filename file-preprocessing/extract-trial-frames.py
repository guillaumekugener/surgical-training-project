'''
This is script is used to generate a zip file of the frames for a particular trial
'''

## IMPORT MODULES
import sys
import subprocess
import glob
import math
import shutil
import re
import os
import datetime

import pandas as pd

trial_name = sys.argv[1]
latest_timestamps_sheet = '/Users/guillaumekugener/Documents/USC/USC_docs/ml/surgical-training-project/data/trial_time_points - Sheet1.tsv'
output_directory = '/Users/guillaumekugener/Documents/USC/USC_docs/ml/surgical-training-project/dev/'
directory_with_videos = '/Users/guillaumekugener/Downloads/'

BUFFER = 0 # How many seconds to buffer the start and stop frames

# Get the video file we want and the start and stop times
latest_time_stamp_data = pd.read_csv(latest_timestamps_sheet, delimiter='\t')
trial_row = latest_time_stamp_data.loc[latest_time_stamp_data['Trial Name'] == trial_name]

if trial_row.shape[0] != 1:
	print(str(trial_row.shape[0]) + ' rows were returned this is wrong')


video_file_name = os.path.join(directory_with_videos, trial_row['Video'].values[0] + '.mp4')
trial_start = trial_row['Start Time'].values[0]
trial_end = trial_row['End Time'].values[0]

# Get the start and stop positions 
print('The video being converted to frames is: ' + video_file_name)
print('The output directory will be: ' + output_directory) + '/'

# The line below can be used to determine the total number of frames in a video
# ffmpeg -i <file_name> -map 0:v:0 -c copy -f null -
ffmpeg_command = [
	'ffmpeg', 
	'-i', 
	video_file_name, 
	'-q:v 1', 
	# '-ss',
	# str(trial_start), 
	'-to',
	str(trial_end),
	os.path.join(output_directory, trial_name, 'frame_%08d.jpeg')
]

print(' '.join(ffmpeg_command))

## Run ffmpeg to generate the images
make_output_directory = subprocess.Popen(' '.join(['mkdir', output_directory + trial_name]), shell=True, stdout=subprocess.PIPE)
conversion_process = subprocess.Popen(' '.join(ffmpeg_command), shell=True, stdout=subprocess.PIPE)
conversion_process.communicate()

# Remove the frames not in the time range (we will simply estimate based on the start)
# We estimate the start and stop frames based on the time stamps in Start and End Time columns
# We know the frame rate is 30 FPS, so we can deduce where it actually is from that

# One challenge - how do we know the exact stop frame (hemostasis could be achieved without a clear signal
# within the video...)
video_frame_rate = 30
start_time = datetime.datetime.strptime(trial_start, '%H:%M:%S')
estSF = (start_time.hour * 60 * 60 + start_time.minute * 60 + start_time.second) * video_frame_rate

for i in range(1, max(estSF - 10, 1)):
	file_name = '00000000' + str(i)
	while len(file_name) > 8:
		file_name = file_name[1:]
	os.remove(os.path.join(output_directory, trial_name, 'frame_' + file_name + '.jpeg'))

