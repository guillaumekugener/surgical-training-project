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

from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('timepoints', '', 'path to the timepoints sheet')
flags.DEFINE_string('output', '','path to output directory')
flags.DEFINE_string('trial', None, 'trial id')
flags.DEFINE_string('video', 'XXX', 'video file to be processed')
flags.DEFINE_integer('downsample', -1, 'downsample FPS')
flags.DEFINE_boolean('overwrite', False, 'overwrites the existing downsample')
flags.DEFINE_string('f1', 'XXX', 'F1 column from trial timepoints data (e.g: Park City 2019 Sim)')
flags.DEFINE_string('f2', 'XXX', 'F2 column from trial timepoints data (e.g: 01182019_170211)')

FLAGS(sys.argv)

latest_timestamps_sheet = FLAGS.timepoints
output_directory = FLAGS.output
video_file_name = FLAGS.video

BUFFER = 0 # How many seconds to buffer the start and stop frames

# If downsample is passed, we downsample the frame
VIDEO_ID = re.sub('\\.mp4$', '', re.sub('.*/', '', video_file_name))


if FLAGS.downsample > 0:
	# The we want to downsample the video files first (unless the file arleady exists)
	output_downsampled = os.path.join(output_directory, VIDEO_ID + '_downsampled.mp4')

	# Overwrite if told to or file does not exist
	if not os.path.isfile(output_downsampled) or FLAGS.overwrite:
		reduction_command = f"ffmpeg -i {video_file_name} -q 1 -r {FLAGS.downsample} {output_downsampled}"

		# Execute the command
		downsample_process = subprocess.Popen(reduction_command, shell=True, stdout=subprocess.PIPE)
		downsample_process.communicate()

	# Overrite the video_file_name as this is the file we are now processing
	video_file_name = output_downsampled

# Get all the trials matching this video
# Get the video file we want and the start and stop times
latest_time_stamp_data = pd.read_csv(latest_timestamps_sheet, delimiter='\t')

all_trials_in_video = latest_time_stamp_data.loc[(latest_time_stamp_data['Video']==VIDEO_ID) & (latest_time_stamp_data['F1']==FLAGS.f1) & (latest_time_stamp_data['F2']==FLAGS.f2)]
all_trials_in_video = [i for i in all_trials_in_video['Trial Name']]

# Trial to process
if FLAGS.trial is not None:
	all_trials_in_video = [FLAGS.trial]

print(f'Processing trials: {all_trials_in_video}')

zip_commands_all = []
# Process all trials in a video
for trial_name in all_trials_in_video:
	# Get all the trials for this video
	trial_row = latest_time_stamp_data.loc[latest_time_stamp_data['Trial Name'] == trial_name]

	if trial_row.shape[0] != 1:
		print(str(trial_row.shape[0]) + ' rows were returned this is wrong')
		break


	trial_start = trial_row['Start Time'].values[0]
	trial_end = trial_row['End Time'].values[0]

	# Get the start and stop positions 
	print('The video being converted to frames is: ' + video_file_name)
	print('The output directory will be: ' + output_directory)

	# The line below can be used to determine the total number of frames in a video
	# ffmpeg -i <file_name> -map 0:v:0 -c copy -f null -
	ffmpeg_command = [
		'ffmpeg', 
		'-i', 
		video_file_name, 
		'-q:v 1', 
		'-ss',
		str(trial_start), 
		'-to',
		str(trial_end), 
		os.path.join(output_directory, 'frames', trial_name + '_frame_%08d.jpeg')
	]

	print(' '.join(ffmpeg_command))

	## Run ffmpeg to generate the images
	# make_output_directory = subprocess.Popen(' '.join(['mkdir', output_directory + trial_name]), shell=True, stdout=subprocess.PIPE)
	conversion_process = subprocess.Popen(' '.join(ffmpeg_command), shell=True, stdout=subprocess.PIPE)
	conversion_process.communicate()

	# Zip file
	zip_command = f"zip {os.path.join(FLAGS.output, 'zips', trial_name)}-downsampled.zip {os.path.join(FLAGS.output, 'frames', trial_name)}*.jpeg"
	zip_commands_all.append(zip_command)
	# zip_process = subprocess.Popen(zip_command)
	# zip_process.communicate()

	# Remove the frames not in the time range (we will simply estimate based on the start)
	# We estimate the start and stop frames based on the time stamps in Start and End Time columns
	# We know the frame rate is 30 FPS, so we can deduce where it actually is from that

	# One challenge - how do we know the exact stop frame (hemostasis could be achieved without a clear signal
	# within the video...)
	# video_frame_rate = 30
	# start_time = datetime.datetime.strptime(trial_start, '%H:%M:%S')
	# estSF = (start_time.hour * 60 * 60 + start_time.minute * 60 + start_time.second) * video_frame_rate

	# for i in range(1, max(estSF - 10, 1)):
	# 	file_name = i.zfill(8)
	# 	os.remove(os.path.join(output_directory, trial_name, 'frame_' + file_name + '.jpeg'))

for c in zip_commands_all:
	print(c)
