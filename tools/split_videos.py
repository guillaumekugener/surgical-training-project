import os
import shutil
import sys
import subprocess
import math
import progressbar

import numpy as np
import pandas as pd

import re
import random

from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('trial', None, 'trial id')
flags.DEFINE_string('tp_file_sheet', None, 'path to csv with trial time points')

flags.DEFINE_string('video_file', '', 'path to video file')
flags.DEFINE_string('output_full', '', 'directory where full video should be stored')
flags.DEFINE_string('output_splits', '', 'directory where the splits of the video should be stored')
flags.DEFINE_string('tmp', '', 'path to temporary directory for intermediate files (only need if padding)')
flags.DEFINE_boolean('padding', False, 'adds some random videos at the end to obscure the true length of the trial')
flags.DEFINE_integer('chunk_size', 15, 'size (in seconds) of segments to make')

FLAGS(sys.argv)

"""
Extract time points from file

Returns an array, since some videos are split
"""
def get_video_time_points(trial_id, tp_data):
	tp_data['tid_no_suffix'] = [re.sub('[a-z]+', '', i) for i in tp_data['Trial Name']]
	relevant_rows = tp_data[tp_data['tid_no_suffix']==re.sub('[a-z]+', '', trial_id)].reset_index()

	videos_to_process = []
	for i in range(relevant_rows.shape[0]):
		videos_to_process.append({
			'vid': relevant_rows['Trial Name'][i],
			'start': relevant_rows['Start Time'][i],
			'end': relevant_rows['End Time'][i]
		})

	return videos_to_process

"""Cut video trial given time stamps"""
def cut_video_trial(video_file_name, s, e, vid, output_directory, overwrite='-n'):
	out_f = os.path.join(output_directory, vid + '.mp4')

	ffmpeg_command = [
		'ffmpeg', 
		overwrite,
		'-i', 
		video_file_name, 
		'-q:v 1', 
		'-ss',
		str(s), 
		'-to',
		str(e), 
		out_f
	]

	conversion_process = subprocess.Popen(' '.join(ffmpeg_command), shell=True, stdout=subprocess.PIPE)
	conversion_process.communicate()

	return out_f

## Need to combine videos

def split_into_segments(vf, vid, output_directory, size=15):
	# Get length of video
	vid_length = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", vf],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
	vid_length = math.ceil(float(vid_length.stdout))

	# Make the output directory
	out = os.path.join(output_directory, vid)
	if not os.path.isdir(out):
		os.mkdir(out)

	print(f"{math.ceil(vid_length/size)} segments will be made")

	segment_paths = []
	segment_id = 1
	for i in progressbar.progressbar(range(0, vid_length, size)):
		segment_path = cut_video_trial(
			video_file_name = vf,
			s = i,
			e = i + size,
			vid = vid + '_' + str(segment_id).zfill(2),
			output_directory = out,
			overwrite = '-y'
		)

		segment_id += 1
		segment_paths.append(segment_path)

	return segment_paths


"""
Obscure ending

We need to add some videos to the end of the file. What we will do
is take the last frame and concatenate with a frame that shows that 
this video has ended. We will create a bunch of these so that all trials
have 30 segments
"""
def mark_and_pad_ending(segments, total_segments=30, tmp_dir='', font_file=''):
	# Get the average size of the segments in the file
	avg_size = np.mean([os.stat(i).st_size for i in segments[:-1]])
	size_last_segment = os.stat(segments[-1]).st_size

	padding_needed = int(np.floor(avg_size/size_last_segment))

	# Create the additional video with text
	overlay_segment = os.path.join(tmp_dir, 'with_overlay.mp4')
	ffmpeg_overlay_text = [
		'ffmpeg -y -i',
		segments[-1],
		f'-vf drawtext="fontfile={font_file}: text=\'TRIAL FINISHED\': fontcolor=white: fontsize=40: box=1: boxcolor=black@0.5: boxborderw=5: x=(w-text_w)/2: y=(h-text_h)/2"',
		'-crf 18',
		'-codec:a copy',
		overlay_segment
	]

	make_overlay = subprocess.run(' '.join(ffmpeg_overlay_text), shell=True, 
		stdout=subprocess.PIPE, 
		stderr=subprocess.PIPE)

	# Now create combined segments 
	# First one is the real ending padding with additional make endings
	# The next n are the ones with trial finished on all frames
	file_list = os.path.join(tmp_dir, 'file_list.txt')
	with open(file_list, "w") as f:
		f.write(f"file\t{segments[-1]}\n")
		for i in range(padding_needed):
			f.write(f"file\t{overlay_segment}\n")

	# Replace the last segment file with this new one
	new_last_segment = os.path.join(tmp_dir, re.sub('.*/', '', segments[-1]))
	concat_command = [
		'ffmpeg -y',
		'-f concat',
		'-safe 0 -i',
		file_list,
		'-c copy',
		new_last_segment
	]

	replace_last = subprocess.run(' '.join(concat_command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	# replace the old file with the new one
	shutil.copyfile(new_last_segment, segments[-1])

	# Now create fake additional ones
	file_list = os.path.join(tmp_dir, 'file_list2.txt')
	for i in range(len(segments)+1, total_segments):
		with open(file_list, "w") as f:
			# Add a random offset
			offset = random.randint(-2, 2)
			for j in range(padding_needed + offset):
				f.write(f"file\t{overlay_segment}\n")

		pad_fname = re.sub('[0-9]+\\.mp4', '', segments[-1]) + str(i).zfill(2) + '.mp4'

		concat_command = [
			'ffmpeg -y',
			'-f concat',
			'-safe 0 -i',
			file_list,
			'-c copy',
			pad_fname
		]

		padding = subprocess.run(' '.join(concat_command), shell=True, stdout=subprocess.PIPE)
		segments.append(pad_fname)

	return segments


"""
SCRIPT

Example use:

python tools/split_videos.py \
  --trial 'S610T1' \
  --tp_file_sheet '/Users/guillaumekugener/Downloads/trial_time_points - Sheet1 (1).csv' \
  --video_file '/Users/guillaumekugener/Downloads/ch1_video_02.mp4' \
  --output_full '/Users/guillaumekugener/Documents/USC/USC_docs/ml/datasets/performance-evaluation/full/' \
  --output_splits '/Users/guillaumekugener/Documents/USC/USC_docs/ml/datasets/performance-evaluation/splits/'

"""
tp_data = pd.read_csv(FLAGS.tp_file_sheet)
tp_data = tp_data.fillna('')

vids = get_video_time_points(
	trial_id=FLAGS.trial,
	tp_data=tp_data
)

# Right now, I have not set it up to handle 
# cases when the videos are split up, so error
# out if this is the case
if len(vids) != 1:
	raise Exception('need to merge files before continuing')

cut_segments = []
for v in vids:
    cut_out_path = cut_video_trial(
        video_file_name=FLAGS.video_file,
        s = v['start'], 
        e = v['end'],
        vid = v['vid'], 
        output_directory=FLAGS.output_full
    )

    cut_segments.append(cut_out_path)

# MERGE TWO FILES GOES HERE ##
merged_file = cut_segments[0]

segments = split_into_segments(
    vf = merged_file,
    vid = FLAGS.trial,
    output_directory=FLAGS.output_splits,
    size=FLAGS.chunk_size
)

if FLAGS.padding:
	mark_and_pad_ending(
	    segments=segments,
	    tmp_dir=FLAGS.tmp,
	    font_file='/Users/guillaumekugener/Documents/USC/USC_docs/ml/datasets/performance-evaluation/font.ttf'
	)


