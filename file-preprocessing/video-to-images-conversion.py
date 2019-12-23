'''
This script is used to split the surgical simulation videos into image segments.
'''

## IMPORT MODULES
import sys
import subprocess
import glob
import math
import shutil
import re

## SETUP
video_file_name = sys.argv[1]
output_directory = re.sub('/$', '', sys.argv[2]) + '/'
batch_size = int(sys.argv[3])

print('The video being converted to frames is: ' + video_file_name)
print('The output directory will be: ' + output_directory) + '/'
print('Batch size is: ' + str(batch_size))

# The line below can be used to determine the total number of frames in a video
# ffmpeg -i <file_name> -map 0:v:0 -c copy -f null -

## Run ffmpeg to generate the images
make_output_directory = subprocess.Popen(' '.join(['mkdir', output_directory]), shell=True, stdout=subprocess.PIPE)
conversion_process = subprocess.Popen(' '.join(['ffmpeg', '-i', video_file_name, '-q:v 1', output_directory + 'frame_%08d.jpeg']), shell=True, stdout=subprocess.PIPE)
conversion_process.communicate()

# Now zip all files in order to upload to drive (in batches of 1000?)
# In the conversion process command, we are prepending 0s until there are 8 digits in the frame
# number. So when we are running the zip command, we will leverage this fact to determine which 
# which files to combine together
all_frame_files = glob.glob(output_directory + 'frame_*.jpeg')
print('Total frames: ' + str(len(all_frame_files)))

# Zip frames into bins
intervals = int(math.floor(len(all_frame_files)/batch_size)) + 1
print(intervals)

# Move files around and generate the zip file
for i in xrange(intervals):
	start = batch_size*i
	end = min(batch_size*(i+1), len(all_frame_files)) - 1
	
	print(' '.join(['Binning and zipping frames:', str(start+1), 'to', str(end+1)]))
	print(all_frame_files[start])
	print(all_frame_files[end])

	zip_folder_name = output_directory + re.sub('.*/', '', re.sub('\\.mp4$', '', video_file_name, flags=re.IGNORECASE)) + '_frame_' + re.sub('(.*/frame_)|(\\.jpeg)', '', all_frame_files[start]) + '_' + re.sub('(.*/)|(\\.jpeg)', '', all_frame_files[end])
	mkzip_dir = subprocess.Popen(' '.join(['mkdir', zip_folder_name]), shell=True, stdout=subprocess.PIPE)
	
	# Wait for the make directory function to be complete before moving folders into it
	exit_codes = mkzip_dir.wait()

	for f in all_frame_files[start: end + 1]:
		shutil.move(f, zip_folder_name)

	shutil.make_archive(zip_folder_name + '_zip', 'zip', zip_folder_name)

	# Once the archive is made, we can delete the previous folder
	del_dir = subprocess.Popen(' '.join(['rm -rf', zip_folder_name]), shell=True, stdout=subprocess.PIPE)








