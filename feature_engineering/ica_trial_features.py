import os
import numpy as np
import pandas as pd
import re

from math import sqrt

"""
ICATrialFeatures

Allows us to extract features from videos based on a csv
of objects within the dataset
"""
class ICATrialFeatures:
	def __init__(
		self,
		csv_file_name,
		csv_class_file
	):
		self.csv_file_name = csv_file_name
		
		na_values = {'file': 0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'class': ''}

		self.data = pd.read_csv(os.path.join(csv_file_name), names=['file', 'x1', 'y1', 'x2', 'y2', 'class'])
		self.data = self.data.fillna(value=na_values)
		self.data['vid'] = [re.search('S[0-9]+T[0-9]+[a-z]?', i).group(0) for i in self.data['file']]
		self.classes = pd.read_csv(os.path.join(csv_class_file), names=['class', 'ind'])

		self.videos = self.__build_videos()

	"""Get all of the trials in this data set"""
	def get_videos_in_set(self):
		return set(self.data['vid'])

	""""""
	def __build_videos(self):
		vids = self.get_videos_in_set()
		videos = []

		for vid in vids:
			videos.append(VideoTrial(video_data=self.data[self.data['vid']==vid]))

		return videos
"""
VideoTrial

Describes an individual video trial
"""
class VideoTrial:
	def __init__(
		self,
		video_data
	):
		self.video_data = video_data.reset_index()
		self.frames = self.__build_frames()

	"""Get length of video"""
	def get_video_length(self):
		return len(set([i for i in self.video_data['file']]))

	""""""
	def __build_frames(self):
		frames = []
		unique_frames = set([i for i in self.video_data['file']])

		for f in unique_frames:
			frames.append(IndividualFrame(self.video_data[self.video_data['file']==f]))

		# We need them to be sorted for future operations
		frames.sort(key=lambda f: f.frame_name)
		return frames

	"""Get number of frames video"""
	def numer_of_frames(self):
		return len(self.frames)

	"""Get number of frames with a tool"""
	def get_number_of_frames_with_tool(self, tool):
		with_tool = self.video_data[self.video_data['class']==tool]

		return len(set([i for i in with_tool['file']]))

	"""
	Get tool path

	This could get complicated when there are two tools of the same type
	(this happens in one video where there are two cottonoids). For now,
	we will only deal with their being a single instance of each
	"""
	def get_tool_path(self, tool):
		positions = []

		for f in self.frames:
			positions.append(f.get_tool_position(tool))

		return positions

	"""
	Get tool path length

	Take the distance between points in the tool path.
	
	assume_off_screen: If a tool goes off screen, then add the min distance
	from the edge to the path (although it could get
	removed due to redout so this is tricky...)
	"""
	def get_tool_path_length(self, tool, assume_off_screen=False):
		tool_path = self.get_tool_path(tool)

		total_distance = 0
		for pi in range(1, len(tool_path)):
			dest = tool_path[pi]
			source = tool_path[pi-1]

			distance = 0

			if dest[0] is not None and source[0] is not None:
				distance = sqrt((dest[0]-source[0])**2 + (dest[1]-source[1])**2)
			
			elif dest[0] is not None and assume_off_screen:
				distance = min(dest[0], dest[1])
			
			elif source[0] is not None and assume_off_screen:
				distance = min(source[0], [1])

			else:
				pass

			total_distance += distance

		return total_distance

	"""
	Get tool velocity

	Get the velocity of the tools.

	TODO: think of more sophisticated ways of doing this (with smoothing,
	polynomiral approximations, etc.)

	Right velocity is None unless both points are defined
	"""
	def get_tool_velocities(self, tool, window_size=2):
		tool_path = self.get_tool_path(tool)

		velocities = []
		for pi in range(1, len(tool_path)):
			vals = tool_path[max(0, pi-1):min(pi+window_size, len(tool_path)-1)]
			dest = vals[-1]
			source = vals[0]

			# Since this is velocity, we will keep track of directionality
			velocity = None
			if dest[0] is not None and source[0] is not None:
				velocity = sqrt((dest[0]-source[0])**2 + (dest[1]-source[1])**2)/len(vals)
			
			velocities.append(velocity)

		return velocities

"""
IndividualFrame

Describes an individual frame
"""
class IndividualFrame:
	def __init__(
		self,
		frame_data
	):
		self.frame_data = frame_data.reset_index()
		self.frame_name = self.frame_data['file'][0]
		self.tools = self.get_tools_in_frame()

	"""Get the tools in this frame"""
	def get_tools_in_frame(self):
		tools = []

		for i in range(self.frame_data.shape[0]):
			x1, y1, x2, y2 = self.frame_data.loc[i,['x1', 'y1', 'x2', 'y2']]
			tools.append(SurgicalTool(
				tool_name=self.frame_data['class'][i],
				bbox_coordinates=[x1, y1, x2, y2]
			))

		return tools

	"""Get position of tools"""
	def get_tool_position(self, tool):
		relevant_tool = [i for i in self.tools if i.get_type() == tool]

		position = (None, None)
		if len(relevant_tool) == 1:
			position = relevant_tool[0].get_tool_position()

		if len(relevant_tool) > 1:
			print(f"There were 2 or more instances of this tool in {self.frame_name}")

		return position

"""
SurgicalTool

Describes a surgical tool within a frame
"""
class SurgicalTool:
	def __init__(
		self,
		tool_name,
		bbox_coordinates
	):
		self.tool_name = tool_name
		self.bbox_coordinates = bbox_coordinates


	"""Get type"""
	def get_type(self):
		return self.tool_name

	"""Get position"""
	def get_tool_position(self):
		return self.get_bbox_center()

	"""Get the center of the bbox"""
	def get_bbox_center(self):
		xc = self.bbox_coordinates[2] - self.bbox_coordinates[0]
		yc = self.bbox_coordinates[3] - self.bbox_coordinates[1]

		return (xc, yc)







