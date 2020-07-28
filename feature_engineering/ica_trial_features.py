import os
import numpy as np
import pandas as pd
import re
import progressbar

import matplotlib.pyplot as plt
from seaborn import scatterplot

from math import sqrt, floor, ceil
from scipy.stats import ttest_rel

"""
ICATrialFeatures

Allows us to extract features from videos based on a csv
of objects within the dataset
"""
class ICATrialFeatures:
	def __init__(
		self,
		csv_file_name,
		csv_class_file,
		verbose=False
	):
		self.csv_file_name = csv_file_name
		
		na_values = {'file': 0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'class': ''}

		self.data = pd.read_csv(os.path.join(csv_file_name), names=['file', 'x1', 'y1', 'x2', 'y2', 'class'])
		self.data = self.data.fillna(value=na_values)
		self.data['vid'] = [re.search('S[0-9]+T[0-9]+[a-z]?', i).group(0) for i in self.data['file']]
		self.classes = pd.read_csv(os.path.join(csv_class_file), names=['class', 'ind'])
		self.verbose = verbose

		self.videos = self.__build_videos()
		self.participants = [re.sub('S', '', i) for i in set([re.search('S[0-9]+', j).group(0) for j in self.get_videos_in_set()])]

	"""Get all of the trials in this data set"""
	def get_videos_in_set(self):
		return set(self.data['vid'])

	"""Get trials in set"""
	def get_participants_in_set(self):
		return self.participants[:]

	""""""
	def __build_videos(self):
		vids = self.get_videos_in_set()
		videos = []

		for vid in vids:
			videos.append(VideoTrial(
				video_data=self.data[self.data['vid']==vid], 
				video_id=vid,
				verbose=self.verbose
			))

		return videos

	"""Get video by ID"""
	def get_video_by_id(self, vid):
		video = None
		for i in self.videos:
			if i.get_video_id() == vid:
				video = i
				break

		return video

	"""
	Get dataset metrics

	Each trial in the dataset gets a row with various metrics 
	calculated for that trial
	"""
	def get_metrics(self, tools=[]):
		output_dict = { 'vid': [] }

		# iterate through each video and add its metrics
		for video in progressbar.progressbar(self.videos):
			output_dict['vid'].append(video.get_video_id())

			video_metrics = video.summarized_features(tools=tools)

			for k in video_metrics:
				if k not in output_dict:
					output_dict[k] = []
				output_dict[k].append(video_metrics[k])

		output_dict['trial'] = [re.sub('S[0-9]+T', '', i) for i in output_dict['vid']]
		return pd.DataFrame(output_dict)

	"""
	Compare metrics between two groups

	Given two lists of video ids, will run a t-test comparing
	the metrics between those two groups of videos

	TODO: make sure that paired implementation is correctly implemented. 
	Want to validate that if our set does not have both pairs that it should 
	exlucde both etc...
	"""
	def compare_metrics_t1_t2(self, tools=[], trials=[]):
		# first, get the metrics
		set_metrics = self.get_metrics(tools=tools).set_index('vid')

		metrics = [i for i in set_metrics.columns.values if i not in ['vid', 'frames', 'trial']]

		t_test_results = {
			'metric': [],
			'mean_g1': [],
			'mean_g2': [],
			't_stat': [],
			'p': []
		}
		missed = set()
		for m in metrics:
			g1_values = []
			g2_values = []

			for t in trials:
				# check that both trials are in the dataset
				if ('S' + t + 'T1' not in set_metrics.index) or ('S' + t + 'T2' not in set_metrics.index):
					missed.add(t)
					continue

				g1_values.append(set_metrics.loc['S' + t + 'T1', m])
				g2_values.append(set_metrics.loc['S' + t + 'T2', m])

			# now compute the t test for the metrics
			t_test_out = ttest_rel(g1_values, g2_values)
			mean_g1 = np.mean(g1_values)
			mean_g2 = np.mean(g2_values)

			t_test_results['metric'].append(m)
			t_test_results['mean_g1'].append(mean_g1)
			t_test_results['mean_g2'].append(mean_g2)
			t_test_results['t_stat'].append(t_test_out[0])
			t_test_results['p'].append(t_test_out[1])

		# print the samples that were missing
		for t in missed:
			print(f"{t} is missing either T1, T2 or both or is in chunks")

		return pd.DataFrame(t_test_results)


"""
VideoTrial

Describes an individual video trial
"""
class VideoTrial:
	def __init__(
		self,
		video_data,
		video_id, 
		frame_size=None,
		verbose=False
	):
		self.video_data = video_data.reset_index()
		self.video_id = video_id
		self.verbose = verbose
		self.frame_size = self.__set_frame_size(frame_size)
		self.frames = self.__build_frames()

	"""
	Set video size

	If no video size is passed in, we are going to guess 
	what the size should be based on the bounding box coordinates
	"""
	def __set_frame_size(self, frame_size):
		if frame_size is None:
			"""Make a guess"""
			max_x = int(max(self.video_data['x2']))
			max_y = int(max(self.video_data['y2']))

			if max_y == 720:
				frame_size = (1280, 720)
			elif max_y == 1280:
				frame_size = (1920, 1080)
			else:
				pass

		return frame_size

	"""Get trial id"""
	def get_video_id(self):
		return self.video_id

	"""Get length of video"""
	def get_video_length(self):
		return len(set([i for i in self.video_data['file']]))

	""""""
	def __build_frames(self):
		frames = []
		unique_frames = set([i for i in self.video_data['file']])

		for f in unique_frames:
			frames.append(IndividualFrame(
				frame_size=self.frame_size,
				frame_data=self.video_data[self.video_data['file']==f],
				verbose=self.verbose
			))

		# We need them to be sorted for future operations
		frames.sort(key=lambda f: f.frame_name)
		return frames

	"""Get number of frames video"""
	def number_of_frames(self):
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
	def get_tool_path(self, tool, n=None):
		positions = []

		for f in self.frames:
			positions.append(f.get_tool_position(tool))

		if n is not None:
			positions = positions[:n]

		return positions

	"""
	Get tool path length

	Take the distance between points in the tool path.
	
	assume_off_screen: If a tool goes off screen, then add the min distance
	from the edge to the path (although it could get
	removed due to redout so this is tricky...)
	"""
	def get_tool_path_length(self, tool, n=None, assume_off_screen=False):
		tool_path = self.get_tool_path(tool, n=n)

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

		velocities = [None]
		for pi in range(1, len(tool_path)):
			vals = tool_path[max(0, pi-1):min(pi+window_size, len(tool_path)-1)]
			vals = [i for i in vals if i is not None]

			dest = vals[-1]
			source = vals[0]

			# Since this is velocity, we will keep track of directionality
			velocity = None
			if (len(vals) > 1) and (dest[0] is not None) and (source[0] is not None):
				velocity = sqrt((dest[0]-source[0])**2 + (dest[1]-source[1])**2)/len(vals)
			
			velocities.append(velocity)

		return velocities

	"""
	Get area covered by tool
	"""
	def get_area_covered_by_tool(self, tool, n=None):
		area_covered = []

		frames = self.frames[:]
		if n is not None:
			frames = frames[:n]

		for f in frames:
			relevant_tools = f.get_specific_tool(tool)

			tot_area = 0
			for rt in relevant_tools:	
				tot_area += rt.get_area()

			area_covered.append(tot_area)

		return area_covered

	"""
	Get tool overlap
	"""
	def get_tool_overlap(self, t1, t2, n=None):
		overlap = []
		frames = self.frames[:]

		if n is not None:
			frames = frames[:n]

		for f in frames:
			overlap.append(f.get_tool_overlap(t1, t2))

		return overlap

	"""
	Get distances between tools
	"""
	def get_distance_between_tools(self, t1, t2, n=None):
		distance = []
		frames = self.frames[:]

		if n is not None:
			frames = frames[:n]

		for f in frames:
			distance.append(f.get_distance_between_tools(t1, t2))

		return distance

	"""
	Video full feature matrix

	Returns a frame x feature matrix for this video
	"""
	def full_feature_matrix(self, tools=[]):
		features_dict = {}

		features_dict['n_tools_in_view'] = [len([t for t in f.get_tools() if t.get_type() in tools]) for f in self.frames]

		# Single tool features
		for tool in tools:
			features_dict['position_' + tool] = self.get_tool_path(tool)
			features_dict['area_covered_' + tool] = self.get_area_covered_by_tool(tool)
			features_dict['velocity_' + tool] = self.get_tool_velocities(tool)

		# Combination tool features
		for t1 in tools:
			for t2 in tools:
				if t1 == t2:
					continue

				# Make the column name sorted
				col_name = [t1, t2]
				col_name.sort()
				col_name = '_'.join(col_name)

				if col_name not in features_dict:
					features_dict['overlap_' + col_name] = self.get_tool_overlap(t1, t2)
					features_dict['distance_between_' + col_name] = self.get_distance_between_tools(t1, t2)

		return pd.DataFrame(features_dict)

	"""
	Summarized feature matrix

	For this video, returns a single row of features. Features are:

	- # frames with 0, 1, 2, ... tools in view
	- are covered by each tool
	- distance covered by each tool
	- proportion frames where each tool is present
	"""
	def summarized_features(self, tools=[]):
		summarized_features_dict = {}
		features_df = self.full_feature_matrix(tools=tools)

		# number of tools in view
		summarized_features_dict['frames'] = self.number_of_frames()
		n_tool_ranges = [0, 1, 2, 3, 4]
		for j in n_tool_ranges:
			summarized_features_dict['proportion_' + str(j) + '_tools_in_view'] = len([i for i in features_df['n_tools_in_view'] if i == j])/self.number_of_frames()
		summarized_features_dict['proportion_' + str(n_tool_ranges[-1]+1) + '+_tools_in_view'] = len([i for i in features_df['n_tools_in_view'] if i > j])/self.number_of_frames()

		# tool specific trial level features
		for tool in tools:
			summarized_features_dict['area_covered_total_' + tool] = np.sum(features_df['area_covered_' + tool])
			summarized_features_dict['path_distances_' + tool] = self.get_tool_path_length(tool)
			summarized_features_dict['proportion_' + tool] = self.get_number_of_frames_with_tool(tool)/self.number_of_frames()

		# add the combination columns (distance from each other)
		for t1 in tools:
			for t2 in tools:
				if t1 == t2:
					continue

				# Make the column name sorted
				col_name = [t1, t2]
				col_name.sort()
				col_name = '_'.join(col_name)

				if col_name not in summarized_features_dict:
					for k_pre in ['overlap_', 'distance_between_']:
						summarized_features_dict[k_pre + col_name] = np.sum(features_df[k_pre + col_name])

		return summarized_features_dict

	"""
	Plot tools over all frames

	Plots the center of each tool across all frames (as a scatter plot)
	"""
	def plot_tool_positions(
		self, 
		tools=[],
		n_frames=None
	):
		# We have to get the plot data
		positions_dict = { 'x': [], 'y': [], 'class': []}
		for tool in tools:
			positions = self.get_tool_path(tool)

			# If we want to only plot data for the first n frames
			if n_frames is not None:
				positions = positions[:n_frames]

			positions_dict['x'] += [i[0] for i in positions]
			positions_dict['y'] += [i[1] for i in positions]
			positions_dict['class'] += [tool for i in positions]

		positions_df = pd.DataFrame(positions_dict)

		ax = scatterplot(data=positions_df, x='x', y='y', hue='class')
		ax.invert_yaxis()

	"""
	Plot paths across frames

	Creates a plot connecting the dots across frames for the different tools
	"""
	def plot_tool_paths(
		self, 
		tools=[],
		tool_colors={
			'suction': 'blue',
			'grasper': 'orange',
			'cottonoid': 'green',
			'muscle': 'red',
			'string': 'purple'
		}
	):
		positions_dict = { 
			'x': [], 'y': [],
			'xend': [], 'yend': [],
			'class': [] 
		}

		for tool in tools:
			positions = self.get_tool_path(tool)
			positions_dict['x'] += [i[0] for i in positions[:len(positions)-1]]
			positions_dict['y'] += [i[1] for i in positions[:len(positions)-1]]

			positions_dict['xend'] += [i[0] for i in positions[1:]]
			positions_dict['yend'] += [i[1] for i in positions[1:]]

			positions_dict['class'] += [tool for i in positions[1:]]

		positions_df = pd.DataFrame(positions_dict)

		for r in range(positions_df.shape[0]):
			x, y, xend, yend, cl = positions_df.loc[r, ['x', 'y', 'xend', 'yend', 'class']]

			if x is None:
				continue

			plt.plot([x, xend], [y, yend], c=tool_colors[cl], ls='-')

		plt.show()

	"""
	Plot tool positions as heatmap
	"""
	def plot_tool_heatmap(self, tool):
		tool_heatmap = np.zeros((self.frame_size[1], self.frame_size[0]))
		for f in self.frames:
			tool_heatmap += f.get_tool_bbox_binary(tool)

		plt.imshow(tool_heatmap, cmap='hot', interpolation='nearest')
		plt.show()

	"""Get n tools in view across frames"""
	def get_n_tools_in_view(self, ignore=[], n=None):
		n_tools_in_view = []
		frames = self.frames[:]

		if n is not None:
			frames = frames[:n]

		for f in frames:
			n_tools_in_view.append(len(f.get_tools(ignore=ignore)))

		return n_tools_in_view

"""
IndividualFrame

Describes an individual frame
"""
class IndividualFrame:
	def __init__(
		self,
		frame_size,
		frame_data,
		verbose=False
	):
		self.frame_data = frame_data.reset_index()
		self.frame_name = self.frame_data['file'][0]
		self.tools = self.get_tools()
		self.frame_size = frame_size
		self.verbose = verbose

	"""Get the tools in this frame"""
	def get_tools(self, ignore=[]):
		tools = []

		if len(ignore) > 0:
			ignore = set(ignore)

		for i in range(self.frame_data.shape[0]):
			if self.frame_data['class'][i] in ignore:
				continue

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

		if len(relevant_tool) > 1 and self.verbose:
			print(f"There were 2 or more instances of this tool in {self.frame_name}")

		return position

	"""Get particular tool"""
	def get_specific_tool(self, tool):
		relevant_tools = []
		for to in self.tools:
			if to.get_type() == tool:
				relevant_tools.append(to)

		return relevant_tools

	"""
	Tool bbox binary
	
	Returns a numpy array the size of the frame
	with 1s where the bounding box is and 0s everywhere
	else
	"""
	def get_tool_bbox_binary(self, tool):
		frame_array = np.zeros((self.frame_size[1], self.frame_size[0]))

		relevant_tools = self.get_specific_tool(tool)

		for to in relevant_tools:
			x1, y1, x2, y2 = [int(i) for i in to.get_standard_coordinates()]


			tool_array = np.ones((y2-y1, x2-x1))
			frame_array[y1:y2,x1:x2] += tool_array

		return frame_array

	"""
	Get tool overlap

	If there are multiple tools, it takes all combinations 
	(if 2 suctions and 3 muscles, then 6 total)
	"""
	def get_tool_overlap(self, t1, t2):
		rt1 = self.get_specific_tool(tool=t1)
		rt2 = self.get_specific_tool(tool=t2)

		area_overlap = 0
		for tool1 in rt1:
			for tool2 in rt2:
				overlap = get_overlap_between_tools(tool1, tool2)
				if overlap is not None:
					area_overlap += overlap

		return area_overlap

	"""
	Get distance between tools

	If there are multiple tools, we take the mean
	"""
	def get_distance_between_tools(self, t1, t2):
		rt1 = self.get_specific_tool(tool=t1)
		rt2 = self.get_specific_tool(tool=t2)

		distance_between = []
		for tool1 in rt1:
			for tool2 in rt2:
				distance_between.append(get_distance_between_tools(tool1, tool2))

		return np.mean(distance_between)

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

	"""Get original x1, y1, x2, y2"""
	def get_standard_coordinates(self):
		return self.bbox_coordinates

	"""Get the center of the bbox"""
	def get_bbox_center(self):
		xc = self.bbox_coordinates[2] - self.bbox_coordinates[0]
		yc = self.bbox_coordinates[3] - self.bbox_coordinates[1]

		return (xc, yc)

	"""Get tool corners"""
	def get_tool_corners(self):
		tl = self.bbox_coordinates[0], self.bbox_coordinates[1]
		bl = self.bbox_coordinates[0], self.bbox_coordinates[3]
		tr = self.bbox_coordinates[2], self.bbox_coordinates[1]
		br = self.bbox_coordinates[2], self.bbox_coordinates[3]

		return tl, bl, tr, br

	"""Get bbox dimensions"""
	def get_dimensions(self):
		h = self.bbox_coordinates[3] - self.bbox_coordinates[1]
		w = self.bbox_coordinates[2] - self.bbox_coordinates[0]
		return w, h

	"""Get tool area"""
	def get_area(self):
		w, h = self.get_dimensions()
		return w * h


"""
Calculate overlapping area of two boxes

Assumes that the coordinates for the boxes
are x1, y1, x2, y2

Returns None if there is no overlap
"""
def overlap_area(a, b):
	dx = min(a[2], b[2]) - max(a[0], b[0])
	dy = min(a[3], b[3]) - max(a[1], b[1])

	if (dx >= 0) and (dy >= 0):
		return dx * dy

"""
Get overlapping area between two tools
"""
def get_overlap_between_tools(t1, t2):
	t1d = t1.get_standard_coordinates()
	t2d = t2.get_standard_coordinates()
	
	return overlap_area(t1d, t2d)

"""
Get the distance between two tools
"""
def get_distance_between_tools(t1, t2):
	t1d = t1.get_tool_position()
	t2d = t2.get_tool_position()

	return sqrt((t2d[1] - t1d[1])**2 + (t2d[0] - t1d[0])**2)