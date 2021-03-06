import os
import numpy as np
import pandas as pd
import re
import progressbar

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from seaborn import scatterplot

from math import sqrt, floor, ceil
from scipy.stats import (
	ttest_rel, ttest_ind
)

FRAME_MAX=100000
TOOL_COLORS={
	'suction': 'blue',
	'grasper': 'orange',
	'cottonoid': 'green',
	'muscle': 'red',
	'string': 'purple'
}

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
		csv_frame_sizes,
		csv_outcomes_file=None,
		verbose=False,
		trials_to_ignore=[]
	):
		self.csv_file_name = csv_file_name
		
		na_values = {'file': 0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'class': ''}

		self.data = pd.read_csv(csv_file_name, names=['file', 'x1', 'y1', 'x2', 'y2', 'class'], engine= 'python')
		self.data = self.data.fillna(value=na_values)
		self.data['original_vid'] = [re.search('S[0-9]+T[0-9]+[a-z]?', i).group(0) for i in self.data['file']]

		# some videos were separated. combine them all here
		self.__merge_fragmented_videos()

		# remove trials to ignore
		self.data = self.data[~self.data['vid'].isin(trials_to_ignore)]

		self.classes = pd.read_csv(csv_class_file, names=['class', 'ind'])
		frame_sizes_df = pd.read_csv(csv_frame_sizes)
		self.frame_sizes = {}
		for fi in range(frame_sizes_df.shape[0]):
			self.frame_sizes[frame_sizes_df.loc[fi,'trial_id']] = (
				int(frame_sizes_df.loc[fi, 'w']),
				int(frame_sizes_df.loc[fi, 'h'])
			)

		self.outcomes_data = pd.read_csv(csv_outcomes_file)
		self.verbose = verbose

		self.videos = self.__build_videos()
		self.participants = [re.sub('S', '', i) for i in set([re.search('S[0-9]+', j).group(0) for j in self.get_videos_in_set()])]

	"""Get all of the trials in this data set"""
	def get_videos_in_set(self):
		return set(self.data['vid'])

	"""Get trials in set"""
	def get_participants_in_set(self):
		return self.participants[:]

	"""
	Merge fragmented videos

	Two things to do:
	- new column named vid with proper name (easy)
	- have to fix frame indexing (create f_index column)

	Updates self.data directly
	"""
	def __merge_fragmented_videos(self):
		self.data['vid'] = [re.sub('[a-z]+$', '', i) for i in self.data['original_vid']]
		# the rank method (i.e. first, last tie) does not matter, 
		# since the order in which the tools in a frame does not mater
		self.data['f_index'] = self.data.groupby("vid")["file"].apply(lambda x: x.rank())

	""""""
	def __build_videos(self):
		vids = self.get_videos_in_set()
		videos = []

		for vid in progressbar.progressbar(vids):
			videos.append(VideoTrial(
				video_data=self.data[self.data['vid']==vid], 
				video_id=vid,
				verbose=self.verbose,
				frame_size=self.frame_sizes[vid],
				outcomes_data=self.outcomes_data[self.outcomes_data['SurveyID']==get_participant_id_from_string(vid)]
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

	n_frames defaults to huge number that should never be less
	than the total number of frames
	"""
	def get_metrics(self, tools=[], n_frames=FRAME_MAX):
		output_dict = { 'vid': [] }

		# iterate through each video and add its metrics
		for video in progressbar.progressbar(self.videos):
			output_dict['vid'].append(video.get_video_id())

			video_metrics = video.summarized_features(tools=tools, n_frames=n_frames)

			for k in video_metrics:
				if k not in output_dict:
					output_dict[k] = []
				output_dict[k].append(video_metrics[k])

		output_dict['trial'] = [re.sub('S[0-9]+T', '', i) for i in output_dict['vid']]
		return pd.DataFrame(output_dict)

	"""Get outcomes data"""
	def get_outcomes_data(self):
		output_dict = { 'vid': [] }

		for video in progressbar.progressbar(self.videos):
			output_dict['vid'].append(video.get_video_id())

			video_metrics = video.get_outcomes_data()

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
	Compare metrics successful trial vs. unsuccessful

	Similar to comparing metrics of T1 vs. T2 but this time
	grouped based on trial success
	"""
	def compare_successful_vs_failed(self, tools=[]):
		set_metrics = self.get_metrics(tools=tools).set_index('vid')
		set_outcomes = self.get_outcomes_data().set_index('vid')

		metrics = [i for i in set_metrics.columns.values if i not in ['vid', 'frames', 'trial']]

		t_test_results = {
			'metric': [],
			'mean_f': [],
			'mean_s': [],
			't_stat': [],
			'p': []
		}
		missed = set()
		trials = set_metrics.index[:]
		for m in metrics:
			f_values = []
			s_values = []

			for t in trials:
				if set_outcomes.loc[t, 'Trial Success'] == 1:
					s_values.append(set_metrics.loc[t, m])
				else:
					f_values.append(set_metrics.loc[t, m])

			# now compute the t test for the metrics
			t_test_out = ttest_ind(f_values, s_values)
			mean_f = np.mean(f_values)
			mean_s = np.mean(s_values)

			t_test_results['metric'].append(m)
			t_test_results['mean_f'].append(mean_f)
			t_test_results['mean_s'].append(mean_s)
			t_test_results['t_stat'].append(t_test_out[0])
			t_test_results['p'].append(t_test_out[1])

		# print the samples that were missing
		for t in missed:
			print(f"{t} is missing either T1, T2 or both or is in chunks")

		return pd.DataFrame(t_test_results)

	"""
	Plot TTH vs. n frames annotated

	To compare what the recorded TTH is and how many frames were annotated for that video.
	We want to make sure that there are not huge differences (and if there are, are they 
	explained)
	"""
	def plot_tth_v_n_frames(self, save_path):
		scatter_outcomes_tth_v_n_frames = { 'trial_id': [], 'n_frames': [], 'tth': [] }
		for vo in self.videos:
		    scatter_outcomes_tth_v_n_frames['trial_id'].append(vo.get_video_id())
		    scatter_outcomes_tth_v_n_frames['tth'].append(vo.outcomes_data['Trial TTH'])
		    scatter_outcomes_tth_v_n_frames['n_frames'].append(vo.get_video_length())

		outcomes_compare_df = pd.DataFrame(scatter_outcomes_tth_v_n_frames)
		# ax = scatterplot(data=outcomes_compare_df, x='tth', y='n_frames')
		
		grid = sns.JointGrid(outcomes_compare_df.tth, outcomes_compare_df.n_frames, space=0, size=6, ratio=50)
		grid.plot_joint(plt.scatter)
		plt.plot([0,300],[0,250], linewidth=2)

		grid.savefig(save_path)

"""
VideoTrial

Describes an individual video trial
"""
class VideoTrial:
	def __init__(
		self,
		video_data,
		video_id, 
		outcomes_data,
		frame_size=None,
		verbose=False
	):
		self.video_data = video_data.reset_index()
		self.video_id = video_id
		self.verbose = verbose
		self.frame_size = self.__set_frame_size(frame_size)
		self.frames = self.__build_frames()
		self.outcomes_data = self.__format_outcomes_data(outcomes_data.reset_index())
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
			elif max_y == 1080:
				frame_size = (1920, 1080)
			elif max_x == 1487 and max_y == 1048:
				frame_size = (1920, 1080)
			elif max_x == 1631 and max_y == 1041:
				frame_size = (1920, 1080)
			elif max_x == 1446 and max_y == 1049:
				frame_size = (1920, 1080)
			elif max_x == 1495 and max_y == 1051:
				frame_size = (1920, 1080)
			else:
				print(f"Could not find frame size for {self.video_id}. {max_x}, {max_y}")
				print(self.video_data.head())
				pass

		return frame_size

	"""Get trial id"""
	def get_video_id(self):
		return self.video_id

	"""Succeeded trial (returns boolean)"""
	def succeeded_trial(self):
		return self.outcomes_data['Trial Success'] == 1

	"""
	Succeeds trial in next n frames

	Given a number of frames, determine if the participant
	achieves hemostasis in the next n frames
	"""
	def succeeds_trial_in_next_n(self, start_frame, n_frames):
		return start_frame + n_frames >= len(self.get_video_length())

	"""Get trial number"""
	def get_trial_number(self):
		s = re.search('T[0-9]+', self.get_video_id()).group(0)
		return s[1:]

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
		frames.sort(key=lambda f: f.frame_index)
		return frames

	"""
	Format outcomes data

	Returns a dict for all of the important stats based on the outcomes
	data file
	"""
	def __format_outcomes_data(self, outcomes_data):
		if outcomes_data.shape[0] != 1:
			return {}

		trial_num = self.get_trial_number()

		outcomes_metrics = [
			'Group', 'Source', 'Specialty',
			'Totyears', 'Attyears', 'Resyears',
			'endolast12mo', 'cadaverlast12',
			'priorreal', 'priorsim', 'simonly', 'realandsim',
			'generalconfidencepre', 'carotidconfidencepre',
			'generalconfidencepost', 'carotidconfidencepost',
			'Trial _num_ TTH', 'Trial _num_ Success', 'Trial _num_ EBL', 

		]
		
		outcomes_dict = {}
		for c in outcomes_metrics:
			# looks complex but isn't: we extract the TTH, EBL, and Success
			# value based on the trial number in the video id
			# then we assign it to Trial TTH, EBL, Success (without trial
			# number in the name) in the output dict. this will make
			# downstream processing easier
			ci = re.sub('_num_', trial_num, c)
			outcomes_dict[re.sub('_num_ ', '', c)] = outcomes_data[ci][0]

		return outcomes_dict

	"""Get outcomes data"""
	def get_outcomes_data(self):
		return self.outcomes_data.copy()

	"""Get number of frames video"""
	def number_of_frames(self):
		return len(self.frames)

	"""Get number of frames with a tool"""
	def get_number_of_frames_with_tool(self, tool, n_frames=FRAME_MAX):

		frames_with_tool = 0
		for f in self.frames[:n_frames]:
			if f.has_tool(tool):
				frames_with_tool += 1

		return frames_with_tool

	"""
	Get tool path

	This could get complicated when there are two tools of the same type
	(this happens in one video where there are two cottonoids). For now,
	we will only deal with their being a single instance of each
	"""
	def get_tool_path(self, tool, n_frames=FRAME_MAX):
		positions = []

		for f in self.frames:
			positions.append(f.get_tool_position(tool))

		return positions[:n_frames]
	
	"""
	Get tool path coords

	Gets a list of tool coordinates [x1,y1,x2,y2] for each frame and returns
	them as a list
	"""
	def get_tool_path_coords(self, tool, n_frames=FRAME_MAX):
		positions = []

		for f in self.frames:
			positions.append(f.get_standard_coordinates(tool))

		return positions[:n_frames]

	"""
	Get tool path length

	Take the distance between points in the tool path.
	
	assume_off_screen: If a tool goes off screen, then add the min distance
	from the edge to the path (although it could get
	removed due to redout so this is tricky...)
	"""
	def get_tool_path_length(self, tool, n_frames=FRAME_MAX, assume_off_screen=False):
		tool_path = self.get_tool_path(tool, n_frames=n_frames)

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
	def get_tool_velocities_standard(self, tool, window_size=2, n_frames=FRAME_MAX):
		tool_path = self.get_tool_path(tool, n_frames=n_frames)

		velocities = [None]
		for pi in range(1, len(tool_path)):
			vals = tool_path[max(0, pi-1):min(pi+window_size, len(tool_path)-1)]
			vals = [i for i in vals if i is not None]

			dest = vals[-1]
			source = vals[0]

			# Since this is velocity, we will keep track of directionality
			velocity = 0
			if (len(vals) > 1) and (dest[0] is not None) and (source[0] is not None):
				velocity = sqrt((dest[0]-source[0])**2 + (dest[1]-source[1])**2)/len(vals)
			
			velocities.append(velocity)

		return velocities[:n_frames]

	"""
	Get the tool velocity as an intersection over union
	"""

	def get_tool_velocities_iou(self, tool, window_size=2, n_frames=FRAME_MAX):
		tool_path = self.get_tool_path_coords(tool, n_frames=n_frames)
		tool_areas = self.get_area_covered_by_tool(tool, n_frames=n_frames)
		assert len(tool_path) == len(tool_areas), "tool path and tool areas are not equal sized"

		velocities = [None]
		for pi in range(1, len(tool_path)):
			vals = tool_path[max(0, pi-1):min(pi+window_size, len(tool_path)-1)]
			vals = [i for i in vals if i is not None]

			areas = tool_areas[max(0, pi-1):min(pi+window_size, len(tool_path)-1)]
			areas = [i for i in areas if i is not None]

			dest = vals[-1]
			source = vals[0]
			dest_area = areas[-1]
			source_area = areas[0]

			# Since this is velocity, we will keep track of directionality
			iou_velocity = 0
			if (len(vals) > 1) and (dest[0] is not None) and (source[0] is not None):
				intersection = overlap_area(dest, source)
				if (intersection != None):
					union = dest_area + source_area - intersection
					iou = intersection/union
					iou_velocity = 1/iou

			
			velocities.append(iou_velocity)

		return velocities[:n_frames]

	"""
	Get area covered by tool
	"""
	def get_area_covered_by_tool(self, tool, n_frames=FRAME_MAX):
		area_covered = []

		frames = self.frames[:n_frames]

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
	def get_tool_overlap(self, t1, t2, n_frames=FRAME_MAX):
		overlap = []
		frames = self.frames[:n_frames]

		for f in frames:
			overlap.append(f.get_tool_overlap(t1, t2))

		return overlap

	"""
	Get distances between tools
	"""
	def get_distance_between_tools(self, t1, t2, n_frames=FRAME_MAX):
		distance = []
		frames = self.frames[:n_frames]

		for f in frames:
			distance.append(f.get_distance_between_tools(t1, t2))

		return distance

	"""
	Video full feature matrix

	Returns a frame x feature matrix for this video
	"""
	def full_feature_matrix(self, tools=[], n_frames=FRAME_MAX):
		features_dict = {}

		features_dict['n_tools_in_view'] = [len([t for t in f.get_tools() if t.get_type() in tools]) for f in self.frames[:n_frames]]

		# Single tool features
		for tool in tools:
			features_dict['position_' + tool] = self.get_tool_path(tool, n_frames=n_frames)
			features_dict['area_covered_' + tool] = self.get_area_covered_by_tool(tool, n_frames=n_frames)
			features_dict['velocity_' + tool] = self.get_tool_velocities_standard(tool, n_frames=n_frames)

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
					features_dict['overlap_' + col_name] = self.get_tool_overlap(t1, t2, n_frames=n_frames)
					features_dict['distance_between_' + col_name] = self.get_distance_between_tools(t1, t2, n_frames=n_frames)

		return pd.DataFrame(features_dict)

	"""
	Summarized feature matrix

	For this video, returns a single row of features. Features are:

	- # frames with 0, 1, 2, ... tools in view
	- are covered by each tool
	- distance covered by each tool
	- proportion frames where each tool is present
	"""
	def summarized_features(self, tools=[], n_frames=FRAME_MAX):
		if n_frames == FRAME_MAX:
			n_frames = self.number_of_frames()

		summarized_features_dict = {}
		features_df = self.full_feature_matrix(tools=tools, n_frames=n_frames)

		# number of tools in view
		summarized_features_dict['frames'] = n_frames
		n_tool_ranges = [0, 1, 2, 3, 4]
		for j in n_tool_ranges:
			summarized_features_dict['proportion_' + str(j) + '_tools_in_view'] = len([i for i in features_df['n_tools_in_view'] if i == j])/n_frames
		summarized_features_dict['proportion_' + str(n_tool_ranges[-1]+1) + '+_tools_in_view'] = len([i for i in features_df['n_tools_in_view'] if i > j])/n_frames

		# tool specific trial level features
		for tool in tools:
			summarized_features_dict['area_covered_total_' + tool] = np.sum(features_df['area_covered_' + tool])/n_frames
			summarized_features_dict['path_distances_' + tool] = self.get_tool_path_length(tool, n_frames=n_frames)/n_frames
			summarized_features_dict['proportion_' + tool] = self.get_number_of_frames_with_tool(tool, n_frames=n_frames)/n_frames
		
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
						summarized_features_dict[k_pre + col_name] = np.sum(features_df[k_pre + col_name])/n_frames

		return summarized_features_dict

	"""
	Plot tools over all frames

	Plots the center of each tool across all frames (as a scatter plot)
	"""
	def plot_tool_positions(
		self, 
		tools=[],
		n_frames=FRAME_MAX,
		s_frame=0
	):
		# We have to get the plot data
		positions_dict = { 'x': [], 'y': [], 'class': []}
		for tool in tools:
			positions = self.get_tool_path(tool, n_frames=s_frame+n_frames)
			positions = positions[s_frame:]

			positions_dict['x'] += [i[0] for i in positions]
			positions_dict['y'] += [i[1] for i in positions]
			positions_dict['class'] += [tool for i in positions]

		positions_df = pd.DataFrame(positions_dict)

		ax = scatterplot(data=positions_df, x='x', y='y', hue='class')
		ax.invert_yaxis()

		return ax.get_figure()

	"""
	Plot paths across frames

	Creates a plot connecting the dots across frames for the different tools
	"""
	def plot_tool_paths(
		self, 
		tools=[],
		tool_colors=TOOL_COLORS,
		n_frames=FRAME_MAX,
		s_frame=0
	):
		positions_dict = { 
			'x': [], 'y': [],
			'xend': [], 'yend': [],
			'class': [] 
		}

		for tool in tools:
			positions = self.get_tool_path(tool, n_frames=s_frame+n_frames)
			positions = positions[s_frame:]

			positions_dict['x'] += [i[0] for i in positions[:len(positions)-1]]
			positions_dict['y'] += [i[1] for i in positions[:len(positions)-1]]

			positions_dict['xend'] += [i[0] for i in positions[1:]]
			positions_dict['yend'] += [i[1] for i in positions[1:]]

			positions_dict['class'] += [tool for i in positions[1:]]

		positions_df = pd.DataFrame(positions_dict)

		fig = plt.figure()
		for r in range(positions_df.shape[0]):
			x, y, xend, yend, cl = positions_df.loc[r, ['x', 'y', 'xend', 'yend', 'class']]

			if x is None:
				continue

			plt.plot([x, xend], [y, yend], c=tool_colors[cl], ls='-')

		return fig

	"""
	Plot tool positions as heatmap
	"""
	def plot_tool_heatmap(self, tools, n_frames=FRAME_MAX, s_frame=0):
		fig = plt.figure()

		subplot_nums = [221, 222, 223, 224]
		for ti, t in enumerate(tools):
			tool_heatmap = np.zeros((self.frame_size[1], self.frame_size[0]))
			for f in self.frames[s_frame:s_frame+n_frames]:
				tool_heatmap += f.get_tool_bbox_binary(t)
			ax1 = fig.add_subplot(subplot_nums[ti])
			ax1.title.set_text(t)
			plt.imshow(tool_heatmap, cmap='hot', interpolation='nearest')

		return fig

	"""
	Plot tool velocity over all frames

	Plots the velocity measured at the center of each tool across all 
 	frames
	"""
	def plot_tool_velocities(
		self, 
		tools=[],
		n_frames=FRAME_MAX,
		s_frame=0,
		plot_type='line',
		calc_type='standard'
	):
		velocity_calcs  =  {'standard':self.get_tool_velocities_standard,
							'iou':self.get_tool_velocities_iou}
		calc = velocity_calcs[calc_type]
		# We have to get the velocity data
		velocities_dict = { 'frame': [], 'velocity': [], 'class': []}
		for tool in tools:
			velocities = calc(tool)
			velocities = velocities[s_frame:s_frame+n_frames]

			# Sets values of velocity dict: frames are numbered from s_frame to n_frames
			velocities_dict['frame'] += [i for i in range(s_frame, s_frame + len(velocities))]
			velocities_dict['velocity'] += [v for v in velocities]
			velocities_dict['class'] += [tool for i in range(len(velocities))]

		velocities_df = pd.DataFrame(velocities_dict)

		if plot_type == 'line':
			ax = velocities_df.plot.line(x='frame', y='velocity', c=TOOL_COLORS[tools[0]], title=tools[0]+' - '+calc_type)
		elif plot_type == 'scatter':
			ax = velocities_df.plot.scatter(x='frame', y='velocity', c=TOOL_COLORS[tools[0]], title=tools[0]+' - '+calc_type)
		elif plot_type == 'bar':
			ax = velocities_df.plot.bar(x='frame', y='velocity', color = TOOL_COLORS[tools[0]], title=tools[0]+' - '+calc_type)

		return ax.get_figure()

	"""Get n tools in view across frames"""
	def get_n_tools_in_view(self, ignore=[], n_frames=FRAME_MAX, s_frame=0):
		n_tools_in_view = []
		frames = self.frames[s_frame:s_frame+n_frames]

		for f in frames:
			n_tools_in_view.append(len(f.get_tools(ignore=ignore)))

		return n_tools_in_view

	"""
	Plot tool presence over course of trial
	"""
	def plot_tool_presence(
		self, 
		tools=[],
		tool_colors=TOOL_COLORS, 
		n_frames=FRAME_MAX, 
		s_frame=0
	):
		tools_data = {}
		for t in tools:
			tools_data[t] = []

		frames = self.frames[s_frame:s_frame+n_frames]

		for f in frames:
			for t in tools:
				if f.has_tool(t):
					tools_data[t].append(1)
				else:
					tools_data[t].append(0)

		# Now have to convert binary arrays into coordinates of rectanges that we will be plotting
		codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]

		vertices = []
		for ti, tool in enumerate(tools):
			state = False
			current_start = 0

			for i, present in enumerate(tools_data[tool]):
				if present == 1 and state:
					# continue as normal
					pass
				elif present == 0 and state:
					if i - current_start > 0:
						# we have come to the end of a block
						vertices.append({
							'path': [
								(current_start, ti),
								(current_start, ti+1),
								(i, ti+1),
								(i, ti),
								(current_start, ti)
							],
							'tool': tool
						})
					state = False
				elif present == 1 and not state:
					current_start = i
					state = True
				else:
					# do nothing
					pass

		fig = plt.figure()
		ax = fig.add_subplot(111)

		for verts in vertices:
			path = Path(verts['path'], codes)
			patch = patches.PathPatch(path, facecolor=tool_colors[verts['tool']], lw=0)
			ax.add_patch(patch)

		plt.yticks(np.arange(len(tools))+0.5, tools)

		ax.set_xlim(-1,len(frames)+1)
		ax.set_ylim(-0.25,len(tools)+0.25)
		
		return fig


"""
IndividualFrame

Describes an individual frame
"""
class IndividualFrame:
	def __init__(
		self,
		frame_size,
		frame_data,
		verbose=False,
		normalize=True
	):
		self.normalize = normalize
		self.verbose = verbose
		self.frame_size = frame_size

		self.frame_data = frame_data.reset_index()
		self.frame_name = self.frame_data['file'][0]
		self.frame_index = self.frame_data['f_index'][0]
		self.tools = self.get_tools()


	"""Get the tools in this frame"""
	def get_tools(self, ignore=[]):
		tools = []
		self.tools_present = {}

		if len(ignore) > 0:
			ignore = set(ignore)

		for i in range(self.frame_data.shape[0]):
			if self.frame_data['class'][i] in ignore:
				continue

			x1, y1, x2, y2 = self.frame_data.loc[i,['x1', 'y1', 'x2', 'y2']]

			# normalize to the size of the frame
			if self.normalize:
				x1 = x1/self.frame_size[0]
				y1 = y1/self.frame_size[1]
				x2 = x2/self.frame_size[0]
				y2 = y2/self.frame_size[1]

			tools.append(SurgicalTool(
				tool_name=self.frame_data['class'][i],
				bbox_coordinates=[x1, y1, x2, y2]
			))

			self.tools_present[self.frame_data['class'][i]] = True

		return tools

	"""Return whether or not this frame has a particular tool class"""
	def has_tool(self, tool):
		return tool in self.tools_present

	"""Get position of tools"""
	def get_tool_position(self, tool):
		relevant_tool = [i for i in self.tools if i.get_type() == tool]

		position = (None, None)
		if len(relevant_tool) == 1:
			position = relevant_tool[0].get_tool_position()

		if len(relevant_tool) > 1 and self.verbose:
			print(f"There were 2 or more instances of this tool in {self.frame_name}")

		return position

	"""Get position coordinates of tools"""
	def get_standard_coordinates(self, tool):
		relevant_tool = [i for i in self.tools if i.get_type() == tool]

		position = [None,None,None,None]
		if len(relevant_tool) == 1:
			position = relevant_tool[0].get_standard_coordinates()

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
			x1, y1, x2, y2 = [i for i in to.get_standard_coordinates()]
			x1, x2 = [int(i*self.frame_size[0]) for i in [x1, x2]]
			y1, y2 = [int(i*self.frame_size[1]) for i in [y1, y2]]

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


"""
Get participant ID from string
"""
def get_participant_id_from_string(s):
	# remove any file prefixes
	s = re.sub('.*/', '', s)
	s = re.search('^S[0-9]+', s).group(0)

	# remove the S prefix
	s = s[1:]

	return s

"""
Get trial image size given dataset path
"""
def get_trial_image_size(trial_id, datasetpath):
	from PIL import Image

	example_image = [i for i in os.listdir(datasetpath) if re.search(trial_id, i)][0]
	img = Image.open(os.path.join(datasetpath, example_image))
	return img.size

"""
Make a file that lists all of the frame sizes
"""
def make_frame_size_file(datasetpath, save_path):
	all_trials_in_cohort = set([re.search('S[0-9]+T[0-9]+', i).group(0) for i in os.listdir(os.path.join(datasetpath, 'JPEGImages'))])

	frame_sizes = { 'trial_id': [], 'w': [], 'h': [] }
	for trial_id in all_trials_in_cohort:
	    size_d = get_trial_image_size(
	        trial_id = trial_id, 
	        datasetpath = os.path.join(
	            datasetpath,
	            'JPEGImages'
	        ))
	    frame_sizes['trial_id'].append(trial_id)
	    frame_sizes['w'].append(size_d[0])
	    frame_sizes['h'].append(size_d[1])

	fs_df = pd.DataFrame(frame_sizes)
	fs_df.to_csv(save_path)
