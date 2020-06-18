import pickle
import os
import numpy as np
import pandas as pd
import copy
from collections import deque
import random

from math import floor

from lxml import etree
import re
import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

class YoloOutput:
    def __init__(self, video_id, start_frame, data_path, classes_map, annotation_dir):
        self.video_id = video_id
        self.start_frame = start_frame
        self.classes_map = classes_map

        self.annotation_path = [os.path.join(annotation_dir, a) for a in os.listdir(annotation_dir) if re.search('^' + video_id, a)]
        self.annotation_path.sort()
        self.__set_annotation_image_size(self.annotation_path[0])

        self.__load_data(data_path)

    def __load_data(self, data_path):
        with open(data_path, 'rb') as stats_array:
            self.data = pickle.load(stats_array)

    def generate_heatmap(self, frame, objects=[], grid_size=32):
        # Create heatmaps object for each class (for each frame)
        heatmap = np.zeros((grid_size, grid_size))

        score_sum = 0
        for oi in range(frame[3][0]):
            x1, y1, x2, y2 = [floor(i * grid_size) for i in frame[0][0][oi]]
            score = frame[1][0][oi]
            pred_class = frame[2][0][oi]

            # Skip if this object is not included in our 
            if pred_class not in objects:
                continue

            # x1, y1, x2, y2
            for y in range(y1, min(y2+1, grid_size)):
                for x in range(x1, min(x2+1, grid_size)):
                    # Need to normalize somehow ()
                    heatmap[y][x] += score
            score_sum += score

        return(heatmap/score_sum)

    def __set_annotation_image_size(self, file_path):
        annotation_xml = etree.parse(file_path)
        img_size = -1

        # Set the image size
        if img_size < 0:
            for node in annotation_xml.iter('size'):
                for sn in node.iter('width'):
                    img_size = float(sn.text)
                for sn in node.iter('height'):
                    if float(sn.text) != img_size:
                        # The annotations are not square
                        print('The annotations are not square so dimensions will be off')

        self.annotation_img_size = img_size

    def get_frame_annotations(self, file_path):
        annotation_xml = etree.parse(file_path)

        annotation_object = { 
            'path': re.sub('(.*/)|(\\.xml$)', '', file_path), 
            'objects': [] 
        }
        for obj in annotation_xml.findall('object'):
            true_class = obj.find('name').text
            if true_class not in self.classes_map[0].tolist():
                continue # We skip this object

            true_coordinates = []
            for corner in ['xmin', 'ymin', 'xmax', 'ymax']:
                true_coordinates.append(int(obj.find('bndbox').find(corner).text)/self.annotation_img_size)
            
            annotation_object['objects'].append({
                'class': true_class,
                'coords': true_coordinates
            })
        return annotation_object


    # For each tool in each frame, creates a concatenated vector
    # This should have the inputs and outputs ordered
    # TODO: turn this into a data generator
    def generate_tool_heatmap_vector(self, output_size=5):
        input_matrix = []
        labels_matrix = []
        for fi, f in tqdm.tqdm(enumerate(self.data)):
            all_tools_heatmap = self.generate_heatmap(f, self.classes_map.index.values)
            frame_ground_truth = self.get_frame_annotations(self.annotation_path[fi])

            tools_heatmap = []
            tools_annotations = []
            for k in self.classes_map.index.values:
                specific_tools_heatmap = self.generate_heatmap(f, [k])

                feature_vector = np.concatenate((
                    all_tools_heatmap.flatten(), 
                    specific_tools_heatmap.flatten()
                ))
                tools_heatmap.append(feature_vector)

                # Now for the labels
                frame_tool_gt = []
                for o in frame_ground_truth['objects']:
                    if o['class'] == self.classes_map.at[k, 0]:
                        frame_tool_gt.append(o['coords'])
                

                while len(frame_tool_gt) != output_size:
                    frame_tool_gt.append([0,0,0,0])
                tools_annotations.append(np.array(frame_tool_gt).flatten())

            input_matrix.append(tools_heatmap)
            labels_matrix.append(tools_annotations)

        return({
            'input': input_matrix,
            'labels': labels_matrix
        })

    # Turns our dataset into a dataset of sequences prepped for an RNN model
    def turn_inputs_labels_into_sequences(self, data, seq_len=6, shuffle=True):
        sequential_data = []

        previous_inputs = deque(maxlen=seq_len)

        for i, row in tqdm.tqdm(enumerate(data['input'])):
            previous_inputs.append(row)

            if (len(previous_inputs) == seq_len):
                pin = np.array(previous_inputs)
                labs = np.array(data['labels'])[i]

                for ti in range(pin.shape[1]):
                    sequential_data.append({
                        'frame_id': self.annotation_path[i],
                        'tool': self.classes_map.at[ti, 0],
                        'input': pin[:,ti,:],
                        'label': labs[ti,:]
                    })

        return(sequential_data)

    # annotations_to_plot is a dictionary of bounding boxes to plot
    def plot_image_and_annotations(self, frame_path, annotations_to_plot, color_map={ 'yolo': 'green', 'lstm': 'red', 'real': 'blue'}):
        img_array = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_RGB2BGR)

        fig,ax = plt.subplots(1)
        ax.imshow(img_array)

        for a in annotations_to_plot:
            coords = [i * img_array.shape[0] for i in a['coords']]

            rect = Rectangle(
                (coords[0], coords[1]), 
                coords[2] - coords[0], 
                coords[3] - coords[1], 
                linewidth=1,edgecolor=color_map[a['source']],facecolor='none')

            ax.add_patch(rect) 

        plt.show()

# This is outside of the class because it can be run on a bunch of combined_datasets
# And will handle the shuffling of them as well
def prepared_for_training(self, sequential_data, shuffle=True):
    # Optional, in case we want to maitain the order (i.e prediction)
    # We will want to deal with cases where the 
    if shuffle:
        random.shuffle(sequential_data)

    frame_ids = [i['frame_id'] for i in sequential_data]
    tools = [i['tool'] for i in sequential_data]
    inputs = np.array([i['input'] for i in sequential_data])
    labels = np.array([i['label'] for i in sequential_data])
    return({ 'frame_ids': frame_ids, 'tools': tools, 'inputs': inputs, 'labels': labels })


