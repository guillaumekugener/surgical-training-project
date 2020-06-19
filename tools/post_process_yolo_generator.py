import numpy as np
import pandas as pd
import pickle
import re
import os
from math import floor

from lxml import etree

import tensorflow as tf
from tensorflow.keras.utils import Sequence

class PostProcessYoloGenerator(Sequence):
    def __init__(self, video_ids, video_data_dir, annotation_dir, frame_ids, classes_map, batch_size=32, seq_len=6, grid_size=32, n_objects=2, out_n_objects=5, shuffle=True):
        self.batch_size = batch_size
        self.video_ids = [k for k in video_ids] # The video ids of videos present in this generator
        self.video_offsets = video_ids # Counter intuitive but this is a dict that holds the start frame index
        self.video_data_dir = video_data_dir
        self.video_image_sizes = {} # We will determine them once and then avoid having to recalculate them
        self.annotation_dir = annotation_dir # All the annotations should be in this directory
        self.classes_map = classes_map # Dictionary of tool name to int id

        self.seq_len = seq_len
        self.grid_size = grid_size
        self.dim = (seq_len, grid_size * grid_size * 2) # Features
        self.n_objects = n_objects
        self.out_dim = out_n_objects # Because we are predicting score (1) + bounding boxes (4)

        self.frame_ids = frame_ids # All the frames to be processed by this generator
        self.__load_all_stats()

        self.shuffle = shuffle

        self.on_epoch_end()

    # Shuffle the sequences at the end of an epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.frame_ids))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Load the outputs from YOLO
    def __load_all_stats(self, ext='_stats.pkl'):
        self.data = {}
        for k in self.video_ids:
            with open(os.path.join(self.video_data_dir, k + ext), 'rb') as stats_array:
                self.data[k] = pickle.load(stats_array)
            # Also, set the video frame size (should be square)
            example_frame = [os.path.join(self.annotation_dir, i) for i in os.listdir(self.annotation_dir) if re.search('^' + k, i)][0]
            annotation_xml = etree.parse(example_frame)

            for node in annotation_xml.iter('size'):
                for sn in node.iter('width'):
                    self.video_image_sizes[k] = int(float(sn.text))
                for sn in node.iter('height'):
                    if int(float(sn.text)) != self.video_image_sizes[k]:
                        # The annotations are not square
                        print(f'The annotations for video {k} are not square')
            print(f'Annotations for {k} are {self.video_image_sizes[k]}x{self.video_image_sizes[k]}')

    def __get_frame_from_id(self, frame_id):
        # Have to map the frame_id to the video and the id
        video_id = re.sub('_.*', '', frame_id)
        fid = int(re.sub('(.*_frame_)|(_tool.*)', '', frame_id)) - self.video_offsets[video_id]

        # So if the frame id is negative we should not be looping around and returning the negative value...
        if fid < 0:
            return None

        frame_data = self.data[video_id][fid] # I think this is right lol...
        return frame_data

    # Generates our feature heatmap for a given frame
    def generate_heatmap(self, frame_id, objects=[]):
        # Create heatmaps object for each class (for each frame)
        heatmap = np.zeros((self.grid_size, self.grid_size))

        frame = self.__get_frame_from_id(frame_id)
        
        # This means we got to an edge. Just return all 0s for now
        # In the future, we may want to use some sort of masking here...
        if frame is None:
            return heatmap 

        score_sum = 0
        for oi in range(frame[3][0]):
            x1, y1, x2, y2 = [floor(i * self.grid_size) for i in frame[0][0][oi]]
            score = frame[1][0][oi]
            pred_class = frame[2][0][oi]

            # Skip if this object is not included in our 
            if pred_class not in objects:
                continue

            # x1, y1, x2, y2
            for y in range(y1, min(y2+1, self.grid_size)):
                for x in range(x1, min(x2+1, self.grid_size)):
                    # Need to normalize somehow ()
                    heatmap[y][x] += score
            score_sum += score

        # So we don't get a divide by 0 error
        if score_sum == 0:
            score_sum += 1
        return heatmap/score_sum

    def get_frame_annotations(self, frame_id, objects=[]):
        video_id = re.sub('_.*', '', frame_id)
        file_path = os.path.join(self.annotation_dir, re.sub('_tool.*', '', frame_id) + '.xml')
        annotation_xml = etree.parse(file_path)

        annotation_object = { 
            'path': re.sub('(.*/)|(\\.xml$)', '', file_path), 
            'objects': [] 
        }

        for obj in annotation_xml.findall('object'):
            true_class = obj.find('name').text # Have to turn this from text to index
            if true_class not in self.classes_map:
                continue # We are not training on this object

            if self.classes_map[true_class] not in objects:
                continue # We skip this object

            true_coordinates = []
            for corner in ['xmin', 'ymin', 'xmax', 'ymax']:
                true_coordinates.append(int(obj.find('bndbox').find(corner).text)/self.video_image_sizes[video_id])
            
            annotation_object['objects'].append({
                'class': true_class,
                'coords': true_coordinates
            })

        return annotation_object

    def __data_generation(self, frame_ids, frame_id_fill_length=8):
        # This is where we load are data in
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.out_dim * 5))

        # Each frame is labeled as <video_id>_<frame_id>_<tool_id>
        for i, frame_id in enumerate(frame_ids):
            # Store sample
            # Get the heatmaps for the frame_id and the five previous ones (so seq length is 6)
            video_id = re.sub('_.*', '', frame_id)
            frame_id_as_int = int(re.sub('(.*frame_)|(_tool.*)', '', frame_id))
            tool_id = int(re.sub('.*tool_', '', frame_id))

            heatmap_seq = []
            frame_labels = []

            for fid in range(frame_id_as_int - self.seq_len + 1, frame_id_as_int + 1):
                # Probably want to make this a function
                frame_id_full = video_id + '_frame_' + str(fid).zfill(frame_id_fill_length)
                # Heatmap of all object in this frame
                h1 = self.generate_heatmap(frame_id_full, objects=range(0, self.n_objects))
                h2 = self.generate_heatmap(frame_id_full, objects=[tool_id])

                # Flatten and combine the heatmaps (h1: is the full, h2: is of only the specific tool)
                heatmap_seq.append(np.concatenate((
                    np.array(h1).flatten(),
                    np.array(h2).flatten()
                )))

            # Now make the label
            annotation_for_frame = self.get_frame_annotations(frame_id, [tool_id])
            labels_all = []
            for o in annotation_for_frame['objects']:
                labels_all.append([1] + o['coords'])

            while len(labels_all) < self.out_dim:
                labels_all.append([0,0,0,0,0])

            # Store label
            X[i,] = np.array(heatmap_seq) # Should be sequence x grid length
            y[i,] = np.array(labels_all).flatten() # Should be 5 x number of objects per tool

        return X, y

    def __len__(self):
        return int(np.floor(len(self.frame_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        frame_ids = [self.frame_ids[k] for k in indexes]

        # Generate the data
        X, y = self.__data_generation(frame_ids)

        return(X, y)


        