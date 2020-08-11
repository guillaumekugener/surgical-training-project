import numpy as np
import pandas as pd
import pickle
import re
import os
from math import floor
import time
import cv2

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
    # TODO: speed this up
    def generate_heatmap(self, frame_id, objects=[], score_threshold=0.05):
        # Create heatmaps object for each class (for each frame)
        heatmap = np.zeros((self.grid_size, self.grid_size))

        frame = self.__get_frame_from_id(frame_id)
        
        # This means we got to an edge. Just return all 0s for now
        # In the future, we may want to use some sort of masking here...
        if frame is None:
            return heatmap 

        score_sum = np.sum(frame[1][0])
        f_use = np.floor(frame[0][0]*self.grid_size).astype(int)

        for oi in range(frame[3][0]):
            x1, y1, x2, y2 = f_use[oi,:]
            score = frame[1][0][oi]

            pred_class = frame[2][0][oi]

            # Skip if this object is not included in our 
            if pred_class not in objects:
                continue

            # x1, y1, x2, y2
            heatmap[y1:min(y2+1, self.grid_size),x1:min(x2+1, self.grid_size)] += score

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

    def __generate_X(self, frame_id, frame_id_fill_length=8):
        # Get the heatmaps for the frame_id and the five previous ones (so seq length is 6)
        video_id = re.sub('_.*', '', frame_id)
        frame_id_as_int = int(re.sub('(.*frame_)|(_tool.*)', '', frame_id))
        tool_id = int(re.sub('.*tool_', '', frame_id))

        heatmap_seq = []
        frame_labels = []

        for fid in range(frame_id_as_int - self.seq_len + 1, frame_id_as_int + 1):
            tool_id = int(re.sub('.*tool_', '', frame_id))
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

        return np.array(heatmap_seq)

    def __data_generation(self, frame_ids, frame_id_fill_length=8):
        # This is where we load are data in
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.out_dim * 5))

        # Each frame is labeled as <video_id>_<frame_id>_<tool_id>
        for i, frame_id in enumerate(frame_ids):
            # Get the heatmaps for the frame_id and the five previous ones (so seq length is 6)
            X[i,] = self.__generate_X(frame_id, frame_id_fill_length) # Should be sequence x grid length

            # Now make the label
            tool_id = int(re.sub('.*tool_', '', frame_id)) # The only tool we care about in this particular frame

            annotation_for_frame = self.get_frame_annotations(frame_id, [tool_id])
            labels_all = []
            for o in annotation_for_frame['objects']:
                labels_all.append([1] + o['coords'])

            while len(labels_all) < self.out_dim:
                labels_all.append([0,0,0,0,0])

            # Store label
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

    def save_process_frames_as_npy_array(self, output_dir):
        for frame_id in self.frame_ids[:64]:
            X, _ = self.data_generation([frame_id])

            # Save this frame as numpy array
            np.save(os.path.join(output_dir, frame_id), X[0,])

    def data_generation_from_file(self, frame_ids, source_dir):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.out_dim * 5))

        for i, frame_id in enumerate(frame_ids):
            # Now make the label
            tool_id = re.sub('.*tool_', '', frame_id)

            annotation_for_frame = self.get_frame_annotations(frame_id, [tool_id])
            labels_all = []
            for o in annotation_for_frame['objects']:
                labels_all.append([1] + o['coords'])

            while len(labels_all) < self.out_dim:
                labels_all.append([0,0,0,0,0])

            X[i,] = np.load(os.path.join(source_dir, frame_id + '.npy'))
            y[i,] = np.array(labels_all).flatten() # Should be 5 x number of objects per tool
        
        return X, y


class PostProcessYoloDetectionGenerator(PostProcessYoloGenerator):
    def __init__(self, video_ids, video_data_dir, annotation_dir, frame_ids, classes_map, batch_size=32, seq_len=6, grid_size=32, n_objects=2, out_n_objects=5):
        self.batch_size = batch_size
        self.video_ids = [k for k in video_ids] # The video ids of videos present in this generator
        self.video_offsets = video_ids # Counter intuitive but this is a dict that holds the start frame index
        self.video_data_dir = video_data_dir
        self.video_image_sizes = {} # We will determine them once and then avoid having to recalculate them
        self.annotation_dir = annotation_dir # All the annotations should be in this directory
        self.classes_map = classes_map # Dictionary of tool name to int id
        self.index_to_class = [0] * len(classes_map)
        for i in self.classes_map:
            self.index_to_class[self.classes_map[i]] = i

        self.seq_len = seq_len
        self.grid_size = grid_size
        self.dim = (seq_len, grid_size * grid_size * 2) # Features
        self.n_objects = n_objects
        self.out_dim = out_n_objects # Because we are predicting score (1) + bounding boxes (4)

        self.frame_ids = frame_ids # All the frames to be processed by this generator
        self.__load_all_stats()

        self.shuffle = False

        self.on_epoch_end()

    # Load the outputs from YOLO
    def __load_all_stats(self, ext='_stats.pkl'):
        self.data = {}
        for k in self.video_ids:
            with open(os.path.join(self.video_data_dir, k + ext), 'rb') as stats_array:
                self.data[k] = pickle.load(stats_array)

    def __generate_X(self, frame_id, frame_id_fill_length=8):
        # Get the heatmaps for the frame_id and the five previous ones (so seq length is 6)
        video_id = re.sub('_.*', '', frame_id)
        frame_id_as_int = int(re.sub('(.*frame_)|(_tool.*)', '', frame_id))
        tool_id = int(re.sub('.*tool_', '', frame_id))

        heatmap_seq = []
        frame_labels = []

        for fid in range(frame_id_as_int - self.seq_len + 1, frame_id_as_int + 1):
            tool_id = int(re.sub('.*tool_', '', frame_id))
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

        return np.array(heatmap_seq)

    def __data_generation(self, frame_ids, frame_id_fill_length=8):
        # This is where we load are data in
        X = np.empty((self.batch_size, *self.dim))

        # Each frame is labeled as <video_id>_<frame_id>_<tool_id>
        for i, frame_id in enumerate(frame_ids):
            # Get the heatmaps for the frame_id and the five previous ones (so seq length is 6)
            X[i,] = self.__generate_X(frame_id, frame_id_fill_length) # Should be sequence x grid length

        return X

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        frame_ids = [self.frame_ids[k] for k in indexes]

        # Generate the data
        X = self.__data_generation(frame_ids)

        return X


class ObjectDetectionTemporalGenerator(Sequence):
    def __init__(
        self,
        csv_input_data,
        csv_labels_data,
        csv_class_map,
        shuffle=False,
        batch_size=8,
        seq_len=6,
        img_size=416,
        n_boxes=5,
        flatten_bbox_input=True
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.img_size = img_size
        self.n_boxes = n_boxes
        self.flatten_bbox_input = flatten_bbox_input

        # Set up the classes we care about
        class_map = pd.read_csv(csv_class_map, names=['class', 'ind'])
        self.class_dict = {}
        for i in range(class_map.shape[0]):
            self.class_dict[class_map.at[i, 'class']] = class_map.at[i, 'ind']
        
        self.input_data = pd.read_csv(csv_input_data, names=['file', 'score', 'x1', 'y1', 'x2', 'y2', 'class'])
        self.labels_data = pd.read_csv(csv_labels_data, names=['file', 'x1', 'y1', 'x2', 'y2', 'class'])

        # The ground truth (labels) csv contains all of the images in the dataset
        self.frame_ids = [i for i in set(self.labels_data['file'])]
        self.frame_ids.sort() # Easier to manage for when the data is not shuffled

        self.on_epoch_end()


    # Shuffle the sequences at the end of an epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.frame_ids))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.frame_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        frame_ids = [self.frame_ids[k] for k in indexes]

        # Generate the data
        X1, X2, y = self.__data_generation(frame_ids)
        return([X1, X2], y)

    """
    We have to return:
        - resize image of input
        - bounding boxes for this frame and previous n
        - ground truth labels for this frame
    """
    def __data_generation(self, frame_ids):
        X_img = np.zeros((self.batch_size, self.img_size, self.img_size, 3))
        X_bboxes = np.zeros((self.batch_size, self.seq_len, self.n_boxes, 6))

        y = np.zeros((self.batch_size, self.n_boxes, 6))

        for i, fid in enumerate(frame_ids):
            # First we load our image
            img_array = cv2.cvtColor(cv2.imread(fid), cv2.COLOR_RGB2BGR)

            y[i,] = self.__read_annotations(fid, img_array.shape)
            
            # Resize after processing the annotations, because we need the original image size
            # in order to normalize properly
            X_img[i,] = cv2.resize(img_array, (self.img_size, self.img_size))/255.0


            """
            Now we make the sequence of bboxes

            (1) Get the frames we need
            (2) Get the predictions made
            (3) Combine into sequence
            """
            for j, s_fid in enumerate(self.__get_prev_n_fids(fid, self.seq_len)):
                if s_fid is None:
                    continue

                X_bboxes[i,j,] = self.__get_predicted_bboxes(s_fid, img_array.shape)

        if self.flatten_bbox_input:
            X_bboxes = np.reshape(X_bboxes, (self.batch_size, self.seq_len, self.n_boxes*6))

        return X_img, X_bboxes, y

    """Given a frame id, returns a list of the frame plus the previous n"""
    def __get_prev_n_fids(self, frame_id, nf, zf=8):
        idn = int(re.sub('(.*_frame_)|(\\.jpeg$)', '', frame_id))
        return [re.sub('_frame_[0-9]+', '_frame_' + str(i).zfill(zf), frame_id) if i > 0 else None for i in range(idn-nf+1, idn+1)]

    """Get the model predicted bounding boxes"""
    def __get_predicted_bboxes(self, img, img_size):
        bboxes_predicted_for_image = self.input_data[self.input_data['file']==img].copy()
        bboxes_predicted_for_image = bboxes_predicted_for_image[bboxes_predicted_for_image['class'].isin(self.class_dict)]
        bboxes_predicted_for_image['class'] = [self.class_dict[k] for k in bboxes_predicted_for_image['class']]

        bboxes_as_np = np.zeros((self.n_boxes, 6))

        bboxes_as_np[:bboxes_predicted_for_image.shape[0],:] = np.array(bboxes_predicted_for_image[['score', 'x1', 'y1', 'x2', 'y2', 'class']]).astype(np.float32)
        bboxes_as_np[:,2] = bboxes_as_np[:, 2]/img_size[0]
        bboxes_as_np[:,4] = bboxes_as_np[:, 4]/img_size[0]
        bboxes_as_np[:,1] = bboxes_as_np[:, 1]/img_size[1]
        bboxes_as_np[:,3] = bboxes_as_np[:, 3]/img_size[1]

        return bboxes_as_np

    """Get the ground truth for this image (and resize)"""
    def __read_annotations(self, img, img_size):
        annotations_for_image = self.labels_data[self.labels_data['file']==img].copy()
        annotations_for_image = annotations_for_image[annotations_for_image['class'].isin(self.class_dict)]

        annotations_for_image['class'] = [self.class_dict[k] for k in annotations_for_image['class']]
        annotations_for_image['score'] = 1
        annotations_as_np = np.zeros((self.n_boxes, 6))

        annotations_as_np[:annotations_for_image.shape[0],:] = np.array(annotations_for_image[['score', 'x1', 'y1', 'x2', 'y2', 'class']]).astype(np.float32)
        annotations_as_np[:,2] = annotations_as_np[:, 2]/img_size[0]
        annotations_as_np[:,4] = annotations_as_np[:, 4]/img_size[0]
        annotations_as_np[:,1] = annotations_as_np[:, 1]/img_size[1]
        annotations_as_np[:,3] = annotations_as_np[:, 3]/img_size[1]
        
        return annotations_as_np
