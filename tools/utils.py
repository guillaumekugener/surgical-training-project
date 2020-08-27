import numpy as np
import pandas as pd
import tqdm
import re
import copy
import os
from lxml import etree

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

# Get the total objects in the video
def total_objects_to_detect_in_video(all_frames, classes_map):
    positives = {}
    for a in tqdm.tqdm(all_frames):
        # I need to get the TOTAL_POSITIVES from the annotations themselves
        annotation_xml = etree.parse(a)

        # Now iterate through and see if we find a match
        # This is using a greedy approach
        for obj in annotation_xml.findall('object'):
            true_class = obj.find('name').text
            if true_class not in classes_map[0].tolist():
                continue # We skip the object
            if true_class not in positives:
                positives[true_class] = 0
            positives[true_class] += 1
    return(positives)

# Make a pandas dataframe from the detected objects
def make_df_of_detected_objects(data, video_id, video_start_id, classes_map, min_score=0.5):
    frame_ids = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    scores = []
    classes = []
    for fi, f in tqdm.tqdm(enumerate(data)):
        # Each frame
        # All the elements have the same length and we are going to generate a pandas df
        for bi in range(f[3][0]): 
            frame_ids.append(video_start_id + fi)
            x1.append(f[0][0][bi][0])
            y1.append(f[0][0][bi][1])
            x2.append(f[0][0][bi][2])
            y2.append(f[0][0][bi][3])
            scores.append(f[1][0][bi])
            classes.append(classes_map[0][int(f[2][0][bi])])

    final_df_all_objects = pd.DataFrame({
        'frame': frame_ids,
        'score': scores,
        'class': classes,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2
    })

    final_df_all_objects['video_id'] = [video_id for i in range(len(frame_ids))]
    
    final_df_all_objects = final_df_all_objects[final_df_all_objects['score'] > min_score]
    
    return(final_df_all_objects)

# Given the coordinates of two bounding boxes (I did not write this, found it online)
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

# For one frame, look at all objects in the frame and compare to the candidates
def compare_detected_to_annotation(real_objects, candidates, iou_thresh=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for obj in real_objects:
        match_score = -1
        match_o = None
        for ci in range(len(candidates)):
            q = candidates.pop()
            if q['class'] == obj['class']:
                iou_score = bb_intersection_over_union(q['coords'], obj['coords'])
                if iou_score > match_score and iou_score > iou_thresh:
                    if match_o != None:
                        # We found a better match for this object, 
                        # so put the previous good match back in the q
                        # I am not sure if this is how it should be implemented
                        candidates.append(match_o) 
                    match_o = q
                    match_score = iou_score
                    true_positives += 1 # This is only if it is above the IoU threshold
        if match_score == -1:
            false_negatives += 1
    false_positives += len(candidates)
    
    return({'tp': true_positives, 'fp': false_positives, 'fn': false_negatives})

# Given a list of dictionaries for model threshold, precision and recall, calculate
# the mAP. The list is assumed to be in increasing IoU threshold
def calculate_map(complete_final_scores):
    ap_sequence_array = []
    for final_scores in complete_final_scores:
        eleven_recall_points = [] # 0 -> 1
        for si, f in enumerate(final_scores):
            if f['recall'] == 0 and f['precision'] == 0:
                continue # Skip this case because the threshold was too high

            thresh = len(eleven_recall_points) * 0.1

            if f['recall'] > thresh:
                eleven_recall_points.append(f)

        # Then the remaining have a precision of 0
        if len(eleven_recall_points) != 11:
            while len(eleven_recall_points) != 11:
                eleven_recall_points.append({'precision': 0})

        AP_score = sum(i['precision']/len(eleven_recall_points) for i in eleven_recall_points)
        ap_sequence_array.append(AP_score)

    map_calc = sum(ap_sequence_array)/len(ap_sequence_array) # This is the mAP
    return(map_calc)

# Given a list of dictionaries of the frame id, the detected objects, their scores, classes, and coordinates and then real objects (from the annotations)
def calculate_precision_recall(pandas_detected_objects, annotation_dir, classes_map, video_id, IOU_THRESH = 0.5):
    # Now we should iterate through each row in the detected objects and label them as TP or FP
    # We also need to know how many total positives there are in our dataset
    IMG_SIZE = -1
    IOU_THRESH = 0.5

    prev_frame_id = -1
    candidates = []
    best_matches = []
    iou_values = []
    for i in tqdm.tqdm(pandas_detected_objects.index):
        frame_id = str(pandas_detected_objects.at[i, 'frame'])
        detected_class = pandas_detected_objects.at[i, 'class']
        detect_coords = [
            pandas_detected_objects.at[i, 'x1'],
            pandas_detected_objects.at[i, 'y1'],
            pandas_detected_objects.at[i, 'x2'],
            pandas_detected_objects.at[i, 'y2']
        ]
        # We have to add the 0 prefixes (I should really fix this...)
        while len(frame_id) < 8:
            frame_id = '0' + frame_id

        full_frame_file = video_id + '_frame_' + frame_id + '.xml'
        annotation_xml = etree.parse(os.path.join(annotation_dir, full_frame_file))

        if IMG_SIZE < 0:
            # Set the image size
            for node in annotation_xml.iter('size'):
                for sn in node.iter('width'):
                    IMG_SIZE = float(sn.text)
                for sn in node.iter('height'):
                    if float(sn.text) != IMG_SIZE:
                        # The annotations are not square
                        print('The annotations are not square so dimensions will be off')

        # Now iterate through and see if we find a match
        # This is using a greedy approach
        if prev_frame_id != frame_id:
            for obj in annotation_xml.findall('object'):
                true_class = obj.find('name').text
                if true_class not in classes_map[0].tolist():
                    continue # We skip the object

                true_coordinates = []
                for corner in ['xmin', 'ymin', 'xmax', 'ymax']:
                    true_coordinates.append(int(obj.find('bndbox').find(corner).text)/IMG_SIZE)

                candidates.append({'class': true_class, 'coords': true_coordinates})

        # Now determine the best match (by IoU)
        found_match = False
        for ci, c in enumerate(candidates):
            iou = bb_intersection_over_union(c['coords'], detect_coords)
            if iou > IOU_THRESH and c['class'] == detected_class:
                found_match = True
                candidates.pop()
                best_matches.append('TP')
                iou_values.append(iou)
                break # Since we found the match
             
        # This was a false positive
        if not found_match:
            best_matches.append('FP') # This was a false positive
            iou_values.append(0)

        prev_frame_id = frame_id
    
    # Add the TP and FP columns
    pandas_detected_objects['IOU'] = [i for i in iou_values]
    pandas_detected_objects['TP'] = [1 if b == 'TP' else 0 for b in best_matches]
    pandas_detected_objects['FP'] = [1 if b == 'FP' else 0 for b in best_matches]
    return pandas_detected_objects

# Calculate the mAP score given a df
def calculate_map(pandas_detected_objects, TOTAL_POSITIVES):
    sorted_dos = pandas_detected_objects.sort_values(by=['score'], ascending=False)
    sorted_dos['ACC_TP'] = sorted_dos['TP'].cumsum()
    sorted_dos['ACC_FP'] = sorted_dos['FP'].cumsum()

    sorted_dos['precision'] = sorted_dos['ACC_TP'] / (sorted_dos['ACC_TP'] + sorted_dos['ACC_FP'])
    sorted_dos['recall'] = sorted_dos['ACC_TP'] / TOTAL_POSITIVES

    unique_idx = sorted_dos.groupby(['recall'])['precision'].transform(max) == sorted_dos['precision']

    # Now we need to go through and do the thresholding thing
    eleven_points = []
    current_threshold_val = 0
    for i in sorted_dos[unique_idx].index.values:
        recall = sorted_dos.at[i, 'recall']
        if recall > current_threshold_val:
            eleven_points.append(sorted_dos.at[i, 'precision'])
            current_threshold_val += 0.1

    if len(eleven_points) != 11:
        print("The AP array is missing points")
        while len(eleven_points) < 11:
            eleven_points.append(0)
    map_value = sum(eleven_points)/len(eleven_points)
    
    return(map_value)
    
# Frames should be a list of tubles where the first element is the file name
# of the annotation and the second 
def combine_predictions_and_ground_truth(all_frames, classes_map, detected_objects, annotation_directory=''):
    # For each object in each frame, determine if it was detected
    IMG_SIZE = -1
    all_frames_objects = []


    for a in tqdm.tqdm(all_frames):
        current_frame_id = int(re.sub('(.*_frame_0+)|(\\.xml$)', '', a))
        annotation_xml = etree.parse(os.path.join(annotation_directory, a))

        if IMG_SIZE < 0:
            # Set the image size
            for node in annotation_xml.iter('size'):
                for sn in node.iter('width'):
                    IMG_SIZE = float(sn.text)
                for sn in node.iter('height'):
                    if float(sn.text) != IMG_SIZE:
                        # The annotations are not square
                        print('The annotations are not square so dimensions will be off')

        data_to_append = { 'frame_id': a, 'real_objects': [], 'detected_objects': [] }
        for obj in annotation_xml.findall('object'):
            true_class = obj.find('name').text
            if true_class not in classes_map[0].tolist():
                continue # We skip the 
            true_coordinates = []
            for corner in ['xmin', 'xmax', 'ymin', 'ymax']:
                true_coordinates.append(int(obj.find('bndbox').find(corner).text)/IMG_SIZE)
            data_to_append['real_objects'].append({ 
                'class': true_class, 
                'coords': true_coordinates
            })

        matches = detected_objects[(detected_objects['frame'] == current_frame_id)]
        for i in matches.index:
            data_to_append['detected_objects'].append({
                'score': matches.at[i, 'score'],
                'class': matches.at[i,'class'],
                'coords': [
                    matches.at[i,'x1'], 
                    matches.at[i,'x2'], 
                    matches.at[i,'y1'], 
                    matches.at[i,'y2']
                ]
            })

        all_frames_objects.append(data_to_append)
    return(all_frames_objects)

def make_frame_object_from_file(file_path, IMG_SIZE=(None, None), scale=True):
    annotation_xml = etree.parse(file_path)

    img_height = IMG_SIZE[1]
    img_width = IMG_SIZE[0]
    if IMG_SIZE[0] is None:
        # Set the image size
        for node in annotation_xml.iter('size'):
            for sn in node.iter('width'):
                img_width = float(sn.text)
            for sn in node.iter('height'):
                img_height = float(sn.text)

    if not scale:
        img_height = 1
        img_width = 1

    data_to_append = { 'frame_id': file_path, 'objects': [] }
    for obj in annotation_xml.findall('object'):
        true_class = obj.find('name').text
        
        true_coordinates = []
        for corner in ['xmin', 'ymin', 'xmax', 'ymax']:
            denom = img_height
            if corner in ['xmin', 'xmax']:
                denom = img_width
            true_coordinates.append(float(obj.find('bndbox').find(corner).text)/denom)
        data_to_append['objects'].append({ 
            'class': true_class, 
            'coords': true_coordinates
        })

    return data_to_append

def plot_frame_with_bb(image_path, annotation_path):
    img_array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2BGR)
    fig,ax = plt.subplots(1)
    ax.imshow(img_array)

    frame_object = make_frame_object_from_file(annotation_path, IMG_SIZE=img_array.shape[0])

    print(f"Image size: {img_array.shape}")

    # Draw the ground truth boxes
    for ro in frame_object['objects']:
        coords = [i * img_array.shape[0] for i in ro['coords']]
        rect = Rectangle(
            (coords[0], coords[1]), 
            coords[2] - coords[0], 
            coords[3] - coords[1], 
            linewidth=1,edgecolor='b',facecolor='none')

        print(f"{ro['class']} coordinates: {coords}")
        ax.add_patch(rect) 

    plt.show()


# Given a frame object, make a plot of the image
def plot_single_frame_with_outlines(frame_object, images_dir, score_thresh=0.5):
    image_path = os.path.join(images_dir, re.sub('\\.xml', '.jpeg', frame_object['frame_id']))
    img_array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2BGR)

    fig,ax = plt.subplots(1)
    ax.imshow(img_array)

    # Draw the ground truth boxes
    for ro in frame_object['real_objects']:
        coords = [i * img_array.shape[0] for i in ro['coords']]
        rect = Rectangle(
            (coords[0], coords[2]), 
            coords[1] - coords[0], 
            coords[3] - coords[2], 
            linewidth=1,edgecolor='b',facecolor='none')

        ax.add_patch(rect) 

    # Draw the detected truth boxes (above some scrore threshold (or just the top one))
    for ro in frame_object['detected_objects']:
        if ro['score'] > score_thresh:
            coords = [i * img_array.shape[0] for i in ro['coords']]
            rect = Rectangle(
                (coords[0], coords[2]), 
                coords[1] - coords[0], 
                coords[3] - coords[2], 
                linewidth=1,edgecolor='r',facecolor='none')

            ax.add_patch(rect) 
    plt.show()

# Parses all the ground truth labels and returns a pandas dataframe
def gt_as_pd_df(annotation_dir, IMG_SIZE=None):
    all_annotation_frames = [os.path.join(annotation_dir, a) for a in os.listdir(annotation_dir) if re.search('xml$', a)]
    all_annotation_frames.sort()

    final_data = {}

    df_cols = ['video_id', 'frame_id', 'tool', 'score', 'x1', 'y1', 'x2', 'y2']
    for c in df_cols:
        final_data[c] = []

    for a in tqdm.tqdm(all_annotation_frames):
        fo = make_frame_object_from_file(a, IMG_SIZE)
        
        video_id = re.sub('_.*', '', re.sub('.*/', '', fo['frame_id']))
        frame_id = re.sub('(.*_frame_)|(\\.xml$)', '', fo['frame_id'])
        
        for o in fo['objects']:
            final_data['video_id'].append(video_id)
            final_data['frame_id'].append(frame_id)
            final_data['tool'].append(o['class'])
            final_data['score'].append(1)
            
            for i, e in enumerate(['x1', 'y1', 'x2', 'y2']):
                final_data[e].append(o['coords'][i])

                
    final_data_df = pd.DataFrame(final_data)
    return final_data_df

def label_tp_fp_in_output(output_df, gt_df, iou_threshold=0.5):
    output_df = output_df.sort_values(by='score', ascending=False)

    all_candidates= set()
    row_match_type = []
    iou_col = []

    for ri in tqdm.tqdm(range(len(output_df))):
        row = output_df.iloc[ri,].values
        candidates = gt_df[(gt_df['video_id']==row[1]) & (gt_df['frame_id']==row[2]) & (gt_df['tool']==row[3])]

        match_type = 'FP'
        current_best_candidate = -1
        current_best_score = -1

        for i in candidates.index:
            #  Not a candidate if it already has matched
            if i in all_candidates:
                continue

            current_cand = candidates.loc[i,].values
            current_iou = bb_intersection_over_union(row[5:], current_cand[4:])
            
            if current_iou > iou_threshold and current_iou > current_best_score:
                if current_best_candidate >= 0:
                    all_candidates.remove(current_best_candidate) # Previous
                all_candidates.add(i) # Marked as taken
                current_best_score = current_iou
                match_type = 'TP'

        iou_col.append(current_best_score)
        row_match_type.append(match_type)

    output_df['IOU'] = [i for i in iou_col]
    output_df['TP'] = [1 if i == 'TP' else 0 for i in row_match_type]
    output_df['FP'] = [1 if i == 'FP' else 0 for i in row_match_type]
    return output_df   
