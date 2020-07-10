'''
Contains all of the helper functions to process the dataset downloaded from dirve and to format it properly
'''

from lxml import etree

import tqdm
import zipfile
import os
from math import floor, ceil
import copy
import re
import shutil
import json
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from utils import plot_frame_with_bb

class SurgicalVideoAnnotation():
    def __init__(self, trial_id, file_path, total_frames, output_directory, delete_at_end=True, assume_missing_empty=True, annotations_only=True):
        self.trial_id = trial_id
        self.file_path = file_path
        self.total_frames = total_frames
        self.output_directory=output_directory
        self.classes = {}

        # We need to to keep track of missing annotations (or we assume they are empty)
        self.assume_missing_empty = assume_missing_empty
        self.missing_frames = []
        # Sometimes, there are tagging errors so we catch them here
        self.too_many_tags_frames = []

        # We need to unzip the file
        zf = zipfile.ZipFile(file_path, 'r')
        
        # Get the relevant json file
        self.images = [i for i in zf.namelist() if re.search('vott\\-json\\-export/.*jpeg$', i)]
        self.annotations_json = [i for i in zf.namelist() if re.search('vott\\-json\\-export/.*\\.json$', i)][0]

        zf.extractall() # It's in current directory

        # Parse the annotations
        self.create_directory()
        self.parse_annotations()

        # Copy the images from the annotations as well, if desired
        if not annotations_only:
            self.move_images()

        # Remove the driectory that we created
        if delete_at_end:
            for f in zf.namelist():
                if not os.path.exists(f):
                    continue

                # Delete theme
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
        zf.close()

    # Make the directory and all subdirectories we will need for this dataset
    def create_directory(self):
        for dir_to_make in [
            self.output_directory, 
            os.path.join(self.output_directory, 'Annotations'),
            os.path.join(self.output_directory, 'JPEGImages'),
            os.path.join(self.output_directory, 'ImageSets'),
            os.path.join(self.output_directory, 'ImageSets', 'Main')
        ]:
            try:
                os.mkdir(dir_to_make)
            except:
                pass


    def parse_annotations(self):
        f = open(self.annotations_json)
        data = json.load(f)

        # Check that all the frames were annotated
        if len(data['assets']) != self.total_frames:
            # print(f"Looks like not all the frames were annotated for {self.trial_id}. There are {self.total_frames} frames but only {len(data['assets'])} annotations")

            # Determine missing frames here
            annotated_frames = [data['assets'][i]['asset']['name'] for i in data['assets']]
            unannotated_frames = [re.sub('.*/', '', i) for i in [self.trial_id + '_frame_' + str((i+1)).zfill(8) + '.jpeg' for i in range(self.total_frames)] if re.sub('.*/', '', i) not in annotated_frames]

            for i in unannotated_frames:
                self.missing_frames.append(i)

        image_size = [-1, -1]

        # Iterate through all of the annotations for this file
        for i in data['assets']:
            frame_object = {
                'name': data['assets'][i]['asset']['name'],
                'height': data['assets'][i]['asset']['size']['height'],
                'width': data['assets'][i]['asset']['size']['width'],
                'tools': []
            }

            # Set the image size
            if image_size[0] == -1:
                image_size = [
                    data['assets'][i]['asset']['size']['width'], 
                    data['assets'][i]['asset']['size']['height']
                ]

            # Regions are the boxes drawn
            regions = data['assets'][i]['regions']

            for r in regions:
                region_class = r['tags'][0]
                # Check if we have wrong number of tags. These samples will be fixed by hand
                if len(r['tags']) != 1:
                    self.too_many_tags_frames.append({
                        'name': data['assets'][i]['asset']['name'],
                        'd': data['assets'][i]['regions']
                    })
                    # print(f"We have a problem with a region in {frame_object['name']} with tags. The object tags are: {r['tags']}")
                    # In this case, set the tag to be undefined, and we will manually define it after the dataset is generated
                    region_class = 'undefined'
                
                # Keep track of how many total objects of each class we have in our datasets
                if region_class not in self.classes:
                    self.classes[region_class] = 0
                self.classes[region_class] += 1

                frame_object['tools'].append({
                    'type': region_class,
                    'coordinates': [
                        (
                            round(r['boundingBox']['left']), 
                            round(r['boundingBox']['top'])
                        ),
                        (
                            round(r['boundingBox']['left'] + r['boundingBox']['width']),
                            round(r['boundingBox']['top'] + r['boundingBox']['height'])
                        )
                    ]
                })

            # Now generate the annotations file
            self.generate_frame_annotation_xml(
                frame_obj=frame_object,
                destination=os.path.join(self.output_directory, 'Annotations')
            )

        # Create the annotations for the missing frames
        if self.assume_missing_empty:
            for ef in self.missing_frames:
                fo = {
                    'name': ef,
                    'height': image_size[1],
                    'width': image_size[0],
                    'tools': []
                }

                # Now generate the annotations file
                self.generate_frame_annotation_xml(
                    frame_obj=fo,
                    destination=os.path.join(self.output_directory, 'Annotations')
                )

        f.close()

    def move_images(self):
        # Grab all of the image files and move them
        for f in self.images:
            # Remove the file if it already exists
            if os.path.exists(os.path.join(self.output_directory, 'JPEGImages', re.sub('.*/', '', f))):
                os.remove(os.path.join(self.output_directory, 'JPEGImages', re.sub('.*/', '', f)))
            shutil.move(f, os.path.join(self.output_directory, 'JPEGImages'))

    # Given a frame object 
    def generate_frame_annotation_xml(self, frame_obj, destination, prefix=''):        
        # create the file structure
        annotation = etree.Element('annotation')
        annotation.set('verified', 'no')
        
        frame_file_name = prefix + re.sub('.*/', '', frame_obj['name'])
        
        folder = etree.SubElement(annotation, 'folder')
        folder.text = 'images'
        
        filename = etree.SubElement(annotation, 'filename')
        filename.text = frame_file_name
        
        path = etree.SubElement(annotation, 'path')
        path.text = os.path.join(destination, frame_file_name)
        
        source = etree.SubElement(annotation, 'source')
        
        database = etree.SubElement(source, 'database')
        database.text = 'Unknown'
        
        size = etree.SubElement(annotation, 'size')
        for i in [['width', frame_obj['width']], ['height', frame_obj['height']], ['depth', 3]]:
            ele = etree.SubElement(size, i[0])
            ele.text = str(i[1])
            
        segmented = etree.SubElement(annotation, 'segmented')
        segmented.text = '0'
        
        for t in frame_obj['tools']:            
            obj = etree.SubElement(annotation, 'object')
            
            for i in [['name', t['type']], ['pose', 'Unspecified'], ['truncated', 0], ['difficult', 0]]:
                n = etree.SubElement(obj, i[0])
                n.text = str(i[1])
            
            bndbox = etree.SubElement(obj, 'bndbox')
            for i in [
                ['xmin', t['coordinates'][0][0]], ['ymin', t['coordinates'][0][1]], 
                ['xmax', t['coordinates'][1][0]], ['ymax', t['coordinates'][1][1]]
            ]:
                bele = etree.SubElement(bndbox, i[0])
                bele.text = str(i[1])
            
        # create a new XML file with the results
        myfile = open(os.path.join(destination, re.sub('\\.jpeg', '.xml', frame_file_name)), "wb")
        myfile.write(etree.tostring(annotation, pretty_print=True))


def get_total_frames(dir_with_image_zips, output_file_name):
    images_zips = [i for i in os.listdir(dir_with_image_zips) if re.search('\\.zip$', i)]

    total_frames = {
        'trial_id': [],
        'frames': []
    }

    for z in images_zips:
        trial_id = re.search('S[0-9]+T[0-9]+[ab]?', z).group(0)
        zf = zipfile.ZipFile(os.path.join(dir_with_image_zips, z), 'r')
        total_frames['trial_id'].append(trial_id)
        total_frames['frames'].append(len([i for i in zf.namelist() if re.search('\\.jpeg$', i)]))
        zf.close()

    return total_frames

def extract_and_move_images(dir_with_image_zips, output_directory, trials_to_process):
    images_zips = [i for i in os.listdir(dir_with_image_zips) if re.search('\\.zip$', i)]

    # Iterate through each zip and move the files over to our main directory
    for z in tqdm.tqdm(images_zips):
        trial_id = re.search('S[0-9]+T[0-9]+[ab]?', z).group(0)

        if trial_id not in trials_to_process:
            continue

        zf = zipfile.ZipFile(os.path.join(dir_with_image_zips, z), 'r') 
        zf.extractall()

        # Get the image files
        extracted_images = [i for i in zf.namelist() if re.search('\\.jpeg$', i)]

        # Move all of the image files
        for f in extracted_images:
            # Remove the file if it already exists
            if os.path.exists(os.path.join(output_directory, 'JPEGImages', re.sub('.*/', '', f))):
                os.remove(os.path.join(output_directory, 'JPEGImages', re.sub('.*/', '', f)))
            shutil.move(f, os.path.join(output_directory, 'JPEGImages'))

        # Delete the files after using them
        for f in zf.namelist():
            if not os.path.exists(f):
                continue

            # Delete them
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)

        zf.close()

