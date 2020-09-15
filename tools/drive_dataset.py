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

from utils import (
    plot_frame_with_bb, 
    make_frame_object_from_file,
    convert_frame_object_to_xml
)

class SurgicalVideoAnnotation():
    def __init__(self, trial_id, file_path, total_frames, output_directory, delete_at_end=True, assume_missing_empty=True, annotations_only=True, manually_fixed_cases=None):
        self.trial_id = trial_id
        self.file_path = file_path
        self.total_frames = total_frames
        self.output_directory=output_directory
        self.classes = {}
        self.manually_fixed_cases = pd.DataFrame({'name' : [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'class': []})
        if manually_fixed_cases is not None:
            self.manually_fixed_cases = pd.read_csv(manually_fixed_cases)

        # We need to to keep track of missing annotations (or we assume they are empty)
        self.assume_missing_empty = assume_missing_empty
        self.missing_frames = []
        # Sometimes, there are tagging errors so we catch them here
        self.too_many_tags_frames = []

        # We need to unzip the file
        zf = zipfile.ZipFile(file_path, 'r')
        
        # Get the relevant json file
        # Not all annotaters used the JSON format (some export in the PASCAL format)
        self.annotation_format = 'vott-json'
        if len([i for i in zf.namelist() if re.search('PascalVOC\\-export', i)]) > 0:
            self.annotation_format = 'PASCAL'
            self.images = [i for i in zf.namelist() if re.search('.*PascalVOC\\-export/JPEGImages/.*jpeg$', i)]
            self.annotations = [i for i in zf.namelist() if re.search('.*PascalVOC\\-export/Annotations/.*xml$', i)]
        else:
            self.images = [i for i in zf.namelist() if re.search('vott\\-json\\-export/.*jpeg$', i)]
            annotation_jsons = [i for i in zf.namelist() if re.search('vott\\-json\\-export/.*\\.json$', i)]
            
            # the export was not done as intructed or there was a problem in the export
            if len(annotation_jsons) == 0 or self.trial_id == 'S303T1':
                annotation_jsons = []
                zf.close()
                zf = zipfile.ZipFile(file_path, 'a')

                # the export did not include it, so we have to put the annotations together manually
                assets = [i for i in zf.namelist() if re.search('asset\\.json$', i)]
                data = {}
                json_data_name = 'vott-json-export/' + self.trial_id + '-export.json'
                for a in assets:
                    af = zf.open(os.path.join(a))
                    data[a] = json.load(af)
                    # print(data[a]['asset']['name'])

                with open('data.json', 'w') as f:
                    json.dump({ 'assets': data } , f)
                
                zf.write('data.json', json_data_name)
                # os.remove('data.json')
                annotation_jsons.append(json_data_name)
                zf.close()

                zf = zipfile.ZipFile(file_path, 'r')

            self.annotations_json = annotation_jsons[0]

        self.frame_objects = []

        zf.extractall() # It's in current directory

        # Parse the annotations
        self.create_directory()
        if self.annotation_format == 'PASCAL':
            # just need to move them to their destination
            self.move_annotations_xmls()    
        else:
            # have to parse the JSON
            self.parse_annotations()

        # Copy the images from the annotations as well, if desired
        if not annotations_only:
            self.move_images()

        # Remove the driectory that we created
        if delete_at_end:
            for f in zf.namelist() + ['_MACOSX', 'vott-json-export']:
                if not os.path.exists(f):
                    continue

                # Delete them
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
                region_class = re.sub('muscle patch', 'muscle', r['tags'][0])
                # Check if we have wrong number of tags. These samples will be fixed by hand
                if len(r['tags']) != 1:
                    # In this case, we had to manually fix it or will have to.
                    # Look in the manually fixed csv to see if we have dealth with this case before
                    region_class = 'undefined'

                    matching_region = self.manually_fixed_cases[
                        (self.manually_fixed_cases['name']==data['assets'][i]['asset']['name']) &
                        (self.manually_fixed_cases['x1']==floor(r['boundingBox']['left'])) &
                        (self.manually_fixed_cases['y1']==floor(r['boundingBox']['top'])) &
                        (self.manually_fixed_cases['x2']==min(image_size[0], ceil(r['boundingBox']['left'] + r['boundingBox']['width']))) &
                        (self.manually_fixed_cases['y2']==min(image_size[1], ceil(r['boundingBox']['top'] + r['boundingBox']['height'])))
                    ]

                    # If we found a single match, the change the class to what it should be
                    if matching_region.shape[0] == 1:
                        region_class = matching_region['class'].values[0]

                    # This is still a frame we need to fix manually
                    if region_class == 'undefined':
                        self.too_many_tags_frames.append({
                            'name': data['assets'][i]['asset']['name'],
                            'd': data['assets'][i]['regions']
                        })

                # Keep track of how many total objects of each class we have in our datasets
                if region_class not in self.classes:
                    self.classes[region_class] = 0
                self.classes[region_class] += 1

                frame_object['tools'].append({
                    'type': region_class,
                    'coordinates': [
                        (
                            floor(r['boundingBox']['left']), 
                            floor(r['boundingBox']['top'])
                        ),
                        (
                            min(image_size[0], ceil(r['boundingBox']['left'] + r['boundingBox']['width'])),
                            min(image_size[1], ceil(r['boundingBox']['top'] + r['boundingBox']['height']))
                        )
                    ]
                })

                # Add all our tools to frame objects
                self.frame_objects.append({
                    'name': data['assets'][i]['asset']['name'],
                    'x1': floor(r['boundingBox']['left']),
                    'y1': floor(r['boundingBox']['top']),
                    'x2': min(image_size[0], ceil(r['boundingBox']['left'] + r['boundingBox']['width'])),
                    'y2': min(image_size[1], ceil(r['boundingBox']['top'] + r['boundingBox']['height'])),
                    'class': region_class
                })

            # Now generate the annotations file
            self.generate_frame_annotation_xml(
                frame_obj=frame_object,
                destination=os.path.join(self.output_directory, 'Annotations')
            )

            # If number of regions is null, we need to add an empty row for this frame
            if len(regions) == 0:
                self.frame_objects.append({
                    'name': data['assets'][i]['asset']['name'],
                    'x1': '',
                    'y1': '',
                    'x2': '',
                    'y2': '',
                    'class': ''
                })

        # Create the annotations for the missing frames
        if self.assume_missing_empty:
            for ef in self.missing_frames:
                fo = {
                    'name': ef,
                    'height': image_size[1],
                    'width': image_size[0],
                    'tools': []
                }

                # Add this to our frame objects array
                self.frame_objects.append({
                    'name': ef,
                    'x1': '',
                    'y1': '',
                    'x2': '',
                    'y2': '',
                    'class': ''
                })

                # Now generate the annotations file
                self.generate_frame_annotation_xml(
                    frame_obj=fo,
                    destination=os.path.join(self.output_directory, 'Annotations')
                )

        f.close()

    def move_annotations_xmls(self):
        # Grab all of the annotation files and move them after reformating them
        img_size = [-1, -1]
        processed = []
        for f in self.annotations:
            file_dest = os.path.join(self.output_directory, 'Annotations', re.sub('.*/', '', f))
            # Remove the file if it already exists
            if os.path.exists(file_dest):
                os.remove(file_dest)
            shutil.move(f, file_dest)

            # Also edit them (so they have integer values)
            # and pull out the stats
            root = etree.parse(file_dest)
            if img_size[0] == -1:
                for ele in root.iter('width'):
                    img_size[0] = int(ele.text)
                for ele in root.iter('height'):
                    img_size[1] = int(ele.text)

            # Have to round all of the values
            for tag in ['xmin', 'ymin']:
                for ele in root.iter(tag):
                    ele.text = str(floor(float(ele.text)))

            for ele in root.iter('xmax'):
                ele.text = str(floor(min(img_size[0], ceil(float(ele.text)))))
            for ele in root.iter('ymax'):
                ele.text = str(floor(min(img_size[1], ceil(float(ele.text)))))

            # Now check if there are any problems with the tags. If there
            # are, we will have to manually fix them
            # TODO: not sure how these errors appear as of yet, so will
            # handle if I see them
            myfile = open(file_dest, "wb")
            myfile.write(etree.tostring(root, pretty_print=True))
            myfile.close()

            root = etree.parse(file_dest)
            fname = ''
            for fn in root.iter('filename'):
                fname = fn.text

            no_objects=True
            for o in root.iter('object'):
                no_objects=False
                d = {}
                for ele in o.iter('bndbox'):
                    for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
                        for e2 in ele.iter(tag):
                            d[tag] = int(e2.text)

                for cl in o.iter('name'):
                    d['class'] = cl.text

                # Now add the object
                self.frame_objects.append({
                    'name': fname,
                    'x1': d['xmin'],
                    'y1': d['ymin'],
                    'x2': d['xmax'],
                    'y2': d['ymax'],
                    'class': re.sub('muscle patch', 'muscle', d['class'])
                })

            # There were no object in this frame
            if no_objects:
                self.frame_objects.append({
                    'name': fname,
                    'x1': '',
                    'y1': '',
                    'x2': '',
                    'y2': '',
                    'class': ''
                })

            processed.append(fname)

        # If there are any images that were not processed,
        # add them as blank now
        self.missing_frames = [re.sub('.*/', '', i) for i in self.images if re.sub('.*/', '', i) not in processed]

        if self.assume_missing_empty:
            for ef in self.missing_frames:
                fo = {
                    'name': ef,
                    'height': img_size[1],
                    'width': img_size[0],
                    'tools': []
                }

                # Add this to our frame objects array
                self.frame_objects.append({
                    'name': ef,
                    'x1': '',
                    'y1': '',
                    'x2': '',
                    'y2': '',
                    'class': ''
                })

                # Now generate the annotations file
                self.generate_frame_annotation_xml(
                    frame_obj=fo,
                    destination=os.path.join(self.output_directory, 'Annotations')
                )

    def move_images(self):
        # Grab all of the image files and move them
        for f in self.images:
            # Remove the file if it already exists
            if os.path.exists(os.path.join(self.output_directory, 'JPEGImages', re.sub('.*/', '', f))):
                os.remove(os.path.join(self.output_directory, 'JPEGImages', re.sub('.*/', '', f)))
            shutil.move(f, os.path.join(self.output_directory, 'JPEGImages'))

    # Given a frame object 
    def generate_frame_annotation_xml(self, frame_obj, destination, prefix=''):        
        convert_frame_object_to_xml(
            frame_obj=frame_obj,
            destination=destination,
            prefix=prefix
        )

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

def create_retinanet_csv(all_objects_ds_df, dir_prefix, final_dataset_directory, csv_name, grouping):
    for g in grouping:
        ret_data_indicies = all_objects_ds_df['name'].str.contains('|'.join(['^' + gi for gi in grouping[g]])) 
        data_csv = all_objects_ds_df[ret_data_indicies].copy()
        data_csv['name'] = dir_prefix + data_csv['name']

        # We need to set the full path    
        data_csv.to_csv(
            os.path.join(final_dataset_directory, 'ImageSets/Main', f"{csv_name}_{g}.csv"),
            sep=',', header=False, index=False
        )

"""This trial was misnamed, so this function fixes the file names"""
def fix_S810T1b(
    image_dir,
    annotations_dir,
    complete_set_df
):
    # Fix the image names
    # Fix the annotation file names
    # Fix the image file names in the xml (of the annotations) and path
    relevant_images = [f for f in os.listdir(image_dir) if re.search('S810Tb', f)]
    relevant_annotation_files = [f for f in os.listdir(annotations_dir) if re.search('S810Tb', f)]

    for f in relevant_images:
        os.rename(os.path.join(image_dir, f), os.path.join(image_dir, re.sub('S810Tb', 'S810T1b', f)))

    for f in relevant_annotation_files:
        # Now fix the names and then rename the file
        old_file = os.path.join(annotations_dir, f)
        root = etree.parse(old_file)

        for tag in ['path', 'filename']:
            for ele in root.iter(tag):
                ele.text = re.sub('S810Tb', 'S810T1b', ele.text)

        myfile = open(old_file, "wb")
        myfile.write(etree.tostring(root, pretty_print=True))
        os.rename(old_file, os.path.join(annotations_dir, re.sub('S810Tb', 'S810T1b', f)))

    # Fix the data frame with all of the data points
    old_name = complete_set_df[complete_set_df['name'].str.contains('S810Tb')]['name'].unique()
    new_name = complete_set_df[complete_set_df['name'].str.contains('S810T1b')]['name'].unique()

    to_remove = [i for i in new_name if re.sub('S810T1b', 'S810Tb', i) in old_name]
    complete_set_df = complete_set_df[~complete_set_df['name'].isin(to_remove)].reset_index(drop=True)

    # Now fix all the names
    complete_set_df['name'] = [re.sub('S810Tb_frame', 'S810T1b_frame', i) for i in complete_set_df['name']]
    return complete_set_df.reset_index(drop=True)

"""Adds missing muscle patches for a set of annotations"""
def add_missing_muscle(
    muscle_annotations_path,
    image_dir,
    annotations_dir,
    complete_set_df
):
    # Need to add the muscle patches detected in the muscle annotations path
    # to the xmls and to the data frame
    muscle_patch_zips = [i for i in os.listdir(muscle_annotations_path) if re.search('\\.zip$', i)]

    assets_dict = {}
    for z in muscle_patch_zips:
        trial_id = re.search('S[0-9]+T[0-9]+[a-z]?', z).group(0)

        zf = zipfile.ZipFile(os.path.join(muscle_annotations_path, z), 'r')
        assets = [i for i in zf.namelist() if re.search('asset\\.json$', i) and re.search('^' + trial_id, i)]

        for a in assets:
            af = zf.open(a)
            assets_dict[a] = json.load(af)

            af.close()
        zf.close()

    data = { 'assets': assets_dict }

    # Has all of the new objects that need to be added
    frame_objects = []
    for i in data['assets']:
        frame_object = {
            'name': data['assets'][i]['asset']['name'],
            'height': data['assets'][i]['asset']['size']['height'],
            'width': data['assets'][i]['asset']['size']['width'],
            'tools': []
        }

        # Set the image size
        image_size = [
            data['assets'][i]['asset']['size']['width'], 
            data['assets'][i]['asset']['size']['height']
        ]

        # Regions are the boxes drawn
        regions = data['assets'][i]['regions']

        for r in regions:
            region_class = re.sub('muscle patch', 'muscle', r['tags'][0])
            # Check if we have wrong number of tags. These samples will be fixed by hand
            if len(r['tags']) != 1:
                print(f"Frame {frame_object['name']} has some errors")

            frame_object['tools'].append({
                    'type': region_class,
                    'coordinates': [
                        (
                            floor(r['boundingBox']['left']), 
                            floor(r['boundingBox']['top'])
                        ),
                        (
                            min(image_size[0], ceil(r['boundingBox']['left'] + r['boundingBox']['width'])),
                            min(image_size[1], ceil(r['boundingBox']['top'] + r['boundingBox']['height']))
                        )
                    ]
                })

        frame_objects.append(frame_object)

        # Now add to the xmls
        old_file = os.path.join(annotations_dir, re.sub('\\.jpeg$', '.xml', frame_object['name']))
        tree = etree.parse(old_file)
        root = tree.getroot()

        skip_update = False
        for ele in root.iter('name'):
            if ele.text == 'muscle':
                # print(f"The file {frame_object['name']} already has muscle annotated")
                skip_update = True

        if skip_update:
            continue

        for t in frame_object['tools']:
            tool_element = etree.SubElement(root, 'object')
            
            name = etree.SubElement(tool_element, 'name')
            name.text = t['type']
            pose = etree.SubElement(tool_element, 'pose')
            pose.text = 'Unspecified'
            for f in ['truncated', 'difficult']:
                fo = etree.SubElement(tool_element, f)
                fo.text = '0'

            bndbox = etree.SubElement(tool_element, 'bndbox')
            for i in [
                ['xmin', t['coordinates'][0][0]], ['ymin', t['coordinates'][0][1]], 
                ['xmax', t['coordinates'][1][0]], ['ymax', t['coordinates'][1][1]]
            ]:
                bele = etree.SubElement(bndbox, i[0])
                bele.text = str(i[1])
        # Save the new data
        myfile = open(old_file, "wb")
        myfile.write(etree.tostring(root, pretty_print=True))
        myfile.close()

    # Now have to add the muscles added above 
    # We also have to remove cases where previously that frame was labelled as blank
    rows_to_add = {
        'name': [],
        'x1': [],
        'y1': [],
        'x2': [],
        'y2': [],
        'class': []
    }
    for fo in frame_objects:
        for t in fo['tools']:
            # add it
            rows_to_add['name'].append(fo['name'])
            rows_to_add['x1'].append(t['coordinates'][0][0])
            rows_to_add['y1'].append(t['coordinates'][0][1])
            rows_to_add['x2'].append(t['coordinates'][1][0])
            rows_to_add['y2'].append(t['coordinates'][1][1])
            rows_to_add['class'].append(t['type'])

            # remove rows that were all blanks for this frame
            # unlikely to happen for muscle
            complete_set_df = complete_set_df[~((complete_set_df['name'] == fo['name']) & (complete_set_df['x1']==''))].reset_index(drop=True)

    return pd.concat([complete_set_df.reset_index(drop=True), pd.DataFrame(rows_to_add)])


"""
Replace failed QC with QC annotated
"""
def replace_qc_frames(
    reannotation_csv,
    qc_directory,
    final_dataset_directory,
    complete_set_df
):
    # We have to change muscle path to say muscle instead
    # We have to copy to annotations files over
    frames_needing_qc = pd.read_csv(reannotation_csv, skiprows=2)

    frame_objects = []
    # Iterate through each one of the frames above and replace it accordingly
    for i in range(frames_needing_qc.shape[0]):
        trial = frames_needing_qc.at[i,'Trial']
        fid = frames_needing_qc.at[i, 'Frame']

        image_file_name = f"{trial}_frame_{str(fid).zfill(8)}.jpeg"
        xml_file_name = f"{trial}_frame_{str(fid).zfill(8)}.xml"
        xml_file_path = os.path.join(qc_directory, xml_file_name)

        # Check if this files exists in QC
        if os.path.exists(xml_file_path):
            # We have to change any muscle patches to muscle in the objects
            frame_object = make_frame_object_from_file(xml_file_path, scale=False)
            frame_object['name'] = image_file_name

            frame_objects.append(frame_object)

            convert_frame_object_to_xml(
                frame_obj=frame_object,
                destination=os.path.join(final_dataset_directory, 'Annotations')
            )

        else:
            # It should be empty (could be missed as well but will assume empty...)
            frame_object = make_frame_object_from_file(
                os.path.join(final_dataset_directory, 'Annotations', xml_file_name)
            )

            # Clear out the tools
            frame_object['tools'] = []
            frame_objects.append(frame_object)

            convert_frame_object_to_xml(
                frame_obj=frame_object,
                destination=os.path.join(final_dataset_directory, 'Annotations')
            )

    # Now have to add the muscles added above 
    # We also have to remove cases where previously that frame was labelled as blank
    rows_to_add = {
        'name': [],
        'x1': [],
        'y1': [],
        'x2': [],
        'y2': [],
        'class': []
    }
    for fo in frame_objects:
        if fo['name'] == 'S301T2_frame_000000178.jpeg':
            print(fo)
        # Remove any cases of this frame being in the dataset
        complete_set_df = complete_set_df[~(complete_set_df['name'] == fo['name'])].reset_index(drop=True)
        if len(fo['tools']) > 0:
            for t in fo['tools']:
                # add it
                rows_to_add['name'].append(fo['name'])
                rows_to_add['x1'].append(t['coordinates'][0][0])
                rows_to_add['y1'].append(t['coordinates'][0][1])
                rows_to_add['x2'].append(t['coordinates'][1][0])
                rows_to_add['y2'].append(t['coordinates'][1][1])
                rows_to_add['class'].append(t['type'])
        else:
            # add blank rows
            rows_to_add['name'].append(fo['name'])
            rows_to_add['x1'].append('')
            rows_to_add['y1'].append('')
            rows_to_add['x2'].append('')
            rows_to_add['y2'].append('')
            rows_to_add['class'].append('')

    # Now go through the frame objects
    return pd.concat([complete_set_df.reset_index(drop=True), pd.DataFrame(rows_to_add)])

"""
Compile QC folder for re-annotation

Given a CSV of trial ids and frames, compiles a folder that can
be imported into VOTT to make edits to the frames

Args:
- reannotation_csv 
    csv with columns for trial id and frame number needing rennotation
- image_directory
    directory that contains the images that need to be reannotated
- complete_annotation_directory
    directory that contains the VOTT assets for all of the trials
- local_directory_path
    where the folder will be created with all of the information locally
- annotator_directory_path
    the path of where the annotator will save this folder and load them
    into VOTT from
"""
def compile_reannotation_folder(
    reannotation_csv,
    image_directory,
    complete_annotation_directory,
    local_directory_path='/Users/guillaumekugener/Desktop/qc_vott/',
    annotator_directory_path='/Users/guillaumekugener/Desktop/qc_vott'
):
    
    re_ann_df = pd.read_csv(reannotation_csv, skiprows=2)

    try:
        os.mkdir(os.path.join(local_directory_path, 'images'))
        os.mkdir(os.path.join(local_directory_path, 'annotations'))
    except:
        pass

    # check if vott file exists
    vott_file = [a for a in os.listdir(os.path.join(local_directory_path, 'annotations')) if re.search('\\.vott$', a)]
    
    vott_setup = False
    if len(vott_file) == 1:
        vott_setup == True

        # Map the file names to their ids (from the vott file)
        vott_file_clean = os.path.join(local_directory_path, 'annotations', vott_file[0])
        vott_data = None
        with open(vott_file_clean) as f:
            vott_data = json.load(f)

        frame_to_asset_id = {}
        for a in vott_data['assets']:
            frame_to_asset_id[re.sub('.*/', '', vott_data['assets'][a]['name'])] = a

        # can move all of the assets over
        make_annotations(
            re_ann_df=re_ann_df,
            image_directory=image_directory,
            complete_annotation_directory=complete_annotation_directory,
            local_directory_path=local_directory_path,
            annotator_directory_path=annotator_directory_path,
            vott_file=vott_file_clean,
            vott_id_mapping=frame_to_asset_id
        )

    elif len(vott_file) == 0:
        move_images(
            re_ann_df=re_ann_df,
            image_directory=image_directory,
            complete_annotation_directory=complete_annotation_directory,
            local_directory_path=local_directory_path 
        )
    else:
        print(f"> 1 vott file")

def move_images(
    re_ann_df,
    image_directory,
    complete_annotation_directory,
    local_directory_path
):
    for i in range(re_ann_df.shape[0]):
        trial = re_ann_df.at[i, 'Trial']
        frame = str(re_ann_df.at[i, 'Frame']).zfill(8)

        image_file_name = f"{trial}_frame_{frame}.jpeg"

        # First we have to move the images into the folder
        shutil.copyfile(
            os.path.join(image_directory, image_file_name),
            os.path.join(local_directory_path, 'images', image_file_name)
        )

def make_annotations(
    re_ann_df,
    image_directory,
    complete_annotation_directory,
    local_directory_path,
    annotator_directory_path,
    vott_file,
    vott_id_mapping
):
    for i in range(re_ann_df.shape[0]):
        trial = re_ann_df.at[i, 'Trial']
        frame = str(re_ann_df.at[i, 'Frame']).zfill(8)

        image_file_name = f"{trial}_frame_{frame}.jpeg" 

        # Then we move the annotations into the folder
        assets_zip = [a for a in os.listdir(complete_annotation_directory) if re.search(f"{trial}.*zip$", a)]
        if len(assets_zip) != 1:
            print('error')

        az = assets_zip.pop() 
        zf = zipfile.ZipFile(f"{complete_annotation_directory}{az}", 'r')

        candidate_assets = [a for a in zf.namelist() if re.search('^' + trial + '.*asset\\.json$', a)]
        
        # Have to find the annotations asset.json file
        for a in candidate_assets:
            af = zf.open(a)
            data = json.load(af)
    
            if re.sub('.*/', '', data['asset']['name']) == image_file_name:
                # This is the file that we want to copy into our assets
                # (1) Change the data to match the new ids
                data['asset']['id'] = vott_id_mapping[image_file_name]
                data['asset']['name'] = image_file_name
                data['asset']['path'] = f"file:{local_directory_path}images/{image_file_name}"
                
                # (2) Save it to our new location
                with open(os.path.join(local_directory_path, 'annotations', vott_id_mapping[image_file_name] + '-asset.json'), 'w') as f:
                    json.dump(data, f)

                af.close()
                break # We found the matching file and moved it, so down for this image

            af.close()
        zf.close()
    












