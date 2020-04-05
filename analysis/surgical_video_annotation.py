import xml.etree.ElementTree as ET
import zipfile
import os
from math import floor, ceil
import copy
import re

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import numpy as np
'''
SurgicalVideoAnnotation

This python class contains all of the objects within an annotation zip file generated through CVAT
'''
class SurgicalVideoAnnotation:
    def __init__(self, file_path):
        self.file_path = file_path
        
        # Because we are dealing with zips as download, we need to unzip, parse the xml, and then delete the file we extracted
        zf = zipfile.ZipFile(file_path, 'r')
        dump_file_path = zf.namelist().pop()
        zf.extractall() # It's in current directory
        zf.close()
        
        tree = ET.parse(dump_file_path)
        self.xml = tree.getroot()
        self.total_frames = self.__get_total_frames()
        self.frames = [None] * self.total_frames
        
        self.__build_object()
        
        # Now remove the created dump file from above
        os.remove(dump_file_path)

    def __get_total_frames(self):
        for child in self.xml:
            if child.tag == 'meta':
                for cc in child:
                    if cc.tag == 'task':
                        for ccc in cc:
                            if ccc.tag == 'size':
                                return(int(ccc.text))

    def __build_object(self):        
        # Generate frame objects for each annotated frame
        # We need to add empty frames. We will do this by taking the first annotated frame and setting its
        # tools attribute to an empty list
        empty_frame = None
        frames_with_tools_id = {}
        
        # We want to have the id of the first frame and the index of the frame
        # Since not all videos start with any tools in view, it might not always be the first frame
        # in our annotations
        first_frame_index = -1
        frame_string_id_length = -1
        for frame in self.xml.iter('image'):
            if empty_frame is None:
                # Make the empty frame
                empty_frame = SurgicalVideoFrame(frame)
                empty_frame.tools = []
                empty_frame.xml = None
            
            current_frame = SurgicalVideoFrame(frame)
            frames_with_tools_id[current_frame.get_id()] = current_frame
            
            # When we add all the frames, we want the naming and ids to be properly indexed
            if first_frame_index == -1:
                p = re.compile("[0-9]+")
                # Set the name of the first frame
                first_frame_name = p.search(current_frame.name).group(0)
                frame_string_id_length = len(first_frame_name)
                first_frame_index = int(first_frame_name) - current_frame.get_id()

        # Now that we have all of the frames that have tools in them, it is time to add all of the frames
        # to our list of frames in our object
        for fi in range(len(self.frames)):
            if fi in frames_with_tools_id:
                self.frames[fi] = frames_with_tools_id[fi]
            # Add an empty frame
            else:
                new_empty_frame = copy.deepcopy(empty_frame)
                new_empty_frame.id = fi
                
                frame_id_name = str(fi + first_frame_index)
                while len(frame_id_name) < frame_string_id_length:
                    frame_id_name = '0' + frame_id_name
                
                new_empty_frame.name = 'frame_' + frame_id_name + '.jpeg'
                
                self.frames[fi] = new_empty_frame
                
    # Returns a binary list whether a tool is present in each frame.
    # The list is in tuples: (frame_name, 0 or 1)
    def tool_presence(self, tool_name):
        res = []
        
        for f in self.frames:
            if tool_name in f.get_tools_in_frame():
                res.append((f.name, 1))
            else:
                res.append((f.name, 0))
        return res
    
'''
SurgicalVideoFrame

This class represents a single frame of a video annotation of a surgical video
'''
class SurgicalVideoFrame:
    def __init__(self, frame):
        self.name = frame.attrib['name']
        self.id = int(frame.attrib['id'])
        self.xml = frame
        self.height = float(frame.attrib['height'])
        self.width = float(frame.attrib['width'])
        self.tools = []
        
        self.__build_object()
    
    def __build_object(self):
        
        for bt in self.xml.iter('box'):
            self.tools.append(SurgicalToolBoxed(bt))
    
    def get_id(self):
        return self.id
    
    # Return tools in this frame
    def get_tools_in_frame(self):
        tif = {}
        for t in self.tools:
            tif[t.get_type()] = True
        return tif
    
    # Get the quadrant location of each tool in the image
    def get_tool_quadrants(self):
        tool_positions = []
        for t in self.tools:
            # Get the position of the tool (e.g: using the center)
            t_pos = t.get_position()
            
            # Determine the quadrant it is in
            if t_pos[0] < self.width/2:
                if t_pos[1] < self.height/2:
                    tool_positions.append((t.get_type(), 'TL'))
                else:
                    tool_positions.append((t.get_type(), 'BL'))
            else:
                if t_pos[1] < self.height/2:
                    tool_positions.append((t.get_type(), 'TR'))
                else:
                    tool_positions.append((t.get_type(), 'BR'))
        
        return tool_positions
    
    # additional_sub_file: in the case where there are subdirectories within the zip file of the frames
    def plot(self, path_to_zip, extract_to_path, additional_sub_file=''):
        full_image_location = os.path.join(path_to_zip, self.name)
        
        # Unzip the file containing the frames
        zf = zipfile.ZipFile(path_to_zip, 'r')
        
        zf.extract(os.path.join(additional_sub_file, self.name), path=extract_to_path) # It's in current directory
        zf.close()
        
        # Plot the image of the frame
        print(os.path.join(extract_to_path, self.name))
        im = np.array(Image.open(os.path.join(extract_to_path, additional_sub_file, self.name)), dtype=np.uint8)

        # Create figure and axes
        fig,ax = plt.subplots(1)

        ax.imshow(im)
        
        # Plot boxes for each of the tools
        for t in self.tools:
            plotting_values = t.get_plotting_values()
            
            rect = patches.Rectangle(
                plotting_values['lc'], 
                plotting_values['w'],
                plotting_values['h'],
                linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

        plt.show()
        
    # For each tool in the frame, create and save an image of the tool (we would later use this for tool ID)
    # to generate a dataset
    def extract_tools_as_images(self, path_to_zip, extract_to_path, image_save_path):
        full_image_location = os.path.join(path_to_zip, self.name)
        
        # Unzip the file containing the frames
        zf = zipfile.ZipFile(path_to_zip, 'r')
        
        zf.extract(self.name, path=extract_to_path) # It's in current directory
        zf.close()
        
        # Plot the image of the frame
        im = np.array(Image.open(os.path.join(extract_to_path, self.name)), dtype=np.uint8)
        
        # Now extract individual tool images
        for t in self.tools:
            plotting_values = t.get_plotting_values() 

            
            tool_im = im[
                max(0, floor(t.coordinates[0][1])):min(im.shape[0], ceil(t.coordinates[1][1])),
                max(0, floor(t.coordinates[0][0])):min(im.shape[1], ceil(t.coordinates[1][0])),
                :
            ]
            
            im_new = Image.fromarray(tool_im)
            im_new.save(os.path.join(image_save_path, self.name + '_' + t.get_type() + '.jpeg'),  "JPEG")
            
        
'''
SurgicalToolBoxed

This class represents a general surgical tool present in the surgery
'''
class SurgicalToolBoxed:
    
    ## We provide the tool type, the top left coordinate of the box (xtl, ytl)
    ## and the bottom right coordinate of the box (xbr, ybr)
    def __init__(self, bt):
        self.type = [i.text for i in bt.iter('attribute')].pop() # If we add more attributes we will want to rethink this
        self.xml = bt
        self.coordinates = [
            (float(bt.attrib['xtl']), float(bt.attrib['ytl'])), 
            (float(bt.attrib['xbr']), float(bt.attrib['ybr']))
        ]
    
    # Get tool type
    def get_type(self):
        return self.type
    
    # Get the center of the box
    def get_center(self):
        mid_x = (self.coordinates[0][0] + self.coordinates[1][0])/2
        mid_y = (self.coordinates[0][1] + self.coordinates[1][1])/2
        return (mid_x, mid_y)
    
    # Get the position of the tool
    # In the future, we might change this based on the tool (for example, the top left coordinate (where
    # the grasping part of the grasper is) could be of more interest than the center. Or just the top part
    # of the box for the suction (which would be close to where suction is actually happening)
    def get_position(self):
        return self.get_center()
    
    # Get the top left corner and height + width for plotting
    def get_plotting_values(self):
        left_corner = self.coordinates[0]
        w = self.coordinates[1][0] - self.coordinates[0][0]
        h = self.coordinates[1][1] - self.coordinates[0][1]
        
        return { 'lc': left_corner, 'w': w, 'h': h }
    
    # Get the area of the box
    def get_box_area(self):
        w = self.coordinates[1][0] - self.coordinates[0][0]
        h = self.coordinates[1][1] - self.coordinates[0][1]
        return w*h
     
        
        
        
        
        
        
        
        
        
    