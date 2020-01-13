import xml.etree.ElementTree as ET
import zipfile
import os

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
        
        # Because we are dealing with zips as download, we need to unzip and then get the path name
        zf = zipfile.ZipFile(file_path, 'r')
        dump_file_path = zf.namelist().pop()
        zf.extractall() # It's in current directory
        zf.close()
        
        tree = ET.parse(dump_file_path)
        self.xml = tree.getroot()
        self.frames = []
        
        self.__build_object()
        
        # Now remove the created dump file
        os.remove(dump_file_path)

    def __build_object(self):        
        # Generate frame objects for each annotated frame
        for frame in self.xml.iter('image'):
            self.frames.append(SurgicalVideoFrame(frame))

'''
SurgicalVideoFrame

This class represents a single frame of a video annotation of a surgical video
'''
class SurgicalVideoFrame:
    def __init__(self, frame):
        self.name = frame.attrib['name']
        self.xml = frame
        self.height = float(frame.attrib['height'])
        self.width = float(frame.attrib['width'])
        self.tools = []
        
        self.__build_object()
    
    def __build_object(self):
        
        for bt in self.xml.iter('box'):
            self.tools.append(SurgicalToolBoxed(bt))
    
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
    
    def plot(self, path_to_zip, exatract_to_path):
        full_image_location = os.path.join(path_to_zip, self.name)
        
        # Unzip the file containing the frames
        zf = zipfile.ZipFile(path_to_zip, 'r')
        
        zf.extract(self.name, path=exatract_to_path) # It's in current directory
        zf.close()
        
        # Plot the image of the frame
        print(os.path.join(exatract_to_path, self.name))
        im = np.array(Image.open(os.path.join(exatract_to_path, self.name)), dtype=np.uint8)

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
     
        
        
        
        
        
        
        
        
        
    