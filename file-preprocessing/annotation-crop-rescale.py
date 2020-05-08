'''
crop-rescale-frames

The purpose of this sript is to rescale the annotation xml appropriately based on how we resecaled the image
This script should be applied on a directory that has xml files that are TF ready
'''

import sys
import os
import time
from math import ceil, floor

from lxml import etree

ANNOTATION_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
IMG_SIZE = 200 # What it was rescaled to

all_frames = [a for a in os.listdir(ANNOTATION_DIR)]

frame_counter = 0
start_time = time.time()
for a in all_frames:
	if frame_counter % 1000 == 0:
		print('Processed: ' + str(frame_counter) + '/' + str(len(all_frames)) + ' (' + str(ceil(time.time()-start_time)) + ')')
	frame_counter += 1

	# Overlay the fixed annotations on the rescaled image
	annotation_xml = etree.parse(os.path.join(ANNOTATION_DIR, a))

	og_w = 0
	og_h = 0
	for node in annotation_xml.iter('size'):
		for sn in node.iter('width'):
			og_w = float(sn.text)
		for sn in node.iter('height'):
			og_h = float(sn.text)

		amount_to_crop_x = (og_w - og_h)/3

		for corner in ['xmin', 'xmax', 'ymin', 'ymax']:    
			for xml_corner in annotation_xml.iter(corner):
				if corner == 'xmin' or corner == 'xmax':
					value = (float(xml_corner.text) - amount_to_crop_x)/(og_w - 2*amount_to_crop_x)
				else:
					value = float(xml_corner.text) / og_h
				xml_corner.text = str(round(IMG_SIZE * value))

	# create a new XML file with the results
	myfile = open(os.path.join(OUTPUT_DIR, a), "wb")
	myfile.write(etree.tostring(annotation_xml, pretty_print=True))

