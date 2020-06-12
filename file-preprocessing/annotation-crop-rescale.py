'''
crop-rescale-frames

The purpose of this sript is to rescale the annotation xml appropriately based on how we resecaled the image
This script should be applied on a directory that has xml files that are TF ready
'''

import sys
import os
import time
import tqdm

from absl import app, flags, logging
from absl.flags import FLAGS

from math import ceil, floor

from lxml import etree

flags.DEFINE_string('source', '', 'path to annotation directory')
flags.DEFINE_string('output', '','path to output directory')
flags.DEFINE_string('original_size', '1280x720', 'size of original image (pass as NNNxNNN)')
flags.DEFINE_integer('size', 200, 'resize images to')
flags.DEFINE_integer('buffer', 10, 'adds a buffer to the cropping')

FLAGS(sys.argv)

ANNOTATION_DIR = FLAGS.source
OUTPUT_DIR = FLAGS.output
IMG_SIZE = FLAGS.size
image_shape = [int(i) for i in FLAGS.original_size.split('x')]

all_frames = [a for a in os.listdir(ANNOTATION_DIR)]

for a in tqdm.tqdm(all_frames):
	# Overlay the fixed annotations on the rescaled image
	annotation_xml = etree.parse(os.path.join(ANNOTATION_DIR, a))

	og_w = 0
	og_h = 0
	for node in annotation_xml.iter('size'):
		for sn in node.iter('width'):
			og_w = float(sn.text)
			sn.text = str(IMG_SIZE) # Have to change the height and width
		for sn in node.iter('height'):
			og_h = float(sn.text)
			sn.text = str(IMG_SIZE) # Have to change the height and width

		left_crop = image_shape[0]/2-1 - image_shape[1]/2 - FLAGS.buffer
		for corner in ['xmin', 'xmax', 'ymin', 'ymax']:    
			for xml_corner in annotation_xml.iter(corner):
				if corner == 'xmin' or corner == 'xmax':
					value = min((float(xml_corner.text) - left_crop)/(og_w - 2*left_crop), IMG_SIZE)
					if value < 0:
						value = 0
				else:
					value = float(xml_corner.text) / og_h
				xml_corner.text = str(round(IMG_SIZE * value))

	# create a new XML file with the results
	myfile = open(os.path.join(OUTPUT_DIR, a), "wb")
	myfile.write(etree.tostring(annotation_xml, pretty_print=True))

