#!/usr/bin/env python
# coding: utf-8

# # Generate Dataset
# 
# The purpose of this file is to generate the pascal style dataset that we will use for our ML project. It is mostly automated, apart from having to deal with updates to errors made in the process of annotations

# In[87]:


# show images inline
get_ipython().run_line_magic('matplotlib', 'inline')

# automatically reload modules when they have changed
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[88]:


import os
import sys
import json
import re
import zipfile
import tqdm


# In[89]:


path_to_repo = '/Users/guillaumekugener/Documents/USC/USC_docs/ml/surgical-training-project/'

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, os.path.join(path_to_repo, 'tools'))


# In[90]:


from drive_dataset import SurgicalVideoAnnotation, extract_and_move_images
from utils import plot_frame_with_bb


# In[91]:


import pandas as pd


# The drive directory is the directory of the zip files with all of the annotations. We download this directly from the drive and decompress it. The variable below points to its location

# In[82]:


from datetime import datetime


# In[132]:


todays_date = datetime.today().strftime('%Y%m%d')

surgical_git_dir = '/Users/guillaumekugener/Documents/USC/USC_docs/ml/surgical-training-project/'

drive_dir = '/Users/guillaumekugener/Downloads/Completed Annotations 1 FPS/'
true_image_dir = '/Users/guillaumekugener/Downloads/1 FPS Reduced/'

final_dataset_directory = '/Users/guillaumekugener/Documents/USC/USC_docs/ml/datasets/fps-1-uncropped/'
csv_of_total_frames = '/Users/guillaumekugener/Documents/USC/USC_docs/ml/surgical-training-project/data/total_frames.csv'

manually_fixed_cases = os.path.join(surgical_git_dir, 'data/manually_fixed_annotations.csv')


# We also need to download the original frames to put into the dataset. Below, we first unzip all the files to count the total number of frames (in case we need to create empty annotation files for images that do not have annotations (because they have no objects). We copy the images into our dataset at the end of this notebook

# In[133]:


images_zips = [i for i in os.listdir(true_image_dir) if re.search('\\.zip$', i)]

total_frames = {
    'trial_id': [],
    'frames': []
}

for z in images_zips:
    trial_id = re.search('S[0-9]+T[0-9]+[ab]?', z).group(0)
    zf = zipfile.ZipFile(os.path.join(true_image_dir, z), 'r')
    total_frames['trial_id'].append(trial_id)
    total_frames['frames'].append(len([i for i in zf.namelist() if re.search('\\.jpeg$', i)]))
    zf.close()

pd.DataFrame(total_frames).to_csv(csv_of_total_frames, index=False)


# In[134]:


total_frames = pd.read_csv(csv_of_total_frames)


# In[135]:


all_zips = [i for i in os.listdir(drive_dir) if i != '.DS_Store']


# In[136]:


frames_to_fix = []
all_dataset_objects = []


# In[137]:


# Iterate through all the trials and parse all the annotations we have so far
# all_zips = ['S814T2-annotations.zip']
for z in tqdm.tqdm(all_zips):
    trial_id = re.search('S[0-9]+T[0-9]+[ab]?', z).group(0)
    frames_total = total_frames[total_frames['trial_id']==trial_id]['frames'].values[0]
    ex = SurgicalVideoAnnotation(
        trial_id=trial_id,
        total_frames=frames_total,
        file_path=os.path.join(drive_dir, z),
        output_directory=final_dataset_directory,
        delete_at_end=True,
        annotations_only=True,
        manually_fixed_cases=manually_fixed_cases
    )
    
    frames_to_fix = frames_to_fix + ex.too_many_tags_frames
    all_dataset_objects = all_dataset_objects + ex.frame_objects


# In[138]:


all_objects_ds_df = pd.DataFrame(all_dataset_objects)


# Create the class map for the dataset below

# In[139]:


class_map = pd.DataFrame({'class': all_objects_ds_df['class'].unique()})
class_map = class_map[class_map['class'] != '']


# In[140]:


class_map.to_csv(os.path.join(final_dataset_directory, 'classes.name'), sep='\t', header=False, index=False)


# We look for undefined objects. We then manually inspect and fix these and save the results to a csv. The csv is then used in the future to fix the labels so we do not have to deal with this manual process again

# In[141]:


if all_objects_ds_df[all_objects_ds_df['class']=='undefined'].shape[0] > 0:
    all_objects_ds_df[all_objects_ds_df['class']=='undefined'].to_csv(os.path.join(surgical_git_dir, 'data', todays_date + '_fixed_annotations.csv'), index=False) # These are the ones we have to fix


# Move the images into our dataset. We should only move the trial for which we have annotations above

# In[142]:


extract_and_move_images(
    dir_with_image_zips=true_image_dir,
    output_directory=final_dataset_directory,
    trials_to_process=[re.sub('\\-.*', '', i) for i in all_zips]
)


# We can catch errors in the annotations below. We need to manually fix these

# In[143]:


for i in range(len(frames_to_fix)):
    print(frames_to_fix[i]['name'])
    plot_frame_with_bb(
        image_path=os.path.join(final_dataset_directory, 'JPEGImages', frames_to_fix[i]['name']),
        annotation_path=os.path.join(final_dataset_directory, 'Annotations', re.sub('\\.jpeg$', '.xml', frames_to_fix[i]['name']))
    )


# In[144]:


frames_in_current_ds = sum(total_frames[total_frames['trial_id'].isin([re.sub('\\-.*', '', i) for i in all_zips])]['frames'])


# In[145]:


print(f"Total frames in current version of ds: {frames_in_current_ds}")


# ## Define training and validation sets
# 
# Below, we create the training and validation csvs. For testing, we will use additional videos not in our original 46 (as this will have the least bias)

# In[146]:


# These were randomly selected
validation_trials = [
    'S306T1', 'S306T2',
    'S504T1', 'S504T2',
    'S612T1', 'S612T2',
    'S807T1', 'S807T2'
]


# In[147]:


all_frames_dataset = all_objects_ds_df['name'].unique()
all_frames_dataset.sort()


# In[148]:


frames_relevant_o = {
    'train': [i for i in all_frames_dataset if re.sub('_.*', '', i) not in validation_trials],
    'val': [i for i in all_frames_dataset if re.sub('_.*', '', i) in validation_trials]
}


for g in ['train', 'val']:
    pascal_training_csv = pd.DataFrame({ 'name': frames_relevant_o[g], 'inc': 1})
    print(f"Dataset {g} size: {pascal_training_csv.shape[0]} frames")
    pascal_training_csv.to_csv(
        os.path.join(final_dataset_directory, 'ImageSets/Main', 'surgical_1fps_' + g + '.txt'),
        sep = '\t', header=False, index=False
    )
    
    pascal_training_csv.head(100).to_csv(
        os.path.join(final_dataset_directory, 'ImageSets/Main', 'small_surgical_1fps_' + g + '.txt'),
        sep = '\t', header=False, index=False
    )
    


# Below is for the retinanet data

# In[149]:


validation_indices = all_objects_ds_df['name'].str.contains('|'.join(validation_trials))
retinanet_training_csv = all_objects_ds_df[~validation_indices].copy()
retinanet_validation_csv = all_objects_ds_df[validation_indices].copy()

# We have to convert the class label to an index first. 
# Read in the class.name file and use that order
class_map = pd.read_csv(os.path.join(final_dataset_directory, 'classes.name'), header=None)
class_dict_mapping = {}
for ki, k in enumerate(class_map[0]):
    class_dict_mapping[k] = ki

# We need to set the full path    
retinanet_training_csv.to_csv(
    os.path.join(final_dataset_directory, 'ImageSets/Main', 'retinanet_surgical_1fps_train.csv'),
    sep=',', header=False, index=False
)
retinanet_validation_csv.to_csv(
    os.path.join(final_dataset_directory, 'ImageSets/Main', 'retinanet_surgical_1fps_validation.csv'),
    sep=',', header=False, index=False
)


# ## Stats
# 
# Gives an overview of the dataset (number of frames in training, validation, and number of tools, number of trials, etc...)

# In[150]:


data_on_ds = {
    'Training': retinanet_training_csv,
    'Validation': retinanet_validation_csv
}
output_string = ""
for g in data_on_ds:
    stat_df = data_on_ds[g][['x1','class']].groupby('class').agg(['count'])
    output_string += f"--- {g} info ---\n\n"
    for i in range(len(stat_df.values)):
        tool = stat_df.index.values[i]
        total = stat_df.values[i][0]
        if tool == '':
            tool = 'None'
        output_string += f"\t{tool}: {total} ({round(total/data_on_ds[g].shape[0]*100, 1)}%)\n"
    output_string += f"\nTotal frames: {data_on_ds[g].shape[0]}\n\n"

    
text_file = open(os.path.join(final_dataset_directory, 'ImageSets/Main', 'stats_' + todays_date + '.txt'), "w")
text_file.write(output_string)
text_file.close()


# In[ ]:




