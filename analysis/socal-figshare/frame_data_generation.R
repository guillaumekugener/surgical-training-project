## Generate the SOCAL Complete CSV (cleaned) and mapping csv ##
library(tidyverse)
library(magrittr)

dataset_dir <- '~/Documents/USC/USC_docs/ml/datasets/fps-1-uncropped/'
published_dataset <- '~/Documents/USC/USC_docs/ml/surgical-training-project/analysis/socal-figshare/data/SOCAL-final/'

# Get all frames
all_frames = list.files(file.path(dataset_dir, 'JPEGImages'))

all_annotations <- read_csv(file.path(dataset_dir, 'ImageSets/Main/retinanet_surgical_1fps_complete.csv'), col_names = F) %>%
  mutate(X1=gsub('.*/', '', X1))

# Check that all frames are in the annotations file
if (length(setdiff(all_frames, all_annotations$X1)) > 0) {
  print('Same frames are missing!')
}

all_annotations %>%
  write.table(., file = file.path(published_dataset, 'socal.csv'), sep=',', quote=F, row.names=F, col.names=F)

# Decide if we want a header or not (going to go without since that seems to be the norm?)
mapping_df <- all_annotations %>%
  distinct(X1) %>%
  mutate(
    video_id=gsub('_.*', '', X1),
    video_index=as.numeric(gsub('\\.jpeg$', '', gsub('.*_', '', X1))),
    trial_id=stringr::str_extract(pattern='S[0-9]+T[0-9]+', string=gsub('_.*', '', X1))
  ) %>% 
  group_by(trial_id) %>%
  arrange(video_id, video_index) %>%
  mutate(true_index=row_number())

# Check that the video lengths are the same
left_join(
  mapping_df %>%
    group_by(trial_id) %>%
    dplyr::summarise(l1=length(trial_id)),
  mapping_df %>%
    group_by(trial_id, video_id) %>%
    dplyr::summarise(m=length(video_index)) %>%
    group_by(trial_id) %>%
    dplyr::summarise(l2=sum(m)),
  by='trial_id'
  ) %>% filter(l1!=l2)

mapping_df %>% distinct(trial_id) %>% nrow()

mapping_df %>%
  dplyr::select(frame=X1, trial_id, frame_number=true_index) %>%
  write.table(.,file = file.path(published_dataset, 'frame_to_trial_mapping.csv'),sep = ',', quote=F, row.names=F)

