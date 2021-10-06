## Generate the SOCAL Outcomes Dataset ##
library(tidyverse)
library(magrittr)

data_dir <- file.path('~/Documents/USC/USC_docs/ml/surgical-training-project/analysis/socal-figshare/data')
outcomes <- read_tsv('~/Documents/USC/USC_docs/ml/surgical-training-project/data/carotid_outcomes/complete_data_set.tsv') %>%
  # Need to update the source column (needs to be more specific than 'USC')
  mutate(Source=ifelse(Source=='USC', paste0('USC_', gsub('[0-9][0-9][a-z]?$', '', SurveyID)), Source)) %>%
  mutate(Source=gsub('USC_USC1', 'USC_0', Source)) %>%
  mutate(Source=gsub('USC', 'SITE', Source)) %>%
  mutate(Source=gsub('Emory', 'SITE_00', Source)) %>%
  mutate(SurveyID=gsub('USC', 'A', SurveyID)) %>%
  mutate(SurveyID=gsub('E', 'A', SurveyID))

# Generate the participant level data
if (FALSE) {
  # To generate a blank file
  colnames(outcomes) %>%
    as.data.frame() %>%
    magrittr::set_colnames('og_column_name') %>%
    mutate(clean_column_name='') %>%
    mutate(column_description='') %>%
    mutate(participant_level=1, trial_level=1, Notes='') %>%
    write.table(., file = file.path(data_dir, 'dataset_column_description.csv'), sep = ',', quote=F, row.names=F)
}

# Load the column descriptions file
col_descriptions <- read_csv(file.path(data_dir, 'dataset_column_description.csv'))

if (nrow(col_descriptions %>% filter(trial_level==1, participant_level==trial_level)) > 1) {
  print("There are some columns in both")
}

## MAKE PARTICIPANT LEVEL DATA ##
participant_columns <- col_descriptions %>%
  filter(participant_level == 1)

# Check that there are no NAs being included
col_descriptions %>%
  filter(participant_level == 1) %>%
  filter(is.na(clean_column_name))

participant_data <- outcomes[,intersect(participant_columns$og_column_name, colnames(outcomes))]

participant_final_ds <- participant_data %>%
  gather(variable, value, -SurveyID) %>%
  dplyr::select(participant_id=SurveyID, og_column_name=variable, value) %>%
  left_join(., col_descriptions %>% dplyr::select(og_column_name, clean_column_name), by='og_column_name') %>%
  filter(clean_column_name %in% participant_columns$clean_column_name) %>%
  dcast(participant_id ~ clean_column_name, value.var='value') %>%
  dplyr::select(participant_columns$clean_column_name)

participant_final_ds %>% nrow()

participant_final_ds %>%
  write.table(., file = file.path(published_dataset, 'socal_participant_demographics.csv'), sep=',', quote=F, row.names=F)

## MAKE TRIAL LEVEL DATA ##
trial_level_columns <- col_descriptions %>%
  filter(trial_level == 1)

# Include the video resolutions in the trial level data
video_resolutions <- read_csv(file.path(dataset_dir, 'image_sizes.csv'))

# Check that there are no NAs being included
col_descriptions %>%
  filter(trial_level == 1) %>%
  filter(is.na(clean_column_name))

trial_level_data <- outcomes[,intersect(trial_level_columns$og_column_name, colnames(outcomes))]

trial_level_final_ds <- trial_level_data %>%
  gather(variable, value, -SurveyID) %>%
  dplyr::select(participant_id=SurveyID, og_column_name=variable, value) %>%
  left_join(., col_descriptions %>% dplyr::select(og_column_name, clean_column_name, trial_number), by='og_column_name') %>%
  filter(clean_column_name %in% trial_level_columns$clean_column_name) %>%
  mutate(trial_id=paste0('S', participant_id, 'T', trial_number)) %>%
  dcast(trial_id + participant_id ~ clean_column_name, value.var='value') %>%
  # Remove cases where neither tth or success have a value, as these trials do not exist
  filter(!is.na(tth) | !is.na(success)) %>%
  dplyr::select(c('trial_id', trial_level_columns$clean_column_name)) %>%
  left_join(., video_resolutions %>% dplyr::select(trial_id, trial_video_width=w, trial_video_height=h), by='trial_id')

# Are any trials in the participants data frame not in the trial level data
participant_final_ds %>%
  filter(!(participant_id %in% trial_level_final_ds$participant_id)) %>%
  nrow()

trial_level_final_ds %>% nrow()

trial_level_final_ds %>%
  write.table(., file = file.path(published_dataset, 'socal_trial_outcomes.csv'), sep=',', row.names=F, quote=F)
  
# Now make the base of the README.txt
descriptions_readme <- col_descriptions %>%
  filter(trial_level == 1 | participant_level == 1) %>%
  mutate(index=row_number()) %>%
  arrange(-participant_level, index) %>%
  mutate(readme_text=paste0(clean_column_name, '\n\t', column_description, '\n'))

descriptions_readme %>%
  filter(participant_level == 1) %>%
  dplyr::select(`## socal_participant_demographics.csv`=readme_text) %>%
  write.table(., file=file.path(published_dataset, 'PARTICIPANT_LEVEL_COL_DESC_README.txt'), sep='\t', quote=F, row.names=F)

descriptions_readme %>%
  filter(trial_level == 1) %>%
  dplyr::select(`## socal_trial_outcomes.csv`=readme_text) %>%
  distinct(`## socal_trial_outcomes.csv`) %>%
  rbind(data.frame(`## socal_trial_outcomes.csv`='trial_id\n\tUnique identifier for this trial') %>% magrittr::set_colnames(c('## socal_trial_outcomes.csv')), .) %>%
  rbind(.,
    data.frame(
      `## socal_trial_outcomes.csv`=c(
        'trial_video_width\n\tPixel width of the frames in this trial\n',
        'trial_video_height\n\tPixel height of the frames in this trial\n'
    )) %>% magrittr::set_colnames(c('## socal_trial_outcomes.csv'))) %>%
  write.table(., file=file.path(published_dataset, 'TRIAL_LEVEL_COL_DESC_README.txt'), sep='\t', quote=F, row.names=F)



