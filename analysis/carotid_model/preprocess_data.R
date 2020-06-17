## LOAD PACKAGES ##

library(tidyverse)
library(ggrepel)
library(magrittr)
library(reshape2)
library(ggpubr)
library(ggforce)
library(ggsci)
library(data.table)
library(DT)

## DATASET FILES ##
source_dir <- '~/Documents/USC/USC_docs/ml/surgical-training-project/data/carotid_outcomes/'
plot_dir <- file.path(source_dir, 'plots')
final_figure_dir <- file.path(source_dir, 'figures')
file_to_use <- file.path(source_dir, 'data', 'UPDATED_Raw Data.xlsx - Sheet1.tsv')
emory_data <- file.path(source_dir, 'data', 'Emory ETN-fixed.txt')

## LOAD DATASETS ##
raw_data <- read_tsv(file_to_use) %>%
  mutate(SurveyID=ifelse(is.na(SurveyID), paste0('USC', row.names(.)), SurveyID)) %>%
  mutate(Group=case_when(
    (!is.na(Attyears) & Attyears) >= 1 ~ 'Attending',
    (!is.na(Resyears) & Resyears) >= 1 ~ 'Trainee',
    grepl('USC', SurveyID) ~ 'Trainee',
    TRUE ~ 'None'
  )) %>%
  mutate(Resyears=ifelse(grepl('USC', SurveyID), NA, Resyears)) %>%
  # For now, filter out those that are not attendings or residents, but will want to come back to this later
  filter(Group != 'None') %>%
  dplyr::rename(`trial 2 ebl`=`trial 2  ebl`) %>%
  mutate(Source='USC')

# We want to add the Emory data to the raw data file we are using
emory_processed <- read_tsv(emory_data) %>%
  mutate(SurveyID=paste0('E', row.names(.))) %>%
  mutate(Group=case_when(
    (grepl('fellow', Year)) ~ 'Trainee',
    (grepl('pgy[0-9]+', Year)) ~ 'Trainee',
    (grepl('[0-9]+', Year)) ~ 'Attending',
    TRUE ~ 'None'
  )) %>% filter(Group != 'None') %>%
  mutate(
    `Trial 1 Success`=ifelse(`tiral 1 in second` >= 300, 0, 1),
    `Trial 2 Success`=ifelse(`Trial 2 time` >= 300, 0, 1)
  ) %>%
  dplyr::select(
    SurveyID,
    OtherSpec=Speciality,
    Totyears,
    `Trial 1 TTH`=`tiral 1 in second`, 
    `trial 1 ebl`=`Trial 1 EBL`,
    `Trial 1 Success`,
    `Trial 2 TTH`=`Trial 2 time`, 
    `trial 2 ebl`=`Trial 2 EBL`,
    `Trial 2 Success`,
    Group
  ) %>% mutate(Source='Emory') %>%
  # Filter incomplete data
  filter(`Trial 2 TTH`!=0, !is.na(`Trial 2 TTH`))


raw_data %<>% plyr::rbind.fill(., emory_processed)

# Create a specialty field
raw_data %<>% mutate(Specialty=case_when(
  NeurosurgerySpec == 'Neurosurgery' ~ 'Neurosurgery',
  ENTSpec == 'ENT / OTO-HNS' ~ 'Otolaryngology',
  OtherSpec == 'ent' ~ 'Otolaryngology',
  OtherSpec == 'nsg' ~ 'Neurosurgery',
  OtherSpec == 'GS' ~ 'General Surgery',
  grepl('USC', SurveyID) ~ 'Neurosurgery',
  TRUE ~ 'Unspecified'
))

# We include even those that did three trials for the summary of participants
complete_data <- raw_data
complete_data %>%
  write.table(., file = file.path(source_dir, 'complete_data_set.tsv'), sep='\t', quote=F, row.names=F)


# We should remove the people who did three trials for now, as they were not trained between trial 1 and 2 but trained between trials 2 and 3, so we will need to think about how to compare them to everyone else
raw_data %<>% filter(is.na(`Trial 3 Start Time (hhmmss)`))