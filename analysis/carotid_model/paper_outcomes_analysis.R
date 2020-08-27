library(tidyverse)
library(ggrepel)
library(magrittr)
library(reshape2)

plot_dir <- '~/Documents/USC/USC_docs/ml/surgical-training-project/data/'
file_to_use <- '~/Documents/USC/USC_docs/ml/surgical-training-project/data/UPDATED_Raw Data.xlsx - Sheet1.tsv'
load_file <- read_tsv(file_to_use)

relevant_columns <- read_tsv('~/Documents/USC/USC_docs/ml/surgical-training-project/data/paper_outcomes_columns.tsv', col_names =F)
outcomes_data_parkcity <- read_tsv('/Users/guillaumekugener/Documents/USC/USC_docs/ml/surgical-training-project/data/parkcity-trial-annotations.txt')

outcomes_data_parkcity$`Start Time`

convert_times_to_seconds <- function(ti) {
  h <- as.double(gsub('\'.*', '', ti)) * 60 * 60
  m <- as.double(gsub(':[0-9][0-9]$', '', gsub('^[0-9][0-9]:', '', ti))) * 60
  s <- as.double(gsub('.*:', '', ti))
  
  return(h+m+s)
}


outcomes_cleaned <- outcomes_data_parkcity %>%
  mutate(trial=case_when(
    Trial == 'I' ~ 1,
    Trial == 'II' ~ 2,
    T ~ -1
  )) %>%
  mutate(tth=as.double(`End Time` - `Start Time`)) %>%
  dplyr::select(SurveyID=`Study ID`, trial, ST=`Start Time`, ET=`End Time`, tth)


cleaned_dataset <- load_file %>% 
  filter(!is.na(SurveyID)) %>%
  dplyr::select(relevant_columns$X1) %>%
  gather(variable, value, -SurveyID, -`Baseline HR`) %>%
  mutate(trial=case_when(
    grepl('trial 1', variable, ignore.case = T) | variable %in% c('Cottonoid Success', 'Muscle Success') ~ 1,
    grepl('trial 2', variable, ignore.case = T) | variable %in% c('Cotto0oid Success', 'Muscle Success_1') ~ 2,
    grepl('trial 3', variable, ignore.case = T) | variable %in% c('Cottonoid Success_1', 'Muscle Success_2') ~ 3,
    T ~ -1
  )) %>%
  mutate(variable=tolower(gsub('^ ', '', gsub('[Tt]rial [123] ', '', variable)))) %>%
  mutate(variable=case_when(
    variable %in% c('cotto0oid') ~ 'cottonoid',
    variable %in% c('cotto0oid success', 'cottonoid success_1') ~ 'cottonoid success',
    variable %in% c('muscle success_1', 'muscle success_2') ~ 'muscle success',
    variable %in% c('start time (hhmmss)') ~ 'start time',
    T ~ variable
  )) %>%
  dcast(SurveyID + `Baseline HR` + trial ~ variable, value.var='value') %>%
  dplyr::select(-`start time`)

difference_t1_t2_stats <- cleaned_dataset %>%
  dplyr::select(SurveyID, trial, tth) %>%
  mutate(trial=paste0('T', trial)) %>%
  mutate(tth=as.double(tth)) %>%
  dcast(SurveyID ~ trial, value.var='tth') %>%
  arrange(-T1, T2, T3) %>%
  mutate(d=T1-T2) %>% 
  arrange(-d)

id_order <- difference_t1_t2_stats %$% SurveyID

cleaned_dataset %<>% mutate(SurveyID=factor(SurveyID, levels=id_order))
difference_t1_t2_stats %<>% mutate(SurveyID=factor(SurveyID, levels=id_order))
difference_t1_t2_stats %<>% mutate(Group=case_when(
  SurveyID %in% outcomes_data_parkcity$`Study ID` ~ 'Annotated',
  T ~ 'None'
))

compare_published_written_drj <- difference_t1_t2_stats %>%
  dplyr::select(SurveyID, T1, T2) %>%
  gather(trial, tth_paper, -SurveyID) %>%
  mutate(trial=gsub('T', '', trial)) %>%
  merge(., outcomes_cleaned, by=c('SurveyID', 'trial'))

compare_published_to_guillaume_plots <- ggplot(compare_published_written_drj, aes(tth, tth_paper, color=trial)) +
  geom_abline(slope=1, linetype=2) +
  geom_point() +
  xlab('TTH in Guillaume plots') +
  ylab('TTH in paper') +
  # geom_text_repel(data=compare_published_written_drj %>% filter(SurveyID ==613), aes(label=SurveyID)) +
  theme_bw()

ggsave(
  compare_published_to_guillaume_plots,
  filename = paste0(plot_dir, 'compare_published_to_guillaume_plots.png'), 
  width = 6, height = 4)


change_in_tth <- ggplot(difference_t1_t2_stats, aes(SurveyID, d, color=Group)) +
  geom_hline(yintercept = 0) +
  geom_point() +
  geom_text_repel(data=difference_t1_t2_stats %>% filter(Group !='None'), aes(label=SurveyID)) + 
  xlab('Study ID') + ylab('Change in TTH (trial 1 - trial 2)') + 
  ggtitle('Change in time to hemostasis from trial 1 to trial 2') +
  theme_bw() +
  theme(
    legend.position = 'none', 
    # legend.justification = c(1,1),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

ggsave(
  change_in_tth,
  filename = paste0(plot_dir, 'change_in_tth.png'), 
  width = 6, height = 4)


id_order_2 <- difference_t1_t2_stats %>%
  arrange(T1) %$% SurveyID
difference_t1_t2_stats_2 <- difference_t1_t2_stats %>% 
  mutate(SurveyID=factor(SurveyID, levels=id_order_2))


difference_t1_t2_stats_2 %<>% mutate(`trial 2 speed` = ifelse(T1 > T2, 'faster', 'slower')) %>%
  filter(!is.na(`trial 2 speed`))

improvement_plot <- ggplot(difference_t1_t2_stats_2) +
  geom_segment(aes(x=SurveyID, xend=SurveyID, y=T1, yend=T2, color=`trial 2 speed`), alpha=0.5) +
  geom_point(aes(SurveyID, T1), color='red') +
  geom_point(aes(SurveyID, T2), color='blue') +
  xlab('Study ID') + ylab('TTH') + 
  ggtitle('TTH in trials 1 (red points) and 2 (blue points)') +
  theme_bw() +
  theme(
    legend.position = c(0.99,0.01),
    legend.justification = c(0.99,0.01),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid = element_blank()
  )

ggsave(
  improvement_plot,
  filename = paste0(plot_dir, 'improvement_plot.png'), 
  width = 6, height = 4)



ggplot(cleaned_dataset %>% filter(!is.na(tth)) %>% head(200), aes(SurveyID, as.double(tth), color=factor(trial), group=factor(trial))) +
  geom_line() +
  geom_point()

cleaned_dataset %>% filter(!is.na(tth))
