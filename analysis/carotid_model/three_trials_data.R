source('~/Documents/USC/USC_docs/ml/surgical-training-project/analysis/carotid_model/preprocess_data.R')

trial3_data <- complete_data %>%
  filter(!is.na(`Trial 3 Success`))

# TTH
trial3_data %$%
  t.test(`Trial 1 TTH`, `Trial 2 TTH`, paired=T)$p.value
trial3_data %$%
  t.test(`Trial 2 TTH`, `Trial 3 TTH`, paired=T)$p.value
trial3_data %$%
  t.test(`Trial 1 TTH`, `Trial 3 TTH`, paired=T)$p.value

# EBL
trial3_data %$%
  t.test(`trial 1 ebl`, `trial 2 ebl`, paired=T)$p.value
trial3_data %$%
  t.test(`trial 2 ebl`, `trial 3 ebl`, paired=T)$p.value
trial3_data %$%
  t.test(`trial 1 ebl`, `trial 3 ebl`, paired=T)$p.value

data.frame(
  Success=c(100*sum(trial3_data$`Trial 1 Success`)/nrow(trial3_data), 100*sum(trial3_data$`Trial 2 Success`)/nrow(trial3_data),100*sum(trial3_data$`Trial 3 Success`)/nrow(trial3_data)),
  TTH=c(mean(trial3_data$`Trial 1 TTH`), mean(trial3_data$`Trial 2 TTH`), mean(trial3_data$`Trial 3 TTH`)),
  EBL=c(mean(trial3_data$`trial 1 ebl`), mean(trial3_data$`trial 2 ebl`), mean(trial3_data$`trial 3 ebl`))
) %>% set_rownames(c('Trial 1', 'Trial 2', 'Trial 3')) %>% t() %>%
  round(., 1) %>%
  as.data.frame() %>%
  mutate(Trial = row.names(.)) %>%
  dplyr::select(Trial, everything()) %>%
  write.table(., file = file.path(source_dir, 'figures', 'three_trial_data.tsv'), quote=F, sep='\t', row.names=F)
