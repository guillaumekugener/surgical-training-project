library(tidyverse)
library(ggplot2)

redding_out <- read_tsv('~/Documents/USC/USC_docs/ml/surgical-training-project/file-preprocessing/output-test/redout-test.tsv') %>%
  mutate(index=rank(frame))


ggplot(redding_out, aes(index, channel_2, color=redded_out)) + 
  geom_point(size=0.5) +
  # scale_color_manual(c('blue', 'red')) +
  xlim(7200, 7555) + 
  theme_bw()


ggk_red_out <- read_tsv('~/Downloads/Red-Out - GGK Frames.tsv') %>%
  mutate(source='GGK')
djp_red_out <- read_tsv('~/Downloads/Red-Out - DJP Frames.tsv') %>%
  mutate(Notes=NA, source='DJP')

levels_order <- c('GGK', 'DJP')
red_outs_combined <- rbind(ggk_red_out, djp_red_out) %>%
  mutate(source=factor(source, levels=levels_order)) %>%
  mutate(y=as.integer(source))

redding_out_overall <- ggplot(red_outs_combined) +
  geom_rect(aes(xmin=`Start Frame`, xmax=`End Frame`, ymin=y-0.5, ymax=y+0.5)) +
  scale_y_continuous(expand = c(0,0), breaks=seq(1, length(levels_order)), labels=levels_order) +
  xlab('Frame') + ylab('Annotator') +
  theme_bw()

## Look at the agreement quantitatively
red_frames_ggk <- apply(ggk_red_out, 1, function(x) {
  seq(x[['Start Frame']], x[['End Frame']])
})

red_frames_ggk_all <- c()
for (rr in red_frames_ggk) {
  red_frames_ggk_all <- c(red_frames_ggk_all, rr)
}

red_frames_djp <- apply(djp_red_out, 1, function(x) {
  seq(x[['Start Frame']], x[['End Frame']])
})

red_frames_djp_all <- c()
for (rr in red_frames_djp) {
  red_frames_djp_all <- c(red_frames_djp_all, rr)
}

length(red_frames_djp_all)
length(red_frames_ggk_all)

overlapping_red_out <- intersect(red_frames_djp_all, red_frames_ggk_all)
overlapping_white_out <- intersect(
  setdiff(seq(1,10000), red_frames_djp_all),
  setdiff(seq(1,10000), red_frames_ggk_all)
)
