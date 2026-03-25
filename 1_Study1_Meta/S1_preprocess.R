# ==============================================================================
# S1_preprocess.R - Data Preprocessing for Study 1
# ==============================================================================
# Purpose: Process raw meta-analysis data into analysis-ready format
# Output: df2_rt, df3_acc (effect size data frames)
# ==============================================================================

# Load packages
if (!requireNamespace('pacman', quietly = TRUE)) {
    install.packages('pacman')
}

pacman::p_load(
  here, tidyverse, bruceR, esc, metafor,
  dplyr, tidyr
)

# Load utility functions
source(here::here("analysis", "utils.R"))

# Disable scientific notation
options(scipen = 9999, digits = 5)

# ------------------------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------------------------
rawdata <- bruceR::import(here::here("Study1", "data", "dimension.xlsx")) %>%
  dplyr::filter(Save == 1) %>%
  # Create unique article ID from first author's name and publication year
  dplyr::mutate(
    First_Author = trimws(stringr::str_split(Author, ";", simplify = TRUE)[,1]),
    ArticleID = paste0(First_Author, Year)
  ) %>%
  dplyr::relocate(ArticleID, .before = Author)

n_articles_init <- length(unique(rawdata$Article_number))
n_groups_init <- nrow(rawdata)

# ------------------------------------------------------------------------------
# 2. Process RT Data (inline, following original logic)
# ------------------------------------------------------------------------------
data_rt <- rawdata %>% 
  dplyr::filter((Status_Practice %in% c(1, 2, 3)) &
                  Data_Integrity_Dimension == "yes" &
                  (Data_Integrity_RT %in% c("yes", "no mismatch"))) %>%
  dplyr::select(ArticleID, Article_number, Study, Group, Status_Practice, Consistent, Sample_size, 
                Female, Mean_age, SD_age, Trial_Condition, Experimetnal_design, 
                Ind_var_I_Matchness, Ind_var_II_Target_words, Ind_var_III, 
                Descriptive_data_RT_SD,
                'Practice_Number_Q', 'Practice_Number', 
                tidyr::starts_with('SnL_PresentTime'), 
                tidyr::starts_with('Response_Window'), 
                tidyr::starts_with('Picture'), 
                tidyr::starts_with('Target_meaning_'), 
                tidyr::starts_with('RT_M_mean_target_'),
                tidyr::starts_with('RT_M_sd_target_'),
                tidyr::starts_with('RT_NM_mean_target_'),
                tidyr::starts_with('RT_NM_sd_target_')) %>%
  dplyr::mutate(across(c('Sample_size', 'Female', 'Practice_Number', 'SnL_PresentTime', 
                        'Response_Window', tidyr::starts_with('RT_M_mean_target_'),
                        tidyr::starts_with('RT_M_sd_target_'),
                        tidyr::starts_with('RT_NM_mean_target_'),
                        tidyr::starts_with('RT_NM_sd_target_')), as.numeric)) %>%
  dplyr::mutate(across(tidyr::starts_with('Target_meaning_'), ~ tidyr::replace_na(., "NA"))) %>%
  dplyr::mutate(row_id = dplyr::row_number())

# Convert SE to SD
data_rt_se <- data_rt %>%
  dplyr::filter(grepl("SE", Descriptive_data_RT_SD, ignore.case = TRUE)) %>%
  dplyr::mutate(across(tidyr::starts_with('RT_M_sd_target_') | tidyr::starts_with('RT_NM_sd_target_'), 
                       ~ se_to_sd(., Sample_size)))

# Convert CI to SD
data_rt_ci <- data_rt %>%
  dplyr::filter(grepl("CI", Descriptive_data_RT_SD, ignore.case = TRUE)) %>%
  dplyr::mutate(across(tidyr::starts_with('RT_M_sd_target_') | tidyr::starts_with('RT_NM_sd_target_'), 
                       ~ ci_to_sd(., Sample_size)))

# Keep normal SD entries
data_rt_normal <- data_rt %>%
  dplyr::filter(!grepl("SE", Descriptive_data_RT_SD, ignore.case = TRUE) &
                  !grepl("CI", Descriptive_data_RT_SD, ignore.case = TRUE))

# Combine
combined_data_rt <- dplyr::bind_rows(data_rt_se, data_rt_ci, data_rt_normal) %>%
  dplyr::arrange(row_id)

# Extract self and stranger conditions
combined_data_rt_diff <- combined_data_rt %>%
  dplyr::mutate(
    RT_mean_match_self = ifelse(Target_meaning_1 == "self", RT_M_mean_target_1, as.numeric(NA)),
    RT_sd_match_self = ifelse(Target_meaning_1 == "self", RT_M_sd_target_1, as.numeric(NA)),
    RT_mean_match_stranger = dplyr::case_when(
      Target_meaning_2 == "stranger" ~ RT_M_mean_target_2,
      Target_meaning_3 == "stranger" ~ RT_M_mean_target_3,
      Target_meaning_5 == "stranger" ~ RT_M_mean_target_5,
      TRUE ~ as.numeric(NA)
    ),
    RT_sd_match_stranger = dplyr::case_when(
      Target_meaning_2 == "stranger" ~ RT_M_sd_target_2,
      Target_meaning_3 == "stranger" ~ RT_M_sd_target_3,
      Target_meaning_5 == "stranger" ~ RT_M_sd_target_5,
      TRUE ~ as.numeric(NA)
    ),
    RT_mean_match_diff = RT_mean_match_stranger - RT_mean_match_self
  ) %>%
  dplyr::mutate(Number = dplyr::row_number(), Female_ratio = Female/Sample_size) %>%
  dplyr::select(
    ArticleID, Study, Group, Status_Practice, Practice_Number, SnL_PresentTime, Response_Window, 
    Sample_size, Female, Female_ratio, Mean_age, SD_age, Trial_Condition, Experimetnal_design, 
    Ind_var_I_Matchness, Ind_var_II_Target_words, Ind_var_III, 
    RT_mean_match_self, RT_sd_match_self, RT_mean_match_stranger, RT_sd_match_stranger, RT_mean_match_diff
  ) %>%
  dplyr::filter(!is.na(Sample_size) & !is.na(Practice_Number) & !is.na(SnL_PresentTime) & 
         !is.na(Response_Window) & !is.na(RT_mean_match_self) & !is.na(RT_mean_match_stranger) & 
         !is.na(RT_sd_match_self) & !is.na(RT_sd_match_stranger))

n_articles_rt <- length(unique(combined_data_rt_diff$Article_number))
n_groups_rt <- nrow(combined_data_rt_diff)

# ------------------------------------------------------------------------------
# 3. Process ACC Data (inline, following original logic)
# ------------------------------------------------------------------------------
data_acc <- rawdata %>% 
  dplyr::filter((Status_Practice %in% c(1, 2, 3)) &
                  Data_Integrity_Dimension == "yes" &
                  (Data_Integrity_ACC %in% c("yes", "no mismatch"))) %>%
  dplyr::select(ArticleID, Article_number, Study, Group, Status_Practice, Consistent, Sample_size, 
                Female, Mean_age, SD_age, Trial_Condition, Experimetnal_design, 
                Ind_var_I_Matchness, Ind_var_II_Target_words, Ind_var_III, 
                Descriptive_data_Accuracy_SD,
                'Practice_Number_Q', 'Practice_Number', 
                tidyr::starts_with('SnL_PresentTime'), 
                tidyr::starts_with('Response_Window'), 
                tidyr::starts_with('Picture'), 
                tidyr::starts_with('Target_meaning_'), 
                tidyr::starts_with('ACC_M_mean_target_'),
                tidyr::starts_with('ACC_M_sd_target_'),
                tidyr::starts_with('ACC_NM_mean_target_'),
                tidyr::starts_with('ACC_NM_sd_target_')) %>%
  dplyr::mutate(across(c('Sample_size', 'Female', 'Practice_Number', 'SnL_PresentTime', 
                        'Response_Window', tidyr::starts_with('ACC_M_mean_target_'),
                        tidyr::starts_with('ACC_M_sd_target_'),
                        tidyr::starts_with('ACC_NM_mean_target_'),
                        tidyr::starts_with('ACC_NM_sd_target_')), as.numeric)) %>%
  dplyr::mutate(across(tidyr::starts_with('Target_meaning_'), ~ tidyr::replace_na(., "NA"))) %>%
  dplyr::mutate(row_id = dplyr::row_number())

# Convert SE to SD
data_acc_se <- data_acc %>%
  dplyr::filter(grepl("SE", Descriptive_data_Accuracy_SD, ignore.case = TRUE)) %>%
  dplyr::mutate(across(tidyr::starts_with('ACC_M_sd_target_') | tidyr::starts_with('ACC_NM_sd_target_'), 
                       ~ se_to_sd(., Sample_size)))

# Keep normal entries
data_acc_normal <- data_acc %>%
  dplyr::filter(!grepl("SE", Descriptive_data_Accuracy_SD, ignore.case = TRUE))

# Combine
combined_data_acc <- dplyr::bind_rows(data_acc_se, data_acc_normal) %>%
  dplyr::arrange(row_id)

# Extract self and stranger conditions
combined_data_acc_diff <- combined_data_acc %>%
  dplyr::mutate(
    ACC_mean_match_self = ifelse(Target_meaning_1 == "self", ACC_M_mean_target_1, as.numeric(NA)),
    ACC_sd_match_self = ifelse(Target_meaning_1 == "self", ACC_M_sd_target_1, as.numeric(NA)),
    ACC_mean_match_stranger = dplyr::case_when(
      Target_meaning_2 == "stranger" ~ ACC_M_mean_target_2,
      Target_meaning_3 == "stranger" ~ ACC_M_mean_target_3,
      Target_meaning_5 == "stranger" ~ ACC_M_mean_target_5,
      TRUE ~ as.numeric(NA)
    ),
    ACC_sd_match_stranger = dplyr::case_when(
      Target_meaning_2 == "stranger" ~ ACC_M_sd_target_2,
      Target_meaning_3 == "stranger" ~ ACC_M_sd_target_3,
      Target_meaning_5 == "stranger" ~ ACC_M_sd_target_5,
      TRUE ~ as.numeric(NA)
    ),
    ACC_mean_match_diff = ACC_mean_match_stranger - ACC_mean_match_self
  ) %>%
  dplyr::mutate(Number = dplyr::row_number(), Female_ratio = Female/Sample_size) %>%
  dplyr::select(
    ArticleID, Study, Group, Status_Practice, Practice_Number, SnL_PresentTime, Response_Window, 
    Sample_size, Female, Female_ratio, Mean_age, SD_age, Trial_Condition, Experimetnal_design, 
    Ind_var_I_Matchness, Ind_var_II_Target_words, Ind_var_III, 
    ACC_mean_match_self, ACC_sd_match_self, ACC_mean_match_stranger, ACC_sd_match_stranger, ACC_mean_match_diff
  ) %>%
  dplyr::filter(!is.na(Sample_size) & !is.na(Practice_Number) & !is.na(SnL_PresentTime) & 
         !is.na(Response_Window) & !is.na(ACC_mean_match_self) & !is.na(ACC_mean_match_stranger) & 
         !is.na(ACC_sd_match_self) & !is.na(ACC_sd_match_stranger))

n_articles_acc <- length(unique(combined_data_acc_diff$Article_number))
n_groups_acc <- nrow(combined_data_acc_diff)

# ------------------------------------------------------------------------------
# 4. Effect Size Calculation
# ------------------------------------------------------------------------------

# RT Effect Sizes
combined_data_rt_diff$cohen_d_rt <- NA
combined_data_rt_diff$vi_rt <- NA

for (i in seq_len(nrow(combined_data_rt_diff))) {
  result <- esc::esc_mean_sd(
    grp1m = combined_data_rt_diff$RT_mean_match_stranger[i],
    grp1sd = combined_data_rt_diff$RT_sd_match_stranger[i],
    grp1n = combined_data_rt_diff$Sample_size[i],
    grp2m = combined_data_rt_diff$RT_mean_match_self[i],
    grp2sd = combined_data_rt_diff$RT_sd_match_self[i],
    grp2n = combined_data_rt_diff$Sample_size[i],
    es.type = "d"
  )
  combined_data_rt_diff$cohen_d_rt[i] <- result$es
  combined_data_rt_diff$vi_rt[i] <- result$var
}

effect_size_rt <- combined_data_rt_diff %>%
  dplyr::select(ArticleID, Study, Group, Status_Practice, Practice_Number, 
                SnL_PresentTime, Response_Window, Sample_size, cohen_d_rt, vi_rt)

df2_rt <- metafor::escalc(
  measure = "SMD",
  data = effect_size_rt,
  yi = cohen_d_rt,
  vi = vi_rt,
  slab = paste("Study ID:", ArticleID, Study, Group)
) %>% 
  dplyr::arrange(cohen_d_rt) %>%
  dplyr::mutate(study_id = paste(ArticleID, Study, Group, sep = "_"))

# ACC Effect Sizes
combined_data_acc_diff$cohen_d_acc <- NA
combined_data_acc_diff$vi_acc <- NA

for (i in seq_len(nrow(combined_data_acc_diff))) {
  result <- esc::esc_mean_sd(
    grp1m = combined_data_acc_diff$ACC_mean_match_self[i],
    grp1sd = combined_data_acc_diff$ACC_sd_match_self[i],
    grp1n = combined_data_acc_diff$Sample_size[i],
    grp2m = combined_data_acc_diff$ACC_mean_match_stranger[i],
    grp2sd = combined_data_acc_diff$ACC_sd_match_stranger[i],
    grp2n = combined_data_acc_diff$Sample_size[i],
    es.type = "d"
  )
  combined_data_acc_diff$cohen_d_acc[i] <- result$es
  combined_data_acc_diff$vi_acc[i] <- result$var
}

effect_size_acc <- combined_data_acc_diff %>%
  dplyr::select(ArticleID, Study, Group, Status_Practice, Practice_Number, 
                SnL_PresentTime, Response_Window, Sample_size, cohen_d_acc, vi_acc)

df3_acc <- metafor::escalc(
  measure = "SMD",
  data = effect_size_acc,
  yi = cohen_d_acc,
  vi = vi_acc,
  slab = paste("Study ID:", ArticleID, Study, Group)
) %>%
  dplyr::arrange(cohen_d_acc) %>%
  dplyr::mutate(study_id = paste(ArticleID, Study, Group, sep = "_"))