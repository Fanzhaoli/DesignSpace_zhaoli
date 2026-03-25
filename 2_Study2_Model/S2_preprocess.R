# ==============================================================================
# S2_preprocess.R - Data Preprocessing for Study 2
# ==============================================================================
# Purpose: Process raw simulation data into analysis-ready format
# Output: df, df_sum, df_sum_cond, df_sum_cond_w (with Cohen's d)
# ==============================================================================

# Load packages
if (!requireNamespace('pacman', quietly = TRUE)) {
    install.packages('pacman')
}

pacman::p_load(
  here, tidyverse, bruceR
)

# Disable scientific notation
options(scipen = 9999)

# ------------------------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------------------------
# Import raw data (match condition)
# response = 1 (correct), = 2 (error)
df <- bruceR::import(here::here("Study2", "data", "260216.csv")) %>%
  dplyr::mutate(
    RT = RT * 1000,
    correct = ifelse(response == 2, 0, 1)
  ) %>%
  dplyr::rename(P_val = P, T_val = `T`, W_val = W)

# ------------------------------------------------------------------------------
# 2. Overall Summary (by subject and design parameters)
# ------------------------------------------------------------------------------
df_sum <- df %>%
  dplyr::group_by(subjectID, P_val, T_val, W_val) %>%
  dplyr::summarise(
    mean_RT = mean(RT),
    sd_RT = sd(RT),
    mean_ACC = mean(response == 1),
    sd_ACC = sd(response == 1),
    a = mean(a),
    v = mean(v),
    n = dplyr::n(),
    .groups = "drop"
  )

# ------------------------------------------------------------------------------
# 3. Condition-Level Summary (by subject, label, and design parameters)
# ------------------------------------------------------------------------------
df_sum_cond <- df %>%
  dplyr::group_by(subjectID, Label, P_val, T_val, W_val) %>%
  dplyr::summarise(
    mean_RT = mean(RT),
    sd_RT = sd(RT),
    mean_ACC = mean(response == 1),
    sd_ACC = sd(response == 1),
    a = mean(a),
    v = mean(v),
    n = dplyr::n(),
    .groups = "drop"
  )

# ------------------------------------------------------------------------------
# 4. Pivot to wide format for self vs stranger comparison
# ------------------------------------------------------------------------------
df_sum_cond_w <- df_sum_cond %>%
  tidyr::pivot_wider(
    id_cols = c(subjectID, P_val, T_val, W_val),
    names_from = Label,
    values_from = c(mean_RT, sd_RT, mean_ACC, sd_ACC, n, a, v)
  )

# ------------------------------------------------------------------------------
# 5. Calculate Cohen's d (Self vs Stranger)
# ------------------------------------------------------------------------------
calc_cohens_d_vec <- function(mean_self, mean_stranger, 
                               sd_self, sd_stranger, 
                               n_self, n_stranger,
                               direction = c("stranger_minus_self", "self_minus_stranger")) {
  direction <- match.arg(direction)
  
  pooled_sd <- sqrt(((n_self - 1) * sd_self^2 + (n_stranger - 1) * sd_stranger^2) / 
                    (n_self + n_stranger - 2))
  
  diff <- if (direction == "stranger_minus_self") {
    mean_stranger - mean_self
  } else {
    mean_self - mean_stranger
  }
  
  dplyr::if_else(is.na(pooled_sd) | pooled_sd == 0, diff, diff / pooled_sd)
}

df_sum_cond_w <- df_sum_cond_w %>%
  dplyr::mutate(
    RT_diff = mean_RT_stranger - mean_RT_self,
    ACC_diff = mean_ACC_self - mean_ACC_stranger,
    cohen_d_RT = calc_cohens_d_vec(
      mean_self = mean_RT_self, mean_stranger = mean_RT_stranger,
      sd_self = sd_RT_self, sd_stranger = sd_RT_stranger,
      n_self = n_self, n_stranger = n_stranger,
      direction = "stranger_minus_self"
    ),
    cohen_d_ACC = calc_cohens_d_vec(
      mean_self = mean_ACC_self, mean_stranger = mean_ACC_stranger,
      sd_self = sd_ACC_self, sd_stranger = sd_ACC_stranger,
      n_self = n_self, n_stranger = n_stranger,
      direction = "self_minus_stranger"
    )
  )

# ------------------------------------------------------------------------------
# 6. Data Completeness Check
# ------------------------------------------------------------------------------
n_complete <- nrow(df_sum)
n_unique_P <- length(unique(df_sum$P_val))
n_unique_T <- length(unique(df_sum$T_val))
n_unique_W <- length(unique(df_sum$W_val))
n_possible <- n_unique_P * n_unique_T * n_unique_W

cat("=== Study 2 Data Completeness Check ===\n")
cat("Actual observations:", n_complete, "\n")
cat("Unique P values:", n_unique_P, "\n")
cat("Unique T values:", n_unique_T, "\n")
cat("Unique W values:", n_unique_W, "\n")
cat("Possible combinations:", n_possible, "\n")
cat("Missing combinations:", n_possible - n_complete, "\n")
cat("Data completeness:", round(n_complete/n_possible * 100, 5), "%\n")

# Return processed data frames
# These will be available in the global environment when sourced
message("S2_preprocess.R complete: df, df_sum, df_sum_cond, df_sum_cond_w created")
