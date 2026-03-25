# ==============================================================================
# S1_generate.R - Generate Results for Study 1
# ==============================================================================
# Purpose: Generate figures, tables, and statistics for manuscript
# Input: df2_rt, df3_acc from S1_preprocess.R
# Output: figure_2, figure_3, table1_data, stats
# ==============================================================================

# Source preprocessing if not already done
if (!exists("df2_rt")) {
  source(here::here("Study1", "S1_preprocess.R"))
}

# Load packages for visualization
pacman::p_load(ggplot2, ggExtra, papaja, cowplot, e1071, nortest, boot, reticulate,
               ggside, patchwork, tidyverse, plotly, viridisLite, magick, ggplotify)

# Load utility functions (for describe_effects and analyze_centrality)
source(here::here("analysis", "utils.R"))

# Disable scientific notation
options(scipen = 9999, digits = 5)
options(warn = -1)

# ------------------------------------------------------------------------------
# 1. Statistical Analysis
# ------------------------------------------------------------------------------

# Descriptive Statistics
rt_stats <- describe_effects(df2_rt$cohen_d_rt, "RT")
acc_stats <- describe_effects(df3_acc$cohen_d_acc, "ACC")

# Bootstrap Confidence Intervals
rt_central <- analyze_centrality(df2_rt$cohen_d_rt)
acc_central <- analyze_centrality(df3_acc$cohen_d_acc)

# ------------------------------------------------------------------------------
# 2. Figure 2: Marginal Distribution Plots
#    This plot should have four subplots: 
#    A, upper-left, a screenshot of a 3D plot of cohen's d for RT, 
#    B, upper-right,point + marginal plot of cohen's d for RT,
#    C, lower-left, a screenshot of a 3D plot of cohen's d for ACC,
#    D, lower-right,point + marginal plot of cohen's d for ACC.
# ------------------------------------------------------------------------------

# RT plot
fig2b <- ggplot2::ggplot(df2_rt, ggplot2::aes(x = Practice_Number, y = Response_Window, 
                              size = cohen_d_rt, color = cohen_d_rt)) +
  ggplot2::geom_point(alpha = 0.15) +
  ggplot2::scale_size_continuous(range = c(2, 8), name = "Cohen's d RT") +
  ggplot2::scale_color_viridis_c(guide = "none") +
  ggplot2::labs(x = "Practice Number", y = "Response Window") +
  ggplot2::guides(size = ggplot2::guide_legend(order = 1)) +
  papaja::theme_apa() +
  ggplot2::theme(legend.position = "right",
        ggside.axis.line.y = ggplot2::element_blank(),
        ggside.axis.text.y = ggplot2::element_blank(),
        ggside.axis.title.y = ggplot2::element_blank(),
        ggside.axis.ticks.y = ggplot2::element_blank(),
        ggside.axis.line.x = ggplot2::element_blank(),
        ggside.axis.text.x = ggplot2::element_blank(),
        ggside.axis.title.x = ggplot2::element_blank(),
        ggside.axis.ticks.x = ggplot2::element_blank()) +
  ggside::geom_xsidehistogram(fill = "gray", color = "black", alpha = 0.5, bins = 30) +
  ggside::geom_ysidehistogram(fill = "gray", color = "black", alpha = 0.5, bins = 30) +
  ggplot2::theme(ggside.panel.scale = 0.2) 

# ACC plot
fig2d <- ggplot2::ggplot(df3_acc, ggplot2::aes(x = Practice_Number, y = Response_Window, 
                                size = cohen_d_acc, color = cohen_d_acc)) +
  ggplot2::geom_point(alpha = 0.15) +
  ggplot2::scale_size_continuous(range = c(2, 8), name = "Cohen's d ACC") +
  ggplot2::scale_color_viridis_c(guide = "none") +
  ggplot2::labs(x = "Practice Number (P)", y = "Response Window (W)") +
  ggplot2::guides(size = ggplot2::guide_legend(order = 1)) +
  papaja::theme_apa() +
  ggplot2::theme(legend.position = "right",
        ggside.axis.line.y = ggplot2::element_blank(),
        ggside.axis.text.y = ggplot2::element_blank(),
        ggside.axis.title.y = ggplot2::element_blank(),
        ggside.axis.ticks.y = ggplot2::element_blank(),
        ggside.axis.line.x = ggplot2::element_blank(),
        ggside.axis.text.x = ggplot2::element_blank(),
        ggside.axis.title.x = ggplot2::element_blank(),
        ggside.axis.ticks.x = ggplot2::element_blank()) +
  ggside::geom_xsidehistogram(fill = "gray", color = "black", alpha = 0.5, bins = 30) +
  ggside::geom_ysidehistogram(fill = "gray", color = "black", alpha = 0.5, bins = 30) +
  ggplot2::theme(ggside.panel.scale = 0.2)


# ------------------------------------------------------------------------------
# 2A. Figure 2A: 3D Scatter Plot for RT Cohen's d
# ------------------------------------------------------------------------------
# set up plotly export
# Export 3D ACC plot as static image
# requires kaleido for static image export, which can be installed via reticulate
reticulate::py_require(c("plotly",'kaleido'))

# requires chrome or chromium for kaleido to work, should list the right path.
Sys.setenv(CHROME_BIN = "/Applications/Google Chrome")

# Define ranges for RT 3D plot
rt_x_range <- base::range(df2_rt$Response_Window)  # x-axis: Response Window
rt_y_range <- base::range(df2_rt$Practice_Number)  # y-axis: Practice Number
rt_z_range <- base::mean(df2_rt$SnL_PresentTime)  # z-axis: Stimuli Present Time (mean)

fig_rt_3d <- plotly::plot_ly() %>%
  # Add gray plane at mean T
  plotly::add_trace(
    x = c(rt_x_range[1], rt_x_range[1], rt_x_range[2], rt_x_range[2]),
    y = c(rt_y_range[1], rt_y_range[2], rt_y_range[1], rt_y_range[2]),
    z = c(rt_z_range, rt_z_range, rt_z_range, rt_z_range),
    type = "mesh3d",
    intensity = c(1, 1, 1, 1),
    colorscale = list(c(0, "grey"), c(1, "grey")),
    opacity = 0.3,
    showscale = FALSE
  ) %>%
  # 3D scatter: colored by Cohen's d RT
  plotly::add_trace(
    data = df2_rt,
    x = ~Response_Window,
    y = ~Practice_Number,
    z = ~SnL_PresentTime,
    type = "scatter3d",
    mode = "markers",
    marker = list(
      size = 5,
      color = ~cohen_d_rt,
      colorscale = "Viridis",
      opacity = 0.8
    )
  ) %>%
  plotly::layout(
    scene = list(
      zaxis = list(
        range = c(0, 200),
        dtick = 50,
        title = "Stimuli Present Time (T)"
      ),
      yaxis = list(title = "Practice Number (P)"),
      xaxis = list(title = "Response Window (W)")
    )
  ) %>%
  plotly::layout(scene = list(camera = list(eye = list(x = 1.9, y = 1.9, z = 1.9))),
                 margin = list(l = 20, r = 20, t = 20, b = 20))

# Export 3D RT plot as static image
plotly::save_image(fig_rt_3d, file = here::here("Study1", "output", "figure_2a_rt_3d.png"), 
                   width = 800, height = 600, scale = 2)

# Create a placeholder for embedding (will be rendered as HTML in Rmd)
fig_rt_3d_file <- here::here("Study1", "output", "figure_2a_rt_3d.png")

# ------------------------------------------------------------------------------
# 2C. Figure 2C: 3D Scatter Plot for ACC Cohen's d
# ------------------------------------------------------------------------------
# Define ranges for ACC 3D plot
acc_x_range <- base::range(df3_acc$Response_Window)  # x-axis: Response Window
acc_y_range <- base::range(df3_acc$Practice_Number)  # y-axis: Practice Number
acc_z_range <- base::mean(df3_acc$SnL_PresentTime)  # z-axis: Stimuli Present Time (mean)

fig_acc_3d <- plotly::plot_ly() %>%
  # Add gray plane at mean T
  plotly::add_trace(
    x = c(acc_x_range[1], acc_x_range[1], acc_x_range[2], acc_x_range[2]),
    y = c(acc_y_range[1], acc_y_range[2], acc_y_range[1], acc_y_range[2]),
    z = c(acc_z_range, acc_z_range, acc_z_range, acc_z_range),
    type = "mesh3d",
    intensity = c(1, 1, 1, 1),
    colorscale = list(c(0, "grey"), c(1, "grey")),
    opacity = 0.3,
    showscale = FALSE
  ) %>%
  # 3D scatter: colored by Cohen's d ACC
  plotly::add_trace(
    data = df3_acc,
    x = ~Response_Window,
    y = ~Practice_Number,
    z = ~SnL_PresentTime,
    type = "scatter3d",
    mode = "markers",
    marker = list(
      size = 5,
      color = ~cohen_d_acc,
      colorscale = "Viridis",
      opacity = 0.8
    )
  ) %>%
  plotly::layout(
    scene = list(
      zaxis = list(
        range = c(0, 200),
        dtick = 50,
        title = "Stimuli Present Time (T)"
      ),
      yaxis = list(title = "Practice Number (P)"),
      xaxis = list(title = "Response Window (W)")
    )
  ) %>%
  plotly::layout(scene = list(camera = list(eye = list(x = 1.9, y = 1.9, z = 1.9))),
                 margin = list(l = 20, r = 20, t = 20, b = 20))

plotly::save_image(fig_acc_3d, file = here::here("Study1", "output", "figure_2c_acc_3d.png"), 
                   width = 800, height = 600, scale = 2)

# Create a placeholder for embedding
fig_acc_3d_file <- here::here("Study1", "output", "figure_2c_acc_3d.png")

# ------------------------------------------------------------------------------
# Combine all 4 subplots into single Figure 2 using patchwork
# ------------------------------------------------------------------------------
# Note: For 3D plots, we'll include them as static images since patchwork
# doesn't support plotly objects directly. Load the exported PNGs.

# Read 3D plot images
# Need to crop the images to remove extra whitespace for better layout
rt_3d_img <- magick::image_read(here::here("Study1", "output", "figure_2a_rt_3d.png")) 
# magick::image_info(rt_3d_img)
rt_3d_img <- magick::image_crop(rt_3d_img, "900x800+220+250")

acc_3d_img <- magick::image_read(here::here("Study1", "output", "figure_2c_acc_3d.png"))
# magick::image_info(acc_3d_img)
acc_3d_img <- magick::image_crop(acc_3d_img, "900x800+220+250")

# Convert to grid for patchwork
fig2a <- ggplotify::as.ggplot(~plot(rt_3d_img))
fig2c <- ggplotify::as.ggplot(~plot(acc_3d_img))

# Combine: A (top-left), B (top-right), C (bottom-left), D (bottom-right)
# Using patchwork layout: 2 columns, 2 rows
figure_2_full <- (
  # Row 1: A (3D RT) | B (2D RT)
  (fig2a | fig2b) /
  # Row 2: C (3D ACC) | D (2D ACC)
  (fig2c | fig2d)
) +
  patchwork::plot_annotation(
    tag_levels = "A",
    tag_suffix = ""
  ) &
  ggplot2::theme(
    plot.tag = ggplot2::element_text(size = 14, face = "bold")
  )

# ------------------------------------------------------------------------------
# 3. Figure 3: Effect Size Distributions
# ------------------------------------------------------------------------------

figure_3 <- ggplot2::ggplot() +
  ggplot2::geom_density(data = df2_rt, ggplot2::aes(x = cohen_d_rt, fill = "RT"), alpha = 0.3) +
  ggplot2::geom_density(data = df3_acc, ggplot2::aes(x = cohen_d_acc, fill = "ACC"), alpha = 0.3) +
  ggplot2::labs(x = "Cohen's d", y = "Density", fill = "Measure") +
  papaja::theme_apa() +
  ggplot2::scale_fill_manual(values = c("RT" = "darkred", "ACC" = "darkblue"))

# ------------------------------------------------------------------------------
# 4. Table 1: Study Information
# ------------------------------------------------------------------------------

# intersect(colnames(combined_data_rt_diff), colnames(combined_data_acc_diff))

table1_data <- combined_data_rt_diff %>%
  dplyr::full_join(combined_data_acc_diff, by = intersect(colnames(combined_data_rt_diff), colnames(combined_data_acc_diff)),
                   ) %>%
  dplyr::select(ArticleID, Study, Sample_size, 
                Practice_Number, SnL_PresentTime, Response_Window,
                cohen_d_rt, cohen_d_acc) %>%
  dplyr::mutate(cohen_d_rt = round(cohen_d_rt, 2),
                cohen_d_acc = round(cohen_d_acc, 2),
                Sample_size = as.integer(Sample_size),
                Practice_Number = as.integer(Practice_Number),
                SnL_PresentTime = as.integer(SnL_PresentTime),
                Response_Window = as.integer(Response_Window)) %>%
  dplyr::rename(dRT = cohen_d_rt, dACC = cohen_d_acc,
                `\\textit{N}` = Sample_size,
                `\\textit{P}` = Practice_Number, 
                `\\textit{T}` = SnL_PresentTime, 
                `\\textit{W}` = Response_Window)
  