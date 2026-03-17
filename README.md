# Design Space for Self-Prioritization Effect

## Authors
- Zhaoli Fan
- Jiahui Wen
- Hu Chuan-Peng

## License

[![Creative Commons Attribution-NonCommercial 4.0 International](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

## Aims
We propose a design space model for self-prioritization effect. We examines how experimental design factors such as length of practices (P), preseting time (T), and time window for response (W) determine the cognitive processes (as indicated by the DDM parameters such as drift rate *v* and boundary separation *a*) and therefore modulate the effect size of the self-prioritization as measured by the self-matching task [(Sui et al, 2012)](http://www.ncbi.nlm.nih.gov/pubmed/22963229). This work is part of Jiahui Wen's master thesis. 

## Folder Structure

### Study 1: Meta-Analysis
- **Purpose**: Meta-analysis of dimensional categorization paradigms
- **Data**: `Study 1/data/dimension.xlsx` - contains published studies with 38 articles (114 groups)
- **Output**: `Study 1/S1 meta.Rmd`

### Study 2: Model Simulation
- **Purpose**: Simulation and analysis using the Wiener diffusion model
- **Data**: `Study 2/data/250430.csv` - simulated response time data
- **Analysis**: `Study 2/S2 ana.Rmd`
- **Additional**: `Study 2/S2 sim & ana_rwiener.Rmd` for Wiener model fitting

### Study 3: Empirical Analysis
- **Purpose**: Analysis of experimental data from multiple participant groups
- **Data**: 
  - `Study 3/data/emp data/` - experimental data files (groups 1-6)
  - `Study 3/data/fast dm/` - DDM parameter estimates
  - `Study 3/data/sim_data/` - simulated data
- **Output**: `Study 3/S3 ana.Rmd`

### Study 4: Extended Empirical Analysis
- **Purpose**: Extended analysis with additional participant groups
- **Data**:
  - `Study 4/data/emp_data/` - experimental data (groups 1-9)
  - `Study 4/data/fast dm/` - DDM parameter estimates
  - `Study 4/data/sim_data/` - simulated data
- **Output**: `Study 4/S4 ana.Rmd`

## Dependencies

### Data Format
Typical columns used across the analysis pipelines:
- `groupID`: 1-7
- `subjectID`: integer subject index
- `Label`: "self", "friend", "stranger"
- `Match`: "match", "mismatch"
- `CorrectKey`: f/j
- `Correct`: integer (0/1) accuracy
- `RT`: response time in seconds
- `trialID`, `P`, `T`, `W`, `v`, `a`: additional trial-level metadata

The analyses use R with the following key packages:
- `tidyverse` - data manipulation
- `bruceR` - statistical analysis
- `lme4` - mixed effects models
- `ggplot2`, `plotly` - visualization
- `RWiener` - Wiener diffusion model

## Reports

- `Reports/Self-bias_Comp_Models.pptx` - Presentation on self-bias and computational models
