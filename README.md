# Data Mining Assignment 1

This repository contains code and analysis for Data Mining Assignment 1, focusing on various data mining techniques applied to different datasets.

## Repository Structure

- `data/`: Contains dataset files
  - `ODI-2025.csv`: Original dataset
  - `ODI-2025_cleaned.csv`: Cleaned dataset (output of Task 1b)
  - `ODI-2025_engineered.csv`: Dataset with engineered features (output of Task 1c)
  - `dataset_mood_smartphone.csv`: Additional dataset for future tasks

- `scripts/`: Contains Python scripts
  - `task1a_eda.py`: Exploratory data analysis script for Task 1a
  - `task1b_data_cleaning.py`: Data cleaning script for Task 1b
  - `task1c_feature_engineering.py`: Feature engineering script for Task 1c

- `task1a_outputs/`: Contains Task 1a visualizations and analysis reports
  - Various visualizations (PNG files)
  - `analysis_report.txt`: Statistical analysis summary
  - `README.md`: Specific information about Task 1a outputs

- `task1b_outputs/`: Contains Task 1b outputs and visualizations
  - `program_name_mapping.txt`: Mapping of original to normalized program names
  - `data_cleaning_report.txt`: Detailed cleaning methodology
  - Various comparison visualizations (PNG files)

- `task1c_outputs/`: Contains Task 1c visualizations and analysis
  - `feature_engineering_summary.txt`: Detailed description of all engineered features
  - `categorical_features_distribution.png`: Distribution of categorical features
  - `numeric_features_distribution.png`: Distribution of numeric features
  - `binary_features_distribution.png`: Distribution of binary features
  - `feature_correlations.png`: Correlation matrix of engineered features

- Documentation:
  - `requirements.txt`: Required Python packages
  - `analysis_report.md`: Comprehensive analysis of findings from all tasks
  - `README.md`: Repository structure and file information (this file)

## Setup and Execution

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Task 1a analysis script:
   ```
   python scripts/task1a_eda.py
   ```

3. Run the Task 1b data cleaning script:
   ```
   python scripts/task1b_data_cleaning.py
   ```

4. Run the Task 1c feature engineering script:
   ```
   python scripts/task1c_feature_engineering.py
   ```

5. View the generated visualizations in the respective output directories

## Task Overview

### Task 1a: Exploratory Data Analysis (EDA)

### Task 1b: Data Cleaning and Standardization

### Task 1c: Feature Engineering

## Future Tasks

Additional tasks will be added to this repository as they are completed:
- Task 2: TBD
- Task 3: TBD