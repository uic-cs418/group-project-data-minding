# AQI Prediction Project

## Overview
This project analyzes and models Air Quality Index (AQI) data across 24 U.S. states using historical data from 2023 to predict AQI in 2024. We explore both linear and non-linear machine learning models to understand pollutant and weather impacts on air quality.

## Goals
- Predict 2024 AQI using 2023 state-level averages.
- Compare model performance: Baseline, Linear Regression, Random Forest.
- Extract insights from feature importance.
- Extend the model to daily, county-level data and AQI category classification.

## Data
- **Source**: U.S. EPA AQI and meteorological datasets
- **Files Used**:
  - `24StateAQI_2023.csv`
  - `24StateAQI_2024.csv`
  - `daily_aqi_by_county_2023.csv`
  - `daily_aqi_by_county_2024.csv`

## How to Run
1. Load and merge datasets (unzip first)
2. Preprocess and aggregate AQI by state.
3. Train and evaluate models.
4. Visualize results and feature importance.

## Requirements
- Python 3.8+
- pandas, scikit-learn, matplotlib, seaborn

---

*Author: [Your Name]*  
*Institution: [Your University or Course Info]*  
