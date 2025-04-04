Here is a README template you can add to your repository for this project:

---

# Flood Susceptibility Mapping Using Machine Learning and Morphometric Analysis

## Overview
This project integrates **Machine Learning (ML)** techniques and **morphometric analysis** to assess flood susceptibility across four major catchments in Bayelsa State, Nigeria. The study leverages **Shuttle Radar Topographic Mission (SRTM)** data to extract hydrological features and applies machine learning classifiers (Random Forest, Support Vector Machine, XGBoost) to classify flood-prone zones. The goal is to enhance flood prediction accuracy and granularity beyond traditional morphometric methods.

## Dataset
The dataset used for this analysis includes key morphometric features, such as:
- Drainage Density (Dd)
- Stream Frequency (Fs)
- Relief Ratio (Rh)
- Bifurcation Ratio (Rbm)
- Infiltration Number (If)

The flood-prone labels are generated from historical flood records, with each catchment marked as either **flood-prone (1)** or **not flood-prone (0)**.

You can access the dataset template for training in `morphometric_ml_dataset.csv`.

## Requirements
Ensure you have the following Python packages installed:
- `pandas`
- `numpy`
- `sklearn`
- `matplotlib`
- `seaborn`
- `xgboost`

You can install them via `pip`:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## File Structure

```
├── flood_susceptibility_model.py    # Python script for model training and evaluation
├── morphometric_ml_dataset.csv     # Dataset template for morphometric features
├── ROC_Comparison.png              # ROC Curve comparison figure
├── XGBoost_Confusion_Matrix.png    # Confusion matrix for XGBoost model
├── XGBoost_Feature_Importance.png  # Feature importance plot for XGBoost model
└── README.md                       # Project overview and instructions
```

## Usage

### 1. Prepare the Data
Load your dataset in `morphometric_ml_dataset.csv`, ensuring the dataset includes morphometric features (Dd, Fs, Rh, Rbm, If) and a binary flood-prone label.

### 2. Model Training
Run the Python script `flood_susceptibility_model.py` to:
- Train three ML models (Random Forest, SVM, XGBoost) on the dataset.
- Evaluate the models using accuracy, precision, recall, F1-score, and ROC-AUC metrics.
- Generate and save confusion matrices, ROC curves, and feature importance plots.

```bash
python flood_susceptibility_model.py
```

### 3. Results
The script will generate the following figures:
- **ROC Comparison**: Shows the ROC curves for all classifiers.
- **Confusion Matrices**: Displays the confusion matrix for each classifier.
- **Feature Importance**: Plots the feature importance scores from the XGBoost model.
- **Flood Susceptibility Map**: Simulated flood-prone areas for visualization.

## Author Contributions


## Acknowledgements
We would like to thank the **Nigerian Hydrological Services Agency (NHSA)** for providing historical flood data, and the **United States Geological Survey (USGS)** for the SRTM data used in this study.

---

Feel free to modify or add any additional details to this README as needed!
