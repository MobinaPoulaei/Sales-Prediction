# Sales Prediction Model

This repository contains the Python script for training a sales prediction model based on historical sales data from retail stores. The script merges sales data with store attributes and uses a Random Forest Regressor to predict future sales.

## Overview

The script processes the input CSV files containing sales data and store information, merges them, and performs feature engineering by extracting date parts and mapping categorical variables to human-readable values. It prepares the data for machine learning by handling missing values and converting categorical variables into dummy/indicator variables. Finally, it trains a Random Forest Regressor model and evaluates its performance.

## Files

- `salesprediction.py`: The main Python script for training and evaluating the sales prediction model.
- `train.csv`: Training data containing historical sales information.
- `store.csv`: Store data containing attributes of each store.

## Setup

### Prerequisites

- Python 3.8 or newer
- pandas
- numpy
- scikit-learn

Ensure you have the above Python libraries installed. You can install them using pip:

```bash
pip install pandas numpy scikit-learn
```

### Running the Script

1. Place your `train.csv` and `store.csv` in the same directory as the script.
2. Run the script using the following command:

```bash
python salesprediction.py
```

The script will train the model and save it as `random_forest_model.pkl` in the same directory.

## Model

The script uses a `RandomForestRegressor` from scikit-learn with 42 estimators. The choice of 42 estimators is arbitrary and can be tuned based on model performance and computational constraints. The model predicts the `Sales` variable based on various features extracted from the data.

## Output

- `random_forest_model.pkl`: A serialized Python object containing the trained Random Forest model. This file can be loaded to make predictions on new data.
