# Experiment 2: Polynomial Regression

This experiment is intended to use the Auto MPG dataset to predict miles per gallon
(`MPG`) from engine displacement. It is designed to compare polynomial regression at
different degrees with ordinary linear regression.

## Intended Tasks

1. Load and preprocess the Auto MPG dataset.
2. Use engine displacement as the predictor and MPG as the target.
3. Create polynomial features for several degrees.
4. Fit polynomial and linear regression models.
5. Compare the models using mean squared error (MSE) and R2 score.
6. Visualize the fitted polynomial curves.

## Current Implementation

The current `exp2.py` and `exp2.ipynb` files contain the initial imports for:

- NumPy and pandas for data handling
- Matplotlib for visualization
- `fetch_openml` for loading the dataset
- `train_test_split` for creating train/test data
- `LinearRegression` and `PolynomialFeatures` for modeling
- MSE and R2 metrics for evaluation

The model-training and visualization steps still need to be added before the
experiment can produce results.

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Run

From the `Experiment2` directory, run:

```bash
python exp2.py
```

Alternatively, open `exp2.ipynb` and execute its cells after completing the
implementation. Loading the Auto MPG dataset may require an internet connection.
