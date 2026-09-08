# Experiment 1: Linear Regression

This experiment predicts California housing prices using the average number of rooms
(`AveRooms`) as the single input feature. It compares two ways to estimate a linear
regression model:

- Gradient descent
- The normal equation

## What It Does

1. Loads the California Housing dataset from scikit-learn.
2. Creates a DataFrame containing the housing features and target price.
3. Selects `AveRooms` as the input and `price` as the target.
4. Splits the data into training and test sets using an 80/20 split.
5. Standardizes the feature for gradient descent.
6. Trains the model manually with gradient descent.
7. Calculates test MSE and R2 score for the gradient-descent model.
8. Estimates parameters with the normal equation and plots its regression line.

## Gradient Descent Settings

- Learning rate: `0.01`
- Epochs: `1000`
- Initial weight: `0`
- Initial bias: `0`
- Random state for the split: `42`

The script prints the cost every 100 epochs, followed by the learned weight, bias,
mean squared error, and R2 score.

## Normal Equation

The normal equation is applied to the unscaled training feature after adding a column
of ones for the intercept. The resulting parameters are printed as `Theta 0` and
`Theta 1`, and the fitted line is displayed in a Matplotlib plot.

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Run

From the `Experiment1` directory, run:

```bash
python exp1.py
```

The California Housing dataset may be downloaded automatically on the first run.
