# Experiment 3: Logistic Regression with MLE and MAP

This experiment classifies breast-cancer samples using the Wisconsin Breast Cancer
dataset. It compares maximum likelihood estimation (MLE) with maximum a posteriori
(MAP) estimation using L1 and L2 regularization.

## What It Does

1. Loads the breast cancer dataset from scikit-learn.
2. Splits the data into stratified training and test sets.
3. Standardizes the features using statistics learned from the training data.
4. Trains three logistic-regression models:
   - MLE with no penalty
   - MAP with L2 regularization
   - MAP with L1 regularization
5. Evaluates each model using accuracy, precision, recall, and F1 score.
6. Prints a confusion matrix and the number of zero coefficients for each model.
7. Displays a metric comparison bar chart.
8. Plots ROC curves and AUC values for all three models.

## Dataset and Split

The dataset contains 569 samples and 30 features. The script uses an 80/20 train/test
split with `random_state=42` and preserves class proportions with stratification.
`StandardScaler` is fitted only on the training data and then used to transform both
training and test features.

## Model Settings

| Model | Regularization | Solver | `C` |
| --- | --- | --- | --- |
| MLE | None | scikit-learn default | Not applicable |
| MAP (L2) | L2 / ridge | `lbfgs` | `1.0` |
| MAP (L1) | L1 / lasso | `liblinear` | `1.0` |

All models use `max_iter=5000`. In this implementation, the regularized models serve
as MAP estimates under their corresponding priors. A smaller `C` means stronger
regularization.

## Evaluation

Accuracy, precision, recall, and F1 score are printed for each model and collected in
a comparison table. The confusion matrices show the counts of correct and incorrect
predictions for each class. The ROC plot uses predicted probabilities and reports the
area under the curve (AUC) for each model.

The script also counts coefficients equal to zero. L1 regularization can produce sparse
models with zero coefficients, while L2 regularization generally shrinks coefficients
without making them exactly zero.

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Run

From the `Experiment3` directory, run:

```bash
python exp3.py
```

Two plots are displayed: the metric comparison bar chart and the ROC curve comparison.
