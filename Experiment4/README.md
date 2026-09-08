# Experiment 4: Naive Bayes Text Classification

This experiment uses the scikit-learn 20 Newsgroups dataset to classify text documents
into discussion-group topics. It compares two Naive Bayes models:

- Multinomial Naive Bayes
- Bernoulli Naive Bayes

## What It Does

1. Loads the predefined training and test splits of the 20 Newsgroups dataset.
2. Removes message headers, footers, and quoted text to reduce noise.
3. Converts the documents into numerical features with `CountVectorizer`.
4. Trains both Naive Bayes classifiers.
5. Evaluates each model using accuracy, macro precision, macro recall, and macro F1 score.
6. Displays a bar chart comparing the models across all four metrics.

## Feature Preparation

Both vectorizers use English stop-word removal and ignore terms that occur in fewer
than two documents (`min_df=2`). The models use different feature representations:

| Model | Feature representation |
| --- | --- |
| `MultinomialNB` | Word occurrence counts |
| `BernoulliNB` | Binary word presence (`0` or `1`) |

The vocabulary is learned from the training data and then applied to the test data.
This prevents information from the test set being used during training.

## Evaluation

The script calculates the following metrics for both models:

- **Accuracy:** Overall proportion of correctly classified documents.
- **Macro precision:** Precision calculated independently for each topic and then averaged.
- **Macro recall:** Recall calculated independently for each topic and then averaged.
- **Macro F1 score:** Harmonic mean of precision and recall, averaged equally across topics.

The script also prints the number of training documents, test documents, topic classes,
features, and training samples. The final comparison chart is displayed after execution.

## Dataset Notes

The dataset contains text documents grouped into 20 newsgroup categories. The dataset
is downloaded automatically by scikit-learn if it is not already available locally.

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Run

From the `Experiment4` directory, run:

```bash
python exp4.py
```

The 20 Newsgroups dataset may be downloaded automatically the first time the script runs.
