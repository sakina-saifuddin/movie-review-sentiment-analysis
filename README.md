# Movie Review Sentiment Analysis

A from-scratch Naive Bayes classifier for positive and negative movie reviews, comparing four text preprocessing configurations. Built with Python, pandas, and spaCy for EDS 6352 Homework 1.

## Experiment

The source dataset contains 40,000 reviews. The notebook samples 1,000 training reviews without replacement, then selects 400 non-overlapping test reviews: 200 negative and 200 positive. All scenarios use the same split with random_state=42.

The classifier learns class priors and token likelihoods from training data, applies add-1 Laplace smoothing, and uses log probabilities for prediction. Test tokens outside the training vocabulary are ignored. Evaluation includes confusion matrices and per-class precision, recall, and F1.

## Results

| Scenario | Preprocessing | Negative F1 | Positive F1 | Accuracy | Correct / 400 |
|---|---|---:|---:|---:|---:|
| 1 | Baseline | 81.03% | 78.28% | 79.75% | 319 |
| 2 | Lemmatization | 80.57% | 78.31% | 79.50% | 318 |
| 3 | Lemmatization and stop-word removal | 81.82% | 80.10% | 81.00% | 324 |
| 4 | Lemmatization, stop-word removal, and negation handling | 81.36% | 80.10% | 80.75% | 323 |

Scenario 3 achieved the highest observed accuracy on this test set. Its advantage over Scenario 4 is one net correct prediction. Statistical significance was not tested.

## Files

- `Saifuddin_HW1_NLP.ipynb`: executed notebook with preprocessing, training, evaluation, and misclassified examples.

## Run in Google Colab

1. Open the notebook in Google Colab.
2. Obtain the assignment's `movie.csv`, containing `text` and `label` columns (0 = negative, 1 = positive). The dataset is not included in this repository.
3. Place the CSV in Google Drive under `My Drive/Colab Notebooks/NLP`, or update `data_folder` in the first code cell to its folder.
4. Run the notebook from top to bottom and authorize the Drive mount. The notebook installs spaCy and downloads `en_core_web_sm`.

The saved results used spaCy's `en_core_web_sm` model version 3.8.0. Changes to the data or NLP dependencies may affect reproducibility.

## Limitations

The classifier uses individual token counts and does not fully capture sentence meaning or word order. The negation heuristic marks words after a negation cue until punctuation; complex clauses can be mishandled. The notebook inspects examples of both positive and negative reviews classified incorrectly.
