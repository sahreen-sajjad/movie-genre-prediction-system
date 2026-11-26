# Movie Genre Prediction System

An NLP and machine learning project exploring whether movie genres can be predicted from dialogue using the Cornell Movie Dialogues Corpus (via ConvoKit).

## Overview

Six target genres: comedy, documentary, drama, horror, romantic comedy and thriller. The original study compared custom TF-IDF and K-means clustering, dummy baselines, logistic regression, SVM, random forest, grid search and RoBERTa experiments. The selected model was random forest with one-hot encoding of preprocessed dialogue strings.

## Historical results

The original final notebook reports the following on its test split (although its evaluation function prints “Validation”):

| Accuracy | Macro precision | Macro recall | Macro F1 |
|---:|---:|---:|---:|
| 88.6% | 95.2% | 73.7% | 81.3% |

These are archived notebook results, not independently reproduced. Joining by movie ID creates repeated dialogue rows, and row-level splitting can leak movie information across partitions. The default CLI instead holds out entire movies; its scores are not yet measured. Grouped splitting is not stratified and may omit rare genres from individual partitions. One-hot encoding whole dialogue strings also limits generalisation to unseen text.

## Setup

Use a Python 3.10 virtual environment. Dependency versions are not locked or verified end-to-end.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
python main.py
```

The corpus downloads on first use; the original movie-ID joins can require substantial RAM. To use the historical row-level split, run `python main.py --split original`. A fixed random forest seed was added, so exact historical scores are not guaranteed.

```bash
python main.py --predict "I love you."
python -m unittest discover -s tests
```

Training writes the model, metrics and confusion matrices to `results/`. Only load trusted model files.

## Repository layout

- `src/`: reusable data, preprocessing, model and evaluation modules.
- `main.py`: training and prediction CLI.
- `notebooks/`: original notebooks, including historical outputs.
- `experiments/`: Python exports of historical experiments; retained for reference, not verified runnable. Install optional dependencies from `requirements-experiments.txt` if exploring them.
- `tests/`: basic model and split checks.

## Dataset

Cornell Movie Dialogues Corpus, distributed through [ConvoKit](https://convokit.cornell.edu/documentation/movie.html). The dataset is not bundled in this repository.
