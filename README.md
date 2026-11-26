# Elevate Labs ML Internship: Content Recommender & Tabular Preprocessing

Completed during a remote Machine Learning Internship at **Elevate Labs** (January 2025 - July 2025, Skill India / MSME Best Performer Award).

## Components

1. **Netflix Content-Based Recommender (`netflix_recommender.py`, dataset: `netflix.csv`)**
   - Loads **`5,967` catalog titles (`13` columns)** from tab-separated [`netflix.csv`](netflix.csv).
   - Cleans text (punctuation stripping, English stopword removal, Snowball stemming), builds TF-IDF features (`ngram_range=(1, 2)`, `max_features=8000`) over `Genres` and `Description`, computes pairwise cosine similarity, and ranks top-10 matches descending (excluding the query title itself).
2. **Tabular Data Cleaning Workflow (`clean_titanic.py`)**
   - Imputes missing numeric `Age` with column median and categorical `Embarked` with mode, removes sparse `Cabin`/ID columns, applies one-hot encoding, and standardizes numeric features using `StandardScaler`.

## Quickstart & Verification

```bash
pip install pandas scikit-learn nltk pytest
pytest -v
python netflix_recommender.py --title "Sacred Games" --top-k 5
```

## Notes on Original Fixes

- Corrected separator handling (`sep="\t"`) and filename reference (`netflix.csv` vs earlier `netflixData.csv`).
- Fixed similarity ranking to sort descending cosine scores (`scores.sort(..., reverse=True)`) instead of slicing unsorted matrix rows `0..9`.
