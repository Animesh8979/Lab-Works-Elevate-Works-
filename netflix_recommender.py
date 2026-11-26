"""Content-based Netflix catalog recommender using NLTK normalization, TF-IDF, and cosine similarity."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import string
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.stem import SnowballStemmer
    _STEMMER = SnowballStemmer("english")
except Exception:
    _STEMMER = None


def clean_text(raw: Any) -> str:
    txt = str(raw or "").lower()
    txt = re.sub(r"\[.*?\]|https?://\S+|www\.\S+|<.*?>+", " ", txt)
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in txt.split() if w and w not in ENGLISH_STOP_WORDS and not any(ch.isdigit() for ch in w)]
    if _STEMMER is not None:
        tokens = [_STEMMER.stem(w) for w in tokens]
    return " ".join(tokens)


def load_netflix_catalog(csv_path: str | Path | None = None) -> pd.DataFrame:
    p = Path(csv_path) if csv_path else Path(__file__).resolve().parent / "netflix.csv"
    df = pd.read_csv(p, sep="\t")
    required = df[["Title", "Description", "Content Type", "Genres"]].copy()
    required = required.dropna(subset=["Title", "Genres"]).reset_index(drop=True)
    required["clean_title"] = required["Title"].astype(str).apply(clean_text)
    required["title_key"] = required["Title"].astype(str).str.strip().str.lower()
    required["soup"] = (
        required["Genres"].fillna("") + " " + required["Description"].fillna("")
    ).apply(clean_text)
    return required


class NetflixRecommender:
    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.df = load_netflix_catalog(csv_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["soup"])
        self.similarity = cosine_similarity(self.tfidf_matrix)
        self._exact_index = {k: i for i, k in enumerate(self.df["title_key"]) if k}
        self._clean_index = {k: i for i, k in enumerate(self.df["clean_title"]) if k}

    def recommend(self, title: str, top_k: int = 10) -> list[dict[str, Any]]:
        key = str(title or "").strip().lower()
        idx = self._exact_index.get(key)
        if idx is None:
            idx = self._clean_index.get(clean_text(title))
        if idx is None:
            raise KeyError(f"Title not found in catalog: {title!r}")

        scores = list(enumerate(self.similarity[idx]))
        scores.sort(key=lambda pair: float(pair[1]), reverse=True)
        recs: list[dict[str, Any]] = []
        for row_idx, sim in scores:
            if row_idx == idx:
                continue
            row = self.df.iloc[row_idx]
            recs.append({
                "title": str(row["Title"]),
                "content_type": str(row["Content Type"]),
                "genres": str(row["Genres"]),
                "similarity": round(float(sim), 4),
            })
            if len(recs) >= top_k:
                break
        return recs


def main() -> int:
    parser = argparse.ArgumentParser(description="Content-based Netflix catalog recommender")
    parser.add_argument("--title", default="Sacred Games", help="Seed movie/show title")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    args = parser.parse_args()
    engine = NetflixRecommender()
    print(f"Catalog loaded: {len(engine.df)} titles")
    for rank, rec in enumerate(engine.recommend(args.title, top_k=args.top_k), start=1):
        print(f"{rank:02d}. {rec['title']} ({rec['content_type']}) | sim={rec['similarity']:.4f} | {rec['genres']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
