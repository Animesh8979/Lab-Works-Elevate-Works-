"""Verification tests for Elevate Labs ML internship deliverables."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from clean_titanic import clean_tabular_frame
from netflix_recommender import NetflixRecommender, load_netflix_catalog


def test_netflix_catalog_schema_and_ranked_recommendations() -> None:
    df = load_netflix_catalog()
    assert len(df) == 5967
    rec = NetflixRecommender()
    results = rec.recommend("Sacred Games", top_k=10)
    assert len(results) == 10
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)
    assert all(r["title"].lower() != "sacred games" for r in results)
    assert sims[0] > 0.15


def test_clean_tabular_frame_imputes_encodes_and_scales() -> None:
    raw = pd.DataFrame({
        "Survived": [0, 1, 1, 0],
        "Name": ["A", "B", "C", "D"],
        "Ticket": ["T1", "T2", "T3", "T4"],
        "Cabin": [None, "C85", None, None],
        "Sex": ["male", "female", "female", "male"],
        "Age": [22.0, None, 38.0, 35.0],
        "Fare": [7.25, 71.28, 7.92, 53.10],
        "Embarked": ["S", "C", None, "S"],
    })
    cleaned = clean_tabular_frame(raw)
    assert cleaned.isna().sum().sum() == 0
    assert "Cabin" not in cleaned.columns
    assert "Name" not in cleaned.columns
    assert abs(float(cleaned["Age"].mean())) < 1e-6
