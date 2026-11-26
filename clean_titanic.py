"""Reusable pandas & scikit-learn tabular preprocessing workflow (Elevate Labs ML Internship)."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_tabular_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Age" in out.columns:
        out["Age"] = out["Age"].fillna(out["Age"].median())
    if "Embarked" in out.columns:
        mode_vals = out["Embarked"].mode()
        if len(mode_vals):
            out["Embarked"] = out["Embarked"].fillna(mode_vals.iloc[0])
    for drop_col in ("Cabin", "Name", "Ticket"):
        if drop_col in out.columns:
            out = out.drop(columns=[drop_col])
    if "Sex" in out.columns:
        out["Sex"] = out["Sex"].map({"male": 0, "female": 1}).fillna(0).astype(int)
    if "Embarked" in out.columns:
        out = pd.get_dummies(out, columns=["Embarked"], drop_first=True)
    num_cols = [c for c in ("Age", "Fare") if c in out.columns]
    if num_cols:
        scaler = StandardScaler()
        out[num_cols] = scaler.fit_transform(out[num_cols])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean tabular Titanic-schema CSV via median/mode imputation and StandardScaler.")
    parser.add_argument("--input", default="Titanic-Dataset.csv")
    parser.add_argument("--output", default="Titanic-Cleaned.csv")
    args = parser.parse_args()
    raw = pd.read_csv(Path(args.input))
    cleaned = clean_tabular_frame(raw)
    cleaned.to_csv(Path(args.output), index=False)
    print(f"Cleaned {len(cleaned)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
