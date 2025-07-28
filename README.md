# 🎬 Movie Recommendation System using Machine Learning

This project implements a **content-based recommendation system** to suggest movies based on a user's input using **Natural Language Processing (NLP)** and **machine learning techniques**. The system is trained on Netflix movie/show data and returns similar titles based on genres.

---

## 📌 Objective

To build a **Movie Recommendation System** that helps users discover movies similar in genre and content to their favorite ones. It uses **TF-IDF vectorization** and **cosine similarity** to recommend relevant titles.

---

## 🗂️ Dataset

The dataset used is `netflixData.csv`, which contains:
- `Title`: Name of the movie or show
- `Description`: Short synopsis
- `Content Type`: Movie or TV Show
- `Genres`: List of genres

---

## ⚙️ How It Works

1. **Data Preprocessing**  
   - Selected key columns
   - Removed null values
   - Cleaned text (lowercasing, stopword removal, stemming, punctuation removal)

2. **Feature Extraction**  
   - Applied **TF-IDF Vectorization** on genres

3. **Similarity Computation**  
   - Used **cosine similarity** to calculate similarity scores between movies

4. **Recommendation Function**  
   - `netflix_recommendation(title)` returns the top 10 most similar movie titles

---

## 📁 Files Included

- `project 2.py` – Python implementation of the recommendation system
- `netflixData.csv` – Dataset used for training and recommendations
- `Movie_Recommendation_System_Report.pdf` – Detailed project report
- `README.md` – This file

---

## ▶️ Example Usage

```python
print(netflix_recommendation("fitoor"))
