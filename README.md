# 🎬 Movie Recommendation System 🍿

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Status](https://img.shields.io/badge/status-Working-green)


A content-based movie recommendation system that suggests similar movies based on their descriptions.  
Built using TF-IDF vectorization and the sigmoid kernel.

---

## 📦 Dataset
- Uses two CSV files:
  - `tmdb_5000_movies.csv`
  - `tmdb_5000_credits.csv`
- Make sure these files are in the same directory as the script.

---

## 🛠️ How It Works
✅ Loads and merges movie data with credits data  
✅ Cleans the dataset by removing unnecessary columns and filling missing overviews  
✅ Vectorizes the movie descriptions (`overview` column) using **TF-IDF Vectorizer**  
✅ Computes the **sigmoid kernel** similarity matrix  
✅ Takes a movie title as input and recommends the top 10 most similar movies based on text similarity

---

## 📊 Methodology
- **TF-IDF Vectorization**: Converts text into weighted feature vectors while ignoring common stop words.
- **Sigmoid Kernel**: Computes pairwise similarity between movies based on these vectors.
- **Content-Based Filtering**: Recommendations depend only on the content of the movie descriptions.

---

## ▶️ How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn

2. Ensure tmdb_5000_movies.csv and tmdb_5000_credits.csv are in the same folder as the script.

3. Run the script:
   ```bash
   python main.py
4. Enter a movie title when prompted to see recommendations.

## ✅ Example CLI Output
   ```bash
   Enter a movie title for recommendations: The Dark Knight
Top 10 movie recommendations for 'The Dark Knight':
1. Batman Begins
2. Batman Returns
3. The Prestige
4. Man of Steel
5. Batman Forever
6. Superman
7. The Dark Knight Rises
8. Watchmen
9. Superman II
10. Green Lantern
```
## ⚠️ Notes
- Recommendations are case-insensitive; titles are converted to lowercase for matching.

- The script only suggests movies present in the dataset; if the movie title is missing, it notifies the user.

- This is a content-based recommender: it doesn't use user ratings or viewing history.


