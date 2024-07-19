import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

def load_and_prepare_data():
    # Load the datasets
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = pd.read_csv("tmdb_5000_movies.csv")

    # Rename the 'movie_id' column to 'id' in credits dataframe
    credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})

    # Merge the movies and credits dataframes on 'id'
    movies_merge = movies.merge(credits_column_renamed, on='id')

    # Drop unnecessary columns
    movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])

    # Fill NaN values in the 'overview' column with an empty string
    movies_cleaned['overview'] = movies_cleaned['overview'].fillna('')

    return movies_cleaned

def compute_similarity_matrix(movies_cleaned):
    # Initialize the TF-IDF Vectorizer
    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', 
                          analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3),
                          stop_words='english')

    # Fit and transform the 'overview' column
    tfv_matrix = tfv.fit_transform(movies_cleaned['overview'])

    # Compute the sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

    return sig

def create_indices(movies_cleaned):
    # Create a Series with movie titles as index and their corresponding index as values
    indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title'].str.lower()).drop_duplicates()
    return indices

def give_recommendations(title, sig, indices, movies_cleaned):
    # Get the index of the movie that matches the title
    idx = indices[title.lower()]

    # Get the pairwise similarity scores of all movies with that movie
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies based on the similarity scores
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Return the top 10 most similar movies
    return movies_cleaned['original_title'].iloc[movie_indices]

def main():
    # Load and prepare data
    movies_cleaned = load_and_prepare_data()

    # Compute similarity matrix
    sig = compute_similarity_matrix(movies_cleaned)

    # Create indices
    indices = create_indices(movies_cleaned)

    # Get user input for movie title
    user_movie = input("Enter a movie title for recommendations: ").strip()

    # Check if the movie is in the dataset
    if user_movie.lower() not in indices:
        print(f"Sorry, the movie '{user_movie}' is not in the database.")
        return

    # Get recommendations
    recommendations = give_recommendations(user_movie, sig, indices, movies_cleaned)

    # Print recommendations
    print(f"Top 10 movie recommendations for '{user_movie}':")
    for i, title in enumerate(recommendations, 1):
        print(f"{i}. {title}")

# Run the main function
if __name__ == "__main__":
    main()
