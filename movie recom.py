import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init, Fore
import time
import sys

init(autoreset=True)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Overview'] = df['Overview'].fillna('')
        df['Genre'] = df['Genre'].fillna('')
        df['IMDB_Rating'] = df['IMDB_Rating'].fillna(0)
        df['combined_features'] = df['Overview'] + ' ' + df['Genre']
        return df
    except FileNotFoundError:
        print(Fore.RED + f"Error: {file_path} not found.")
        sys.exit()

def calculate_similarity(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    return cosine_similarity(tfidf_matrix)

def list_genres(df):
    genres = set()
    for sublist in df['Genre'].str.split(','):
        if isinstance(sublist, list):
            for genre in sublist:
                genres.add(genre.strip())
    return sorted(genres)

def recommend_movies(df, genre=None, mood=None, rating=None, top_n=5):
    filtered_df = df.copy()
    if genre:
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(genre, case=False, na=False)]
    if rating:
        filtered_df = filtered_df[filtered_df['IMDB_Rating'] >= rating]
    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)
    recommendations = []
    for _, row in filtered_df.iterrows():
        overview = row['Overview']
        if not overview:
            continue
        polarity = TextBlob(overview).sentiment.polarity
        if mood:
            mood_score = TextBlob(mood).sentiment.polarity
            if (mood_score >= 0 and polarity >= 0) or (mood_score < 0 and polarity < 0):
                recommendations.append((row['Series_Title'], row['Genre'], row['IMDB_Rating'], polarity))
        else:
            recommendations.append((row['Series_Title'], row['Genre'], row['IMDB_Rating'], polarity))
        if len(recommendations) >= top_n:
            break
    return recommendations if recommendations else "No suitable movie recommendations found."

def display_recommendations(recommendations, name):
    print(Fore.YELLOW + f"\nAI-Analyzed Movie Recommendations for {name}:\n")
    for idx, (title, genre, rating, polarity) in enumerate(recommendations, 1):
        sentiment = "Positive 😊" if polarity > 0 else "Negative 😞" if polarity < 0 else "Neutral 😐"
        print(f"{idx}. {title} | Genre: {genre} | IMDB: {rating} | Sentiment: {sentiment} (Polarity: {polarity:.2f})")

def processing():
    for _ in range(3):
        print(Fore.YELLOW + ".", end='', flush=True)
        time.sleep(0.5)
    print()

def handle_ai(name, df):
    print(Fore.BLUE + "\n🎯 Let's find the perfect movie for you!")
    genres = list_genres(df)
    print(Fore.GREEN + "\nAvailable Genres:")
    for idx, genre in enumerate(genres, 1):
        print(f"{idx}. {genre}")
    while True:
        genre_input = input(Fore.YELLOW + "\nEnter genre number or name: ").strip()
        if genre_input.isdigit() and 1 <= int(genre_input) <= len(genres):
            genre = genres[int(genre_input) - 1]
            break
        elif genre_input.title() in genres:
            genre = genre_input.title()
            break
        else:
            print(Fore.RED + "Invalid input. Try again.")
    mood = input(Fore.YELLOW + "How do you feel today? (Describe your mood): ").strip()
    while True:
        try:
            rating = float(input(Fore.YELLOW + "Minimum IMDB rating? (0-10): ").strip())
            if 0 <= rating <= 10:
                break
        except ValueError:
            pass
        print(Fore.RED + "Please enter a valid number.")
    print(Fore.CYAN + "\n🔍 Analyzing preferences", end='')
    processing()
    recs = recommend_movies(df, genre=genre, mood=mood, rating=rating, top_n=5)
    if isinstance(recs, str):
        print(Fore.RED + recs)
    else:
        display_recommendations(recs, name)
    while True:
        again = input(Fore.YELLOW + "\nWould you like more recommendations? (yes/no): ").strip().lower()
        if again == 'yes':
            handle_ai(name, df)
            break
        elif again == 'no':
            print(Fore.GREEN + f"\n🎬 Enjoy your movie picks, {name}!")
            break
        else:
            print(Fore.RED + "Invalid input. Try again.")

def main():
    print(Fore.BLUE + "🎥 Welcome to your Personal Movie Recommendation Assistant!")
    name = input(Fore.YELLOW + "What's your name? ").strip()
    print(Fore.GREEN + f"\nNice to meet you, {name}!\n")
    df = load_data("movies.csv")
    _ = calculate_similarity(df)
    handle_ai(name, df)

if __name__ == "__main__":
    main()
