import json
import joblib
from collections import defaultdict

FOOD_NAMES_PATH = "Food_names.json"
with open(FOOD_NAMES_PATH, "r") as f:
    food_data = json.load(f)
food_names = food_data["foods"]


char_vectorizer = joblib.load('char_vectorizer.pkl')
char_nn = joblib.load('char_nn.pkl')
word_vectorizer = joblib.load('word_vectorizer.pkl')
word_nn = joblib.load('word_nn.pkl')

def find_closest_foods(query):
    query_char_vec = char_vectorizer.transform([query])
    char_distances, char_indices = char_nn.kneighbors(query_char_vec)
    query_word_vec = word_vectorizer.transform([query])
    word_distances, word_indices = word_nn.kneighbors(query_word_vec)

    combined = defaultdict(float)

    for i in range(len(char_distances[0])):
        char_food = food_names[char_indices[0][i]]
        word_food = food_names[word_indices[0][i]]

        char_distance = 1 - char_distances[0][i]
        word_distance = 1 - word_distances[0][i]

        combined[char_food] += 0.5 * char_distance
        combined[word_food] += 0.5 * word_distance
            
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    return [food for food, _ in sorted_results][:10]