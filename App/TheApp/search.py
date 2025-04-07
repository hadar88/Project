import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import time

start_time = time.time()

# Load the data
with open('Food_names.json', 'r') as f:
    food_data = json.load(f)
food_names = list(food_data.keys())
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(food_names)
nn = NearestNeighbors(n_neighbors=10, metric='cosine')
nn.fit(X)

# Function to find closest foods
def find_closest_foods(query):
    query_vec = vectorizer.transform([query])
    distances, indices = nn.kneighbors(query_vec)
    return [food_names[i] for i in indices[0]]



# Example search
print(find_closest_foods("White Rice glutinous"))

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
