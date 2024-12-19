import json

with open('../../initial_data/FoodDataFirst.json', 'r') as file:
    data = json.load(file)

nutrients_to_keep = [
    "Sodium"
]

filtered_food = {}
for food in data.get('Foods', []):
    for nutrient in food.get('Nutritional data', []):
        nutrient_name = nutrient.get('nutrient', {}).get('name')
        if nutrient_name in nutrients_to_keep:
            filtered_food[food.get('Food Name', '')]= nutrient.get('amount')

data = filtered_food




with open('FoodDataSodium.json', 'w') as file:
    json.dump(data, file, indent=4)