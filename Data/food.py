import json

with open('FoodDataFirst.json', 'r') as file:
    data = json.load(file)

nutrients_to_keep = [
    "Protein",
    "Fat",
    "Carbohydrate",
    "Calories",
    "Sugars", 
    "Fiber",
    "Water"
]

for food in data.get('Foods', []):
    filtered_food = {
        'Food Name': food.get('Food Name', ''),
        'Nutritional data': []
    }
    for nutrient in food.get('Nutritional data', []):
        nutrient_name = nutrient.get('nutrient', {}).get('name')
        if nutrient_name in nutrients_to_keep:
            filtered_food['Nutritional data'].append({
                'name': nutrient_name,
                'amount': nutrient.get('amount')
            })
    food.clear()
    food.update(filtered_food)


with open('FoodDataFinal.json', 'w') as file:
    json.dump(data, file, indent=4)