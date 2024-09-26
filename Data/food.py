import json

# Load the JSON data from the file
with open('FoodData.json', 'r') as file:
    data = json.load(file)

# List of nutrients to keep
nutrients_to_keep = [
    "Protein",
    "Fat",
    "Carbohydrate",
    "Calories",
    "Sugars", 
    "Fiber"
]

# Process each item in the SurveyFoods list
for food in data.get('Foods', []):
    # Create a new dictionary to store the filtered data
    filtered_food = {
        'Food Name': food.get('Food Name', ''),
        'Nutritional data': []
    }

    # Filter and simplify the foodNutrients array
    for nutrient in food.get('Nutritional data', []):
        nutrient_name = nutrient.get('nutrient', {}).get('name')
        if nutrient_name in nutrients_to_keep:
            filtered_food['Nutritional data'].append({
                'name': nutrient_name,
                'amount': nutrient.get('amount')
            })

    # Update the original food item with the filtered one
    food.clear()
    food.update(filtered_food)

# Save the filtered data back to the JSON file
with open('FoodData.json', 'w') as file:
    json.dump(data, file, indent=4)