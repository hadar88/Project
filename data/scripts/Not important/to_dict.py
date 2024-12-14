# This script converts the FoodData.json file into a dictionary and saves it as FoodDataDict.json.
# This is done to make it easier to access the nutritional data of a food item by its name
# (as opposed to having to iterate through the list of food items to find the one with the matching name).

import json

# Load the food data from the FoodData.json file
food_data_file = open("FoodData.json")
food_data = json.load(food_data_file)
food_data_file.close()

food_data: dict = food_data["Foods"]

food_data_dict = {}

# Convert the food data into a dictionary with the food name as the key
for food in food_data:
    name = food["Food Name"]
    data = food["Nutritional data"]

    data_dict = {}

    for nutrient in data:
        data_dict[nutrient["name"]] = nutrient["amount"]

    food_data_dict[name] = data_dict

# Save the food data dictionary to the FoodDataDict.json file
with open("FoodDataDict.json", "w") as food_data_dict_file:
    food_data_dict_file.write("{\n")

    for food in food_data_dict:
        food_data_dict_file.write(f'"{food}": {json.dumps(food_data_dict[food])},\n')

    # Remove the last comma
    food_data_dict_file.seek(food_data_dict_file.tell() - 3)
    food_data_dict_file.truncate()

    food_data_dict_file.write("\n}")
