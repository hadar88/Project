# This script adds nutritional data to the FoodsByName.json and FoodsByID.json files.
# The nutritional data is taken from the FoodDataDict.json file.
# The script will print an error message if it cannot find the nutritional data for a food.

import json

foods_by_name_file = open("../../layouts/FoodsByName.json", "a+")
foods_by_id_file = open("../../layouts/FoodsByID.json", "w")

foods_by_name_file.seek(0)
foods_by_id_file.seek(0)

foods_by_name: dict[str, dict] = json.load(foods_by_name_file)
foods_by_id: dict[str, dict] = dict()

foods_data_file = open("../../food_data/FoodData.json")
food_data = json.load(foods_data_file)
foods_data_file.close()

id = 1

for food_name in foods_by_name:
    data = food_data.get(food_name, None)

    if data is None:
        print(f"{food_name} : is not in the dict!")
    else:
        foods_by_name[food_name]["Protein"] = data["Protein"]
        foods_by_name[food_name]["Fat"] = data["Fat"]
        foods_by_name[food_name]["Carbohydrate"] = data["Carbohydrate"]
        foods_by_name[food_name]["Calories"] = data["Calories"]
        foods_by_name[food_name]["Sugars"] = data["Sugars"]

    foods_by_name[food_name]["ID"] = id

    foods_by_id[str(id)] = foods_by_name[food_name].copy()
    foods_by_id[str(id)]["Name"] = food_name
    del foods_by_id[str(id)]["ID"]

    id += 1

foods_by_name_file.seek(0)
foods_by_name_file.truncate(0)

json.dump(foods_by_name, foods_by_name_file, indent=4)
json.dump(foods_by_id, foods_by_id_file, indent=4)

foods_by_name_file.close()
foods_by_id_file.close()

# One line

db = open("../../layouts/FoodsByName.json")

dictionary = json.load(db)

db.close()

onelinedb = open("../../layouts/FoodsByNameOneLine.json", "w")

onelinedb.writelines("{\n")

for food in dictionary:
    onelinedb.write('"' + food + '": ')
    onelinedb.write(json.dumps(dictionary[food]["ID"]))
    onelinedb.write(',\n')

onelinedb.writelines(["}\n"])

onelinedb.close()

###


with open("../../layouts/FoodsByID.json", "r") as f:
    foods = json.load(f)
    fixed = {}

    count = 0

    # pbar = tqdm(foods)

    # for food_id in pbar:
    for food_id in foods:
        food = foods[food_id]

        # pbar.set_postfix_str(food["Name"])

        food["Properties"] = {
            "Vegetarian": food["Vegetarian"],
            "Vegan": food["Vegan"],
            "Eggs": food["Contains eggs"],
            "Fish": food["Contains fish"],
            "Gluten": food["Contains gluten"],
            "Lactose": food["Contains milk"],
            "Nuts": food["Contains peanuts or nuts"],
            "Sesame": food["Contains sesame"],
            "Soy": food["Contains soy"],
        }

        del food["Vegetarian"]
        del food["Vegan"]
        del food["Contains eggs"]
        del food["Contains fish"]
        del food["Contains gluten"]
        del food["Contains milk"]
        del food["Contains peanuts or nuts"]
        del food["Contains sesame"]
        del food["Contains soy"]

        food["Alternatives"] = {
            "Vegetarian": [],
            "Vegan": [],
            "Eggs": [],
            "Fish": [],
            "Gluten": [],
            "Lactose": [],
            "Nuts": [],
            "Sesame": [],
            "Soy": [],
        }

        fixed[food_id] = food

    with open("../../layouts/FoodsByIDV2.json.json", "w") as f:
        json.dump(fixed, f, indent=4)
