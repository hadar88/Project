import json

# from tqdm import tqdm


with open("data/layouts/FoodsByID.json", "r") as f:
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

    with open("data/layouts/FoodsByIDV2.json", "w") as f:
        json.dump(fixed, f, indent=4)
