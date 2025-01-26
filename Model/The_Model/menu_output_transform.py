import json
import torch

# This function is used to convert the menu dictionary to a tensor
def menu_dict_to_tensor(menu_dict: dict):
    ten = []
    max_foods = max(len(menu_dict[day][meal]) for day in menu_dict for meal in menu_dict[day])
    for day in menu_dict:
        d = []
        for meal in menu_dict[day]:
            m = []
            for food in menu_dict[day][meal]:
                f = [int(food), menu_dict[day][meal][food]]
                m.append(f)
            while len(m) < max_foods:
                m.append([0, 0])  # Padding with zeros
            d.append(m)
        ten.append(d)
    return torch.tensor(ten)


# This function is used to convert the menu tensor to a dictionary
def menu_tensor_to_dict(menu: torch.Tensor):
    days = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    meals = ["breakfast", "lunch", "dinner"]
    menu_dict = {}
    for i, day in enumerate(days):
        menu_dict[day] = {}
        for j, meal in enumerate(meals):
            menu_dict[day][meal] = {}
            for food in menu[i][j]:
                food_id, amount = food.tolist()
                if food_id != 0:  # Ignore padding
                    menu_dict[day][meal][str(food_id)] = amount
    return menu_dict


# This function is used to transform the menu to the menu data
def transform(foods_data_path: str, menu: torch.Tensor):
    menu = menu_tensor_to_dict(menu)
    Calories = 0
    Calories1 = 0
    Calories2 = 0
    Calories3 = 0
    Calories_MSE = 0
    Carbohydrate = 0
    Sugars = 0
    Fat = 0
    Protein = 0
    Fruit = 0
    Vegetable = 0
    Cheese = 0
    Meat = 0
    Cereal = 0
    Vegetarian = 1
    Vegan = 1
    Contains_eggs = 0
    Contains_milk = 0
    Contains_peanuts_or_nuts = 0
    Contains_fish = 0
    Contains_sesame = 0
    Contains_soy = 0
    Contains_gluten = 0
    daily_calories = [0, 0, 0, 0, 0, 0, 0]

    with open(foods_data_path, "r") as data:
        data = json.load(data)
        for i, day in enumerate(menu):
            breakfast = menu[day]["breakfast"]
            lunch = menu[day]["lunch"]
            dinner = menu[day]["dinner"]

            for meal in [breakfast, lunch, dinner]:
                for id in meal:
                    grams = meal[id]
                    food = data[id]
                    daily_calories[i] += food["Calories"] * (grams / 100)
                    Calories = Calories + food["Calories"] * (grams / 100)
                    Carbohydrate = Carbohydrate + food["Carbohydrate"] * (grams / 100)
                    Sugars = Sugars + food["Sugars"] * (grams / 100)
                    Fat = Fat + food["Fat"] * (grams / 100)
                    Protein = Protein + food["Protein"] * (grams / 100)

                    if meal == breakfast:
                        Calories1 = Calories1 + food["Calories"] * (grams / 100)
                    if meal == lunch:
                        Calories2 = Calories2 + food["Calories"] * (grams / 100)
                    if meal == dinner:
                        Calories3 = Calories3 + food["Calories"] * (grams / 100)

                    if food["Fruit"] == 1:
                        Fruit = Fruit + 1
                    if food["Vegetable"] == 1:
                        Vegetable = Vegetable + 1
                    if food["Cheese"] == 1:
                        Cheese = Cheese + 1
                    if food["Meat"] == 1:
                        Meat = Meat + 1
                    if food["Cereal"] == 1:
                        Cereal = Cereal + 1
                    if food["Vegetarian"] == 0:
                        Vegetarian = 0
                    if food["Vegan"] == 0:
                        Vegan = 0
                    if food["Contains eggs"] == 1:
                        Contains_eggs = 1
                    if food["Contains milk"] == 1:
                        Contains_milk = 1
                    if food["Contains peanuts or nuts"] == 1:
                        Contains_peanuts_or_nuts = 1
                    if food["Contains fish"] == 1:
                        Contains_fish = 1
                    if food["Contains sesame"] == 1:
                        Contains_sesame = 1
                    if food["Contains soy"] == 1:
                        Contains_soy = 1
                    if food["Contains gluten"] == 1:
                        Contains_gluten = 1


        Calories = Calories / 7
        Calories1 = Calories1 / 7
        Calories2 = Calories2 / 7
        Calories3 = Calories3 / 7

        Calories_MSE = 1/7 * sum([(daily_calories[i] - Calories) ** 2 for i in range(7)])

        Carbohydrate = Carbohydrate / 7
        Sugars = Sugars / 7
        Fat = Fat / 7
        Protein = Protein / 7

        Calories = round(Calories, 3)
        Calories1 = round(Calories1, 3)
        Calories2 = round(Calories2, 3)
        Calories3 = round(Calories3, 3)
        Calories_MSE = round(Calories_MSE, 3)
        Carbohydrate = round(Carbohydrate, 3)
        Sugars = round(Sugars, 3)
        Fat = round(Fat, 3)
        Protein = round(Protein, 3)
        
        final_data = {
            "Calories": Calories,
            "Calories1": Calories1,
            "Calories2": Calories2,
            "Calories3": Calories3,
            "Calories_MSE": Calories_MSE,
            "Carbohydrate": Carbohydrate,
            "Sugars": Sugars,
            "Fat": Fat,
            "Protein": Protein,
            "Fruit": Fruit,
            "Vegetable": Vegetable,
            "Cheese": Cheese,
            "Meat": Meat,
            "Cereal": Cereal,
            "Vegetarian": Vegetarian,
            "Vegan": Vegan,
            "Contains_eggs": Contains_eggs,
            "Contains_milk": Contains_milk,
            "Contains_peanuts_or_nuts": Contains_peanuts_or_nuts,
            "Contains_fish": Contains_fish,
            "Contains_sesame": Contains_sesame,
            "Contains_soy": Contains_soy,
            "Contains_gluten": Contains_gluten
        }
        return final_data



# Example
menu_dict = {
    "sunday": {"breakfast": {"192": 110, "146": 10, "8": 30}, "lunch": {"73": 150, "42": 50}, "dinner": {"84": 150, "126": 75, "165": 25, "95": 20}},
    "monday": {"breakfast": {"51": 150, "170": 50}, "lunch": {"6": 200, "90": 50}, "dinner": {"169": 200, "21": 100, "131": 75, "118": 25}},
    "tuesday": {"breakfast": {"51": 150, "170": 50}, "lunch": {"6": 150, "76": 30, "91": 20}, "dinner": {"169": 150, "21": 25, "116": 25, "131": 75, "118": 15, "100": 10}},
    "wednesday": {"breakfast": {"61": 100, "60": 100, "52": 100}, "lunch": {"75": 175, "67": 75}, "dinner": {"102": 150, "49": 150, "16": 75, "81": 25}},
    "thursday": {"breakfast": {"42": 75, "66": 75}, "lunch": {"116": 100, "41": 50, "40": 50}, "dinner": {"158": 150, "57": 25, "94": 75, "73": 75, "30": 25}},
    "friday": {"breakfast": {"117": 125, "178": 25}, "lunch": {"84": 150, "186": 50}, "dinner": {"119": 200, "66": 50, "163": 50}},
    "saturday": {"breakfast": {"195": 100, "60": 25, "90": 25}, "lunch": {"18": 170, "76": 20, "134": 10}, "dinner": {"107": 200, "126": 50, "170": 50, "22": 50}}
}

ten = menu_dict_to_tensor(menu_dict)
data = transform("../../Data/layouts/FoodsByID.json", ten)
print(data)