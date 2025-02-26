import json
import torch

FOODS_DATA_PATH = "../../Data/layouts/FoodsByID.json"

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
def transform(menu: torch.Tensor):
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

    with open(FOODS_DATA_PATH, "r") as data:
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
        
        final_data = [
            Calories, Calories1, Calories2, Calories3, Calories_MSE,
            Carbohydrate, Sugars, Fat, Protein, Fruit, Vegetable,
            Cheese, Meat, Cereal, Vegetarian, Vegan, Contains_eggs,
            Contains_milk, Contains_peanuts_or_nuts, Contains_fish,
            Contains_sesame, Contains_soy, Contains_gluten
        ]
        return torch.tensor(final_data)



# Example usage

menu_dict = {
    "sunday": {"breakfast": {"1": 180, "2": 28, "3": 150, "4": 34, "5": 28}, "lunch": {"6": 113, "7": 94, "8": 38, "9": 16, "10": 14, "11": 15}, "dinner": {"12": 113, "13": 130}},
    "monday": {"breakfast": {"14": 30, "1": 180, "15": 28, "16": 182, "17": 32}, "lunch": {"18": 113, "19": 14, "20": 30, "52": 30, "15": 28}, "dinner": {"6": 170, "21": 156}},
    "tuesday": {"breakfast": {"22": 170, "23": 40, "5": 28, "24": 120, "25": 150}, "lunch": {"26": 170, "27": 28, "28": 45, "29": 155, "30": 130}, "dinner": {"31": 170, "32": 28, "33": 30, "7": 20, "34": 15, "35": 20}},
    "wednesday": {"breakfast": {"22": 70, "36": 60, "37": 120, "38": 28, "39": 150, "30": 130}, "lunch": {"6": 170, "7": 94, "40": 40, "41": 40, "42": 40, "43": 15, "10": 14, "44": 2, "52": 30}, "dinner": {"26": 170, "45": 320}},
    "thursday": {"breakfast": {"52": 30, "61": 100, "17": 16, "53": 240, "46": 30}, "lunch": {"47": 113, "27": 28, "48": 32, "35": 20, "33": 30, "3": 150, "15": 28}, "dinner": {"6": 113, "49": 100, "50": 14}},
    "friday": {"breakfast": {"51": 30, "3": 56, "52": 30, "53": 60, "29": 155, "30": 130}, "lunch": {"6": 85, "28": 45, "22": 170, "55": 150}, "dinner": {"56": 170, "57": 120}},
    "saturday": {"breakfast": {"58": 136, "59": 18, "60": 30, "42": 40, "2": 28, "3": 160, "53": 240, "4": 38, "61": 50}, "lunch": {"56": 170, "57": 120, "37": 120}, "dinner": {"6": 113, "63": 100, "62": 15}}
}

# ten = menu_dict_to_tensor(menu_dict)
# print(ten)
# data = transform(ten)
# print("\n#######################\n")
# print(data)