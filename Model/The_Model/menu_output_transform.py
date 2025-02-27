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
def transform(menu: torch.Tensor, food_data):
    output = torch.zeros(23, dtype=torch.int32)

    output[14], output[15] = 1, 1   # vegetarian, vegan

    daily_calories = torch.zeros(7)

    total_calories, carbs, sugars, fats, proteins = 0, 0, 0, 0, 0

    meal_calories = torch.zeros(3)

    for didx, day in enumerate(menu):
        for midx, meal in enumerate(day):
            for fidx, food in enumerate(meal):
                # food: [id, amount]

                food_id, food_amount = food
                food_amount = food_amount.int().item()
                food_id = food_id.int().item()

                if food_id == 0:
                    continue

                food_amount /= 100

                food_nut = food_data[str(food_id)]

                food_calories = food_nut["Calories"] * food_amount

                daily_calories[didx] += food_calories

                total_calories += food_calories

                carbs += food_nut["Carbohydrate"] * food_amount
                sugars += food_nut["Sugars"] * food_amount
                fats += food_nut["Fat"] * food_amount
                proteins += food_nut["Protein"] * food_amount
                
                meal_calories[midx] += food_calories

                output[9] += food_nut["Fruit"]
                output[10] += food_nut["Vegetable"]
                output[11] += food_nut["Cheese"]
                output[12] += food_nut["Meat"]
                output[13] += food_nut["Cereal"]

                output[14] *= food_nut["Vegetarian"]
                output[15] *= food_nut["Vegan"]

                output[16] |= food_nut["Contains eggs"]
                output[17] |= food_nut["Contains milk"]
                output[18] |= food_nut["Contains peanuts or nuts"]
                output[19] |= food_nut["Contains fish"]
                output[20] |= food_nut["Contains sesame"]
                output[21] |= food_nut["Contains soy"]
                output[22] |= food_nut["Contains gluten"]

    output[0] = total_calories // 7
    output[1] = meal_calories[0] // 7
    output[2] = meal_calories[1] // 7
    output[3] = meal_calories[2] // 7
    output[4] = int((1 / 7) * sum((dcal - (total_calories / 7)) ** 2 for dcal in daily_calories))
    output[5] = carbs // 7
    output[6] = sugars // 7
    output[7] = fats // 7
    output[8] = proteins // 7

    return output.clone().detach().float().requires_grad_(True)

def transform_batch(menu_batch: torch.Tensor, food_data):
    return torch.stack([transform(menu, food_data) for menu in menu_batch])



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

# foods = open(FOODS_DATA_PATH, "r")
# data = json.load(foods)

# ten = menu_dict_to_tensor(menu_dict)
# print(ten)
# print(ten.shape)
# data = transform(ten, data)
# print("\n#######################\n")
# print(data)