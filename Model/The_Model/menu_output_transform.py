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
def transform(menu: torch.Tensor, food_data, device):
    output = torch.zeros(23, dtype=torch.int32, device=device)

    output[14], output[15] = 1, 1  # vegetarian, vegan

    daily_calories = torch.zeros(7, device=device)

    total_calories, carbs, sugars, fats, proteins = 0, 0, 0, 0, 0

    meal_calories = torch.zeros(3, device=device)

    menu = menu.to(device)  # Move menu tensor to GPU

    for didx, day in enumerate(menu):
        for midx, meal in enumerate(day):
            for food in meal:
                food_id, food_amount = food.int().tolist()

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

    return output.clone().detach().float().to(device)


def transform_batch(menu_batch: torch.Tensor, food_data, device):
    return torch.stack([transform(menu, food_data, device) for menu in menu_batch])


def transform_batch2(menu_batch: torch.Tensor, food_data, device):
    return torch.stack([transform2(menu, food_data, device) for menu in menu_batch])

def transform2(menu: torch.Tensor, food_data, device):
    output = torch.zeros(23, dtype=torch.float32, device=device)

    output[14] = 1.0
    output[15] = 1.0
    daily_calories = torch.zeros(7, device=device)
    
    for didx, day in enumerate(menu):
        for midx, meal in enumerate(day):
            for food in meal:
                # food: [id, amount]
                food_id = int(food[0].item())
                food_amount = food[1].item() / 100.0
                
                if food_id == 0:
                    continue
                    
                food_nut = food_data[str(food_id)]
                food_calories = food_nut["Calories"] * food_amount
                
                # Update accumulators
                daily_calories[didx] += food_calories
                output[0] = output[0] + food_calories
                output[midx + 1] = output[midx + 1] + food_calories
 
                output[5] = output[5] + food_nut["Carbohydrate"] * food_amount
                output[6] = output[6] + food_nut["Sugars"] * food_amount
                output[7] = output[7] + food_nut["Fat"] * food_amount
                output[8] = output[8] + food_nut["Protein"] * food_amount
                output[9] = output[9] + food_nut["Fruit"]
                output[10] = output[10] + food_nut["Vegetable"]
                output[11] = output[11] + food_nut["Cheese"]
                output[12] = output[12] + food_nut["Meat"]
                output[13] = output[13] + food_nut["Cereal"]
                output[14] = output[14] * food_nut["Vegetarian"]
                output[15] = output[15] * food_nut["Vegan"]
                output[16] = torch.maximum(output[16], torch.tensor(float(food_nut["Contains eggs"]), device=menu.device))
                output[17] = torch.maximum(output[17], torch.tensor(float(food_nut["Contains milk"]), device=menu.device))
                output[18] = torch.maximum(output[18], torch.tensor(float(food_nut["Contains peanuts or nuts"]), device=menu.device))
                output[19] = torch.maximum(output[19], torch.tensor(float(food_nut["Contains fish"]), device=menu.device))
                output[20] = torch.maximum(output[20], torch.tensor(float(food_nut["Contains sesame"]), device=menu.device))
                output[21] = torch.maximum(output[21], torch.tensor(float(food_nut["Contains soy"]), device=menu.device))
                output[22] = torch.maximum(output[22], torch.tensor(float(food_nut["Contains gluten"]), device=menu.device))
    
    # Division operations
    output_final = torch.clone(output)
    output_final[:4] = output[:4] / 7.0
    
    # Calculate daily calorie variance
    output_final[4] = torch.mean((daily_calories - output_final[0]) ** 2)
    
    output_final[5:9] = output[5:9] / 7.0
    
    # Make it require gradients at the end of all operations
    return output_final.requires_grad_(True)


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

foods = open(FOODS_DATA_PATH, "r")
data = json.load(foods)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ten = torch.tensor([[[[[   0.0000,   17.1768],
           [  30.0142,   21.6952],
           [   0.0000,   39.7209],
           [  12.4468,   13.7426],
           [   0.0000,   22.7111],
           [  39.8772,    0.0000],
           [  30.4019,   36.3246],
           [   0.0000,   22.0959],
           [   9.0069,    6.7828],
           [  52.0355,   42.1181]],

          [[  10.3486,    5.2115],
           [  30.8820,   15.2926],
           [   5.9273,   22.0618],
           [  32.6106,   36.8997],
           [   0.0000,   25.3562],
           [  11.0528,    6.0014],
           [   0.0000,   10.0992],
           [  52.6582,   39.5099],
           [  50.1714,   39.5962],
           [  13.2724,   31.0376]],

          [[  31.8785,    0.0000],
           [   6.7387,    0.0000],
           [  10.6516,    8.3881],
           [  11.9279,    4.3864],
           [  23.4180,    0.0000],
           [  30.0874,   62.5886],
           [   0.0000,    2.3418],
           [   0.0000,  134.9892],
           [  12.6375,    0.0000],
           [   0.0000,   23.4727]]],


         [[[   8.4398,    0.0000],
           [  29.7227,   10.6617],
           [  12.4304,   32.7929],
           [   0.0000,    0.0000],
           [  24.6736,    0.0000],
           [  12.2455,    8.6382],
           [  13.5440,    8.8372],
           [   0.0000,   22.0145],
           [   0.0000,   13.1982],
           [  10.8886,    8.0388]],

          [[  14.9169,    0.0000],
           [  51.6139,  140.5629],
           [  73.0315,   54.8961],
           [  10.9834,   20.7797],
           [   0.0000,   20.9463],
           [  11.6223,   10.1481],
           [  12.1470,   32.2493],
           [  44.4784,   42.2348],
           [  14.9769,   17.6536],
           [   6.9187,    4.9452]],

          [[   0.0000, 1729.1578],
           [   0.0000,   10.7200],
           [  29.8060,   43.4946],
           [   0.0000,    3.0446],
           [  11.8908,    3.8774],
           [   0.0000,   27.6857],
           [  51.7838,   36.4892],
           [  15.2489,    0.0000],
           [  44.7930,   42.3554],
           [  12.1844,    5.2652]]],


         [[[   0.0000,   28.6042],
           [  12.5503,   26.2060],
           [  29.3310,   33.2121],
           [  13.1393,   28.9691],
           [  14.4885,    3.6455],
           [  26.6685,    0.0000],
           [   0.0000,    6.5978],
           [  30.5510,   31.0649],
           [   5.6302,    0.0000],
           [  10.1133,   42.5256]],

          [[   0.0000,   43.9262],
           [  10.2890,   43.3273],
           [   0.0000, 1755.6788],
           [  31.0612,   20.1881],
           [   3.8014,    0.0000],
           [  65.5804,  127.4142],
           [  12.8176,    6.4466],
           [  28.2533,    8.2124],
           [  12.7389,    0.0000],
           [  50.1400,  113.7603]],

          [[  75.2503,   68.4490],
           [  10.3986,    0.0000],
           [  56.6494,  114.9066],
           [   0.0000,    0.0000],
           [ 103.0347,   73.6619],
           [  28.4451,   26.2730],
           [  50.0545,    0.0000],
           [  45.8210,   48.5938],
           [  41.4800,    0.0000],
           [  12.5628,   48.6993]]],


         [[[  49.5525,   70.5439],
           [  27.6542,   51.4533],
           [   0.0000,   12.8925],
           [   0.0000, 1782.4897],
           [  30.4708,    0.0000],
           [   0.0000,   17.5376],
           [  53.8125,   19.6628],
           [   8.1301,   11.0077],
           [  40.3409,    0.0000],
           [  11.4928,   30.7450]],

          [[  27.4463,   51.7739],
           [  12.0009,    5.1589],
           [  16.3674,    0.0000],
           [   0.0000, 1776.9613],
           [   0.0000,   25.3321],
           [  43.4758,    0.0000],
           [  26.8906,    0.0000],
           [  12.1496,   58.8203],
           [   0.0000, 1771.1683],
           [  30.4553,   55.8823]],

          [[   0.0000,   27.8884],
           [   0.0000,   34.1954],
           [   7.9311,    5.3028],
           [  45.2084,   53.8488],
           [   0.0000,   17.1869],
           [   0.0000, 1767.4948],
           [  11.6271,   40.9397],
           [  15.6296,    0.0000],
           [  10.8940,    6.0324],
           [  29.0281,   87.4988]]],


         [[[  11.0935,   16.1988],
           [  50.1129,  128.0673],
           [  10.6788,   34.2192],
           [  39.8304,    0.0000],
           [  48.3383,  127.8672],
           [  10.8283,    7.8837],
           [  11.7166,   22.9517],
           [  11.9744,    5.1813],
           [  50.2455,   33.9584],
           [  47.9087,  102.6822]],

          [[  48.0820,   31.0241],
           [  12.0141,    6.1736],
           [   0.0000,   31.9732],
           [  50.7606,   62.5407],
           [  59.0405,   75.2994],
           [  20.8016,   19.6998],
           [  11.8346,   18.1722],
           [   0.0000,   54.1385],
           [  49.1531,  164.2641],
           [  11.3495,   20.9356]],

          [[  13.7323,    7.4384],
           [  11.8772,   12.7295],
           [  32.2454,    0.0000],
           [   0.0000, 1755.0348],
           [   0.0000,   25.1252],
           [   5.4073,    0.0000],
           [  23.3605,    0.0000],
           [  12.1667,    8.5300],
           [  25.9719,    0.0000],
           [  11.3101,   21.1213]]],


         [[[  50.2904,  112.5195],
           [   0.0000,   20.8623],
           [  31.6965,    9.8784],
           [  43.2156,   32.0539],
           [  28.5256,   39.7258],
           [   6.4205,    0.0000],
           [  30.3775,   40.7010],
           [   0.0000, 1756.0840],
           [   0.0000,   23.6204],
           [  46.1715,  109.8620]],

          [[  76.3137,   22.9323],
           [  48.6877,  120.3984],
           [   8.9416,    9.5481],
           [  11.2933,   15.4715],
           [   0.0000, 1763.2839],
           [  32.1070,   37.2864],
           [  13.5437,    0.0000],
           [  47.7750,   69.6657],
           [  49.5842,  168.8957],
           [  12.4832,    9.3148]],

          [[   0.0000,    0.0000],
           [  28.0966,   71.8216],
           [  11.0238,   11.5530],
           [  27.6904,   61.3403],
           [  12.4695,    9.5973],
           [   0.0000, 1761.0240],
           [  67.1755,   78.9214],
           [  16.5201,    0.0000],
           [  26.5453,    0.0000],
           [  28.6019,   19.7209]]],


         [[[  11.1511,    0.0000],
           [   0.0000,   40.0934],
           [  29.8916,   44.4618],
           [  12.1308,   26.2803],
           [   0.0000,   13.5730],
           [  31.3532,   27.6119],
           [   0.0000,    8.9363],
           [  27.2434,   14.0844],
           [  28.7637,   69.4851],
           [  45.1584,   39.3014]],

          [[  77.3049,   32.8041],
           [   0.0000,   16.5800],
           [   0.0000,   27.1757],
           [  11.8617,   32.1072],
           [  11.0183,   37.9876],
           [   0.0000,   29.4596],
           [   0.0000,   10.3109],
           [  26.1463,    0.0000],
           [  11.8719,   18.6913],
           [   0.0000,   16.5747]],

          [[  54.2231,  174.4303],
           [  12.2116,   32.1216],
           [   0.0000,    0.0000],
           [  19.3119,    0.0000],
           [   0.0000,   17.8413],
           [  12.4223,   26.8345],
           [  58.1422,  125.3748],
           [  19.6996,   24.5779],
           [  11.9430,   37.1660],
           [  33.1581,    0.0000]]]]])

# data = transform_batch2(ten, data, device)
# print(data)