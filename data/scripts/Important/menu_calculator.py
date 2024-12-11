import json
foodbyid = open("../../layouts/FoodsByID.json", "r")
data = json.load(foodbyid)
foodbyid.close()


## fiil the data

sunday = {"breakfast": {"3": 245, "64": 30, "4": 75, "65": 28, "20": 30, "39": 150}, "lunch": {"18": 85, "19": 14, "67": 50, "25": 150, "55": 150, "100": 30, "37": 150}, "dinner": {"6": 140, "68": 15, "69": 140, "57": 90, "10": 14, "44": 120}}
monday = {"breakfast": {"2": 28, "42": 75, "70": 120, "16": 180, "71": 150, "15": 35}, "lunch": {"41": 130, "40": 45, "65": 28, "67": 50, "72": 60, "24": 60, "30": 60, "73": 60, "52": 60, "74": 240}, "dinner": {"26": 150, "32": 50, "76": 20, "33": 30, "34": 15, "77": 200, "78": 100, "79": 60, "17": 30, "61": 120}}
tuesday = {"breakfast": {"51": 230, "17": 30, "4": 75, "80": 40, "15": 22, "52": 30, "74": 240}, "lunch": {"18": 113, "42": 40, "82": 28, "33": 20, "108": 56, "83": 30, "73": 60}, "dinner": {"84": 85, "67": 50, "72": 60, "16": 180, "81": 16, "85": 40}}
wednesday = {"breakfast": {"108": 28, "17": 30, "61": 120, "3": 245, "36": 60, "86": 30}, "lunch": {"87": 85, "88": 40, "89": 35, "33": 50, "90": 50, "91": 30, "92": 60, "93": 30, "24": 60, "37": 100}, "dinner": {"56": 113, "94": 150, "21": 80, "95": 20, "52": 30, "74": 240}}
thursday = {"breakfast": {"79": 90, "109": 45, "110": 28, "16": 150, "17": 16, "3": 200}, "lunch": {"2": 56, "42": 75, "37": 120, "80": 40, "15": 22}, "dinner": {"96": 300, "77": 100, "84": 113, "44": 120, "36": 150}}
friday = {"breakfast": {"3": 245, "64": 30, "4": 75, "92": 60, "73": 60}, "lunch": {"26": 113, "98": 15, "33": 20, "76": 10, "108": 56, "30": 100, "65": 28, "99": 150, "100": 30, "37": 100}, "dinner": {"101": 140, "102": 113, "103": 60, "104": 30, "105": 50, "53": 240}}
saturday = {"breakfast": {"108": 28, "42": 75, "70": 60, "106": 40}, "lunch": {"73": 60, "76": 20, "33": 40, "90": 40, "88": 40, "89": 35, "67": 50, "66": 30, "39": 150}, "dinner": {"49": 150, "107": 85, "41": 60, "24": 60, "72": 60, "95": 20, "52": 30, "74": 240}}
menu_id = "2"

##

days = [sunday, monday, tuesday, wednesday, thursday, friday, saturday]
daily_calories = [0, 0, 0, 0, 0, 0, 0]

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

for i, day in enumerate(days):
    breakfast = day["breakfast"]
    lunch = day["lunch"]
    dinner = day["dinner"]

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

###

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

menusinput = open("../../layouts/MenusInput.json", "a+")
menusinput.seek(0)

menus = json.load(menusinput)
data = menus[menu_id]
Menu = data.get("Menu")
Menu["Calories"] = Calories
Menu["Calories1"] = Calories1
Menu["Calories2"] = Calories2
Menu["Calories3"] = Calories3
Menu["Calories MSE"] = Calories_MSE
Menu["Carbohydrate"] = Carbohydrate
Menu["Sugars"] = Sugars
Menu["Fat"] = Fat
Menu["Protein"] = Protein
Menu["Fruit"] = Fruit
Menu["Vegetable"] = Vegetable
Menu["Cheese"] = Cheese
Menu["Meat"] = Meat
Menu["Cereal"] = Cereal
Menu["Vegetarian"] = Vegetarian
Menu["Vegan"] = Vegan
Menu["Contains eggs"] = Contains_eggs
Menu["Contains milk"] = Contains_milk
Menu["Contains peanuts or nuts"] = Contains_peanuts_or_nuts
Menu["Contains fish"] = Contains_fish
Menu["Contains sesame"] = Contains_sesame
Menu["Contains soy"] = Contains_soy
Menu["Contains gluten"] = Contains_gluten

menusinput.seek(0)
menusinput.truncate()
json.dump(menus, menusinput, indent=4)
menusinput.close()

print("Menu calculated successfully")