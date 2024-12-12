import json
foodbyid = open("../../layouts/FoodsByID.json", "r")
data = json.load(foodbyid)
foodbyid.close()


## fiil the data

sunday = {"breakfast": {"108": 60, "42": 75, "70": 100, "3": 145, "111": 170, "61": 120, "17": 32}, "lunch": {"18": 140, "19": 28, "67": 70, "112": 28, "113": 100, "114": 70}, "dinner": {"6": 113, "115": 30, "116": 185, "21": 91, "10": 14, "117": 96, "81": 32}}
monday = {"breakfast": {"131": 245, "64": 55, "4": 74, "118": 14, "71": 90, "15": 28}, "lunch": {"119": 375, "108": 60, "65": 56, "33": 40, "120": 28, "30": 61, "73": 122}, "dinner": {"75": 113, "32": 60, "77": 196, "10": 14, "78": 85, "10": 14, "44": 132, "121": 43}}
tuesday = {"breakfast": {"51": 234, "132": 245, "17": 32, "16": 150, "54": 2.6, "55": 150, "100": 14}, "lunch": {"41": 86, "40": 82, "65": 28, "67": 70, "72": 61, "24": 122, "131": 245, "64": 28}, "dinner": {"56": 113, "96": 332, "21": 45, "10": 7, "122": 90}}
wednesday = {"breakfast": {"123": 90, "33": 40, "109": 43, "110": 85, "16": 200, "15": 21, "80": 28}, "lunch": {"87": 86, "88": 34, "89": 28, "33": 90, "90": 52, "91": 30, "92": 57, "73": 61, "65": 28, "20": 30}, "dinner": {"84": 113, "49": 195, "25": 119, "29": 40, "14": 32, "105": 60, "44": 60}}
thursday = {"breakfast": {"108": 60, "17": 32, "61": 120, "37": 100, "114": 70}, "lunch": {"67": 70, "75": 113, "65": 28, "98": 16, "19": 14, "124": 154, "83": 28, "125": 28, "24": 122}, "dinner": {"107": 113, "126": 210, "57": 200, "10": 14, "127": 16, "95": 56}}
friday = {"breakfast": {"131": 245, "64": 55, "36": 62, "118": 14, "108": 30, "17": 32}, "lunch": {"18": 140, "19": 28, "67": 70, "112": 28, "113": 100, "128": 45, "30": 61, "120": 28}, "dinner": {"102": 113, "101": 210, "103": 50, "29": 80, "104": 32, "117": 72, "81": 32, "95": 28}}
saturday = {"breakfast": {"108": 60, "70": 100, "42": 75, "3": 245, "129": 74, "61": 120, "17": 32}, "lunch": {"87": 86, "88": 34, "89": 28, "33": 90, "90": 52, "91": 30, "92": 57, "73": 61, "15": 21, "133": 28}, "dinner": {"69": 195, "68": 32, "130": 132, "10": 14, "127": 16, "5": 28, "95": 28}}
menu_id = "3"

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
                Fruit = Fruit + 13
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