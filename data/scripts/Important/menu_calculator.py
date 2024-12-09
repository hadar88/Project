import json
foodbyid = open("../../layouts/FoodsByID.json", "r")
data = json.load(foodbyid)
foodbyid.close()
## fiil the data

sunday = {"breakfast": {1: 180, 2: 28, 3: 150, 4: 34, 5: 28}, "lunch": {6: 113, 7: 94, 8: 38, 9: 16, 10: 14,11: 15}, "dinner": {12: 113, 13: 130}}
monday = {"breakfast": {14: 30, 1: 180, 15: 28, 16: 182, 17: 32}, "lunch": {18: 113, 19: 14, 20:30, 52: 30, 15: 28}, "dinner": {6: 170, 21: 156}}
tuesday = {"breakfast": {22: 170, 23 : 40, 5: 28, 24: 120, 25:150}, "lunch": {26: 170, 27: 28, 28:45, 29: 155, 30: 130}, "dinner": {31: 170, 32: 28, 33: 30, 7:20,34:15, 35:20 }}
wednesday = {"breakfast": {22: 70, 36: 60, 37:120, 38:28,39:150,30:130 }, "lunch": {6:170, 7:94, 40:40, 41:40,42:40, 43:15, 10:14,44:2, 52:30}, "dinner": {26:170, 45:320}}
thursday = {"breakfast": {52:30,61:100,17:16, 3:240, 46:30 }, "lunch": {47:113,27:28, 48:32,35:20, 33:30,3:150,15:28,}, "dinner": {6:113,49:100,50:14}}
friday = {"breakfast": {}, "lunch": {}, "dinner": {}}
saturday = {"breakfast": {}, "lunch": {}, "dinner": {}}

days = [sunday, monday, tuesday, wednesday, thursday, friday, saturday]
daily_calories = [0, 0, 0, 0, 0, 0, 0]

menu_id = ""
##

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