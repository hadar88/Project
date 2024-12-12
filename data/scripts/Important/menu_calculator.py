import json
foodbyid = open("../../layouts/FoodsByID.json", "r")
data = json.load(foodbyid)
foodbyid.close()


## fiil the data

sunday = {"breakfast": {2:56, 81:32, 29:155, 30:130}, "lunch": {92:60, 73:120, 33:40, 76:20, 35:20, 127:24, 85:28, 15:28}, "dinner": {116:90, 87:40, 33:40, 35:40, 60:30, 10:5, 134:15}}
monday = {"breakfast": {52:30, 74:240, 4:140, 61:60, 135:10, 93:10, 9:35}, "lunch": {116:135, 29:35, 76:60, 10:5, 11:15, 25:75, 35:15, 108:28, 17:32, 61:60}, "dinner": {45:140, 147:15, 136:35}}
tuesday = {"breakfast": {14:40, 74:240, 137:20, 15:28, 46:30}, "lunch": {28:45, 138:60, 41:40, 72:30, 24:120, 25:120}, "dinner": {63:120, 29:75, 90:60, 30:60, 139:5, 140:3, 42:40, }}
wednesday = {"breakfast": {67:45, 142:30, 143:30, 36:30, 93:40, 39:150}, "lunch": {7:100, 41:80, }, "dinner": {}}
thursday = {"breakfast": {}, "lunch": {}, "dinner": {}}
friday = {"breakfast": {}, "lunch": {}, "dinner": {}}
saturday = {"breakfast": {}, "lunch": {}, "dinner": {}}
menu_id = "4"

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