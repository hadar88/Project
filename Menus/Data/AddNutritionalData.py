import json

foods = json.load(open("Foods.json"))

foodData = json.load(open("FoodData.json"))

foods = foods["Foods"]

foodData = foodData["Foods"]

for food in foods:
    name = food["Name"]
    for food1 in foodData:
        if name == food1["Food Name"]:
            data = food1["Nutritional data"]
            for data1 in data:
                if data1["name"] == "Protein":
                    food["Protein"] = data1["amount"]
                if data1["name"] == "Fat":
                    food["Fat"] = data1["amount"]
                if data1["name"] == "Carbohydrate":
                    food["Carbohydrate"] = data1["amount"]
                if data1["name"] == "Calories":
                    food["Calories"] = data1["amount"]
                if data1["name"] == "Sugars":
                    food["Sugars"] = data1["amount"]



with open("Foods.json", "w") as file:
    json.dump({"Foods": foods}, file, indent=4)




