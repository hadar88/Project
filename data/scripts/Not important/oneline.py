import json

db = open("FoodData.json")

dictionary = json.load(db)

db.close()

dictionary = dictionary["Foods"]

onelinedb = open("FoodData2.json", "w")

onelinedb.writelines(["{\n", "\"Foods\": [\n"])

for food in dictionary:
    if "dressing" not in food["Food Name"]:
        onelinedb.write(json.dumps(food))
        onelinedb.write(',\n')

onelinedb.writelines(["]\n", "}\n"])

onelinedb.close()
