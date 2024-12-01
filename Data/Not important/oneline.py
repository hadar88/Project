import json

db = open("FoodDataFinal.json")

dictionary = json.load(db)

db.close()

dictionary = dictionary["Foods"]

onelinedb = open("FoodData.json", "w")

onelinedb.writelines(["{\n", "\"Foods\": [\n"])

for food in dictionary:
    onelinedb.write(json.dumps(food))
    onelinedb.write(',\n')

onelinedb.writelines(["]\n", "}\n"])

onelinedb.close()
