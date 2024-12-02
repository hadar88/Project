import json

db = open("FoodDataNR.json")

dictionary = json.load(db)

db.close()

dictionary = dictionary["Foods"]

onelinedb = open("FoodDataNRS.json", "w")

onelinedb.writelines(["{\n", "\"Foods\": [\n"])

for food in dictionary:
    if "school" not in food["Food Name"]:
        onelinedb.write(json.dumps(food))
        onelinedb.write(',\n')

onelinedb.writelines(["]\n", "}\n"])

onelinedb.close()
