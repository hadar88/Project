import json

db = open("../../food_data/FoodData.json")

dictionary = json.load(db)

db.close()

onelinedb = open("../../food_data/FoodData1.json", "w")

onelinedb.writelines("{\n")

for food in dictionary:
    onelinedb.write('"' + food + '": ')
    onelinedb.write(json.dumps(dictionary[food]))
    onelinedb.write(',\n')

onelinedb.writelines(["}\n"])

onelinedb.close()
