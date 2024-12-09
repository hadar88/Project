import json

db = open("../../layouts/FoodsByName.json")

dictionary = json.load(db)

db.close()

onelinedb = open("FoodDataByName2.json", "w")

onelinedb.writelines("{\n")

for food in dictionary:
    onelinedb.write('"' + food + '": ')
    onelinedb.write(json.dumps(dictionary[food]))
    onelinedb.write(',\n')

onelinedb.writelines(["}\n"])

onelinedb.close()
