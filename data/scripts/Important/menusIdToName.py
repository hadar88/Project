import json


menus = open("../../layouts/menusById.json", "r")
menusById = json.load(menus)
menusName = {}
menus.close()

foods = open("../../layouts/FoodsByID.json", "r")
foodsByid = json.load(foods)
foods.close()


for menuid in menusById:
    for day in menusById[menuid]:
        for meal in menusById[menuid][day]:
            for foodid in menusById[menuid][day][meal]:
                if foodid not in foodsByid:
                    print("Food not found: " + foodid)
                    continue
                foodname = foodsByid[foodid]["Name"]
                amount = menusById[menuid][day][meal][foodid]

                if menuid not in menusName:
                    menusName[menuid] = {}
                if day not in menusName[menuid]:
                    menusName[menuid][day] = {}
                if meal not in menusName[menuid][day]:
                    menusName[menuid][day][meal] = {}

                menusName[menuid][day][meal][foodname] = amount

menusByName = open("../../layouts/menusByName.json", "w")
json.dump(menusName, menusByName, indent=4)
menusByName.close()




db = open("../../layouts/menusByName.json")
dict = json.load(db)
db.close()


onelinedb = open("../../layouts/menusByName.json", "w")

onelinedb.writelines("{\n")

for id in dict:
    onelinedb.writelines('\t"' + id + '": {\n')
    for day in dict[id]:
        onelinedb.writelines('\t\t"' + day + '": ')
        onelinedb.writelines(json.dumps(dict[id][day]))
        if day != "saturday":
            onelinedb.writelines(",\n")
        else:
            onelinedb.writelines("\n")
    onelinedb.writelines("\t},\n")

onelinedb.writelines("}")
onelinedb.close()


