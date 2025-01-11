import json

menus = open("../../layouts/menusId.json", "r")
menusId = json.load(menus)
menus.close()

foods = open("../../layouts/FoodsByID.json", "r")
foodsbyid = json.load(foods)
foods.close()

menusName = open("../../layouts/menusName.json", "w")

for menuid in menusId:
    for day in menusId[menuid]:
        for meal in menusId[menuid][day]:
            for foodid in menusId[menuid][day][meal]:
                if foodid not in foodsbyid:
                    print("Food not found: " + foodid)
                    continue
                foodname = foodsbyid[foodid]["Name"]
                menusId[menuid][day][meal][foodid] = foodname


json.dump(menusId, menusName, indent=4)
menusName.close()



