import json

menus_input = open("../../layouts/menusInput.json", "a+")

menus_input.seek(0)

menus = json.load(menus_input)

for number in menus:
    data = menus[number]
    initial = data.get("Initial")
    menu = data.get("Menu")
    
    initial["Calories"] = menu["Calories"]
    initial["Carbohydrate"] = menu["Carbohydrate"]
    initial["Sugars"] = menu["Sugars"]
    initial["Fat"] = menu["Fat"]
    initial["Protein"] = menu["Protein"]
    initial["Vegetarian"] = menu["Vegetarian"]
    initial["Vegan"] = menu["Vegan"]
    initial["Contains eggs"] = menu["Contains eggs"]
    initial["Contains milk"] = menu["Contains milk"]
    initial["Contains peanuts or nuts"] = menu["Contains peanuts or nuts"]
    initial["Contains fish"] = menu["Contains fish"]
    initial["Contains sesame"] = menu["Contains sesame"]
    initial["Contains soy"] = menu["Contains soy"]
    initial["Contains gluten"] = menu["Contains gluten"]


menus_input.seek(0)
menus_input.truncate()
json.dump(menus, menus_input, indent=4)


menus_input.close()
