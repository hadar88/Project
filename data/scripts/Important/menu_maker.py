################################################################################
# This sctipt is used to convert a menu to a new menu with a desired property. #
# I assumed a new format for the food data.                                    #
# First run "reformat_foodsbid.py" to reformat the food data to the new format.#
################################################################################

import json
import random
import copy

# ! Paths relative to the root folder
MENUS_FILE = "../../layouts/menusById.json"
FOODS_FILE = "../../layouts/FoodsByID.json"  
ALTER_FILE = "../../layouts/FoodAlternatives.json" 


def read_menu(id: int):
    with open(MENUS_FILE, "r") as f:
        menus = json.load(f)
        return menus[str(id)]


def read_food(id: int):
    with open(FOODS_FILE, "r") as f:
        foods = json.load(f)
        return foods[str(id)]
    

def get_alternatives(food_id: int, property: str):
    with open(ALTER_FILE, "r") as f:
        alternatives = json.load(f)
        return alternatives[str(food_id)][property]


def convert(property: str, menu_id: int, exhaustive: bool = False) -> dict | list[dict]:
    """
    Covert some menu to a new menu with the desired property.
    For examle non-vegetarian to vegetarian, or one with lactose to one without lactose.

    Args:
        property (str): The problematic property (e.g. "Vegetarian", "Vegan", "Lactose").
        menu_id (int): The id of the menu to convert.
        exhaustive (bool, optional): Whether to try all alternatives or just a random one. Defaults to False.

    Raises:
        ValueError: When there are no alternatives for the property.

    Returns:
        dict | list[dict]: The new menu or a list of new menus if exhaustive.
    """

    menus = [read_menu(menu_id)]
    good_menus = []
    good_menus_set = set()
    made_new_menus = True

    # If the property is Vegetarian or Vegan, the positive value is 1, otherwise
    # like in the case of Lactose, the positive value is 0 (no lactose is a restriction).
    desired_property_value = 1 if property in ["Vegetarian", "Vegan"] else 0

    while made_new_menus:
        made_new_menus = False
        new_menus = []

        while menus:
            menu = menus.pop()
            changed_menu = False

            # day is the name of the day (e.g. "Monday")
            for day in menu:
                # meal is the name of the meal (e.g. "Breakfast")
                for meal in menu[day]:
                    meal_foods = menu[day][meal]

                    for food_id in meal_foods:
                        food = read_food(food_id)

                        if food[property] != desired_property_value:
                            alternatives = get_alternatives(food_id, property)

                            if alternatives == []:
                                raise ValueError(
                                    f"No {property} alternatives for food {food_id}"
                                )
                            
                            made_new_menus = True
                            changed_menu = True
                            
                            if alternatives[0] == -1:
                                new_menu = copy.deepcopy(menu)
                                new_meal = copy.deepcopy(meal_foods)
                                del new_meal[food_id]
                                new_menu[day][meal] = new_meal
                                new_menus.append(new_menu)
                                continue

                            amount = meal_foods[food_id]

                            if exhaustive:
                                for alternative in alternatives:
                                    # For each alternative, create a new menu with the alternative food.
                                    new_menu = copy.deepcopy(menu)
                                    new_meal = copy.deepcopy(meal_foods)
                                    del new_meal[food_id]
                                    new_meal[str(alternative)] = amount
                                    new_menu[day][meal] = new_meal
                                    new_menus.append(new_menu)
                            else:
                                # Choose a random alternative and replace the food with it.
                                new_menu = copy.deepcopy(menu)
                                new_meal = copy.deepcopy(meal_foods)
                                alternative = random.choice(alternatives)
                                del new_meal[food_id]
                                new_meal[str(alternative)] = amount
                                new_menu[day][meal] = new_meal
                                new_menus.append(new_menu)     

            if not changed_menu:
                menu_str = json.dumps(menu, sort_keys=True)
                
                if menu_str not in good_menus_set:
                    good_menus_set.add(menu_str)
                    good_menus.append(menu)  

        menus = new_menus                     

    return good_menus if exhaustive else good_menus[0]

properties = ["Vegetarian", "Vegan", "Contains eggs", "Contains milk", "Contains peanuts or nuts", "Contains fish", "Contains sesame", "Contains soy", "Contains gluten"]

menu_to_fix = input("Enter the menu id to fix: ")

for i, name in enumerate(properties):
    print(f"{i}) {name}")

property_to_fix = int(input("Enter the property to fix: "))

org = read_menu(menu_to_fix)

print("{")

for day in org:
    print(f'"{day}": ', end="")
    print(json.dumps(org[day]), end="")

    if day != "saturday":
        print(",")

print("\n}")

print("\n\n")
print("Fixed menus:")
print("\n\n")

for i, m in enumerate(convert(properties[property_to_fix], int(menu_to_fix), exhaustive=True)):
    print(f"Menu {i + 1}:")
    print("{")

    for day in m:
        print(f'"{day}": ', end="")
        print(json.dumps(m[day]), end="")

        if day != "saturday":
            print(",")

    print("\n}")

    print("\n\n")

