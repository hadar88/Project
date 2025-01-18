################################################################################
# This sctipt is used to convert a menu to a new menu with a desired property. #
# I assumed a new format for the food data.                                    #
# First run "reformat_foodsbid.py" to reformat the food data to the new format.#
################################################################################

import json
import random

# ! Paths relative to the root folder
MENUS_FILE = "data/layouts/menusById.json"
FOODS_FILE = "data/layouts/FoodsByIDV2.json"  # ? chnage this name


def read_menu(id: int):
    with open(MENUS_FILE, "r") as f:
        menus = json.load(f)
        return menus[str(id)]


def read_food(id: int):
    with open(FOODS_FILE, "r") as f:
        foods = json.load(f)
        return foods[str(id)]


def replace_food(menu_id: int, old_food_id: int, new_food_id: int):
    menu: dict[str, dict[str, dict[str, int]]] = read_menu(menu_id).copy()

    for day in menu:
        for meal in menu[day]:
            meal_foods = menu[day][meal]

            if str(old_food_id) in meal_foods:
                amount = meal_foods[old_food_id]
                del meal_foods[old_food_id]
                meal_foods[new_food_id] = amount
                menu[day][meal] = meal_foods

    return menu


# ! I did NOT test this
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
    replaced = True

    # If the property is Vegetarian or Vegan, the positive value is 1, otherwise
    # like in the case of Lactose, the positive value is 0 (no lactose is a restriction).
    desired_property_value = 1 if property in ["Vegetarian", "Vegan"] else 0

    while replaced:
        replaced = False
        new_menus = []

        while menus:
            menu = menus.pop()

            # day is the name of the day (e.g. "Monday")
            for day in menu:
                # meal is the name of the meal (e.g. "Breakfast")
                for meal in menu[day]:
                    meal_foods = menu[day][meal]

                    for food_id in meal_foods:
                        food = read_food(food_id)

                        if food[property] != desired_property_value:
                            alternatives = food["Alternatives"][property]

                            if alternatives == []:
                                raise ValueError(
                                    f"No {property} alternatives for food {food_id}"
                                )

                            replaced = True

                            amount = meal_foods[food_id]

                            if exhaustive:
                                for alternative in alternatives:
                                    # For each alternative, create a new menu with the alternative food.
                                    new_menu = menu.copy()
                                    new_meal = meal_foods.copy()
                                    del new_meal[food_id]
                                    new_meal[str(alternative)] = amount
                                    new_menu[day][meal] = new_meal
                                    new_menus.append(new_menu)
                            else:
                                # Choose a random alternative and replace the food with it.
                                new_menu = menu.copy()
                                new_meal = meal_foods.copy()
                                alternative = random.choice(alternatives)
                                del new_meal[food_id]
                                new_meal[str(alternative)] = amount
                                menu[day][meal] = meal_foods

        menus = new_menus

    return menus if exhaustive else menus[0]
