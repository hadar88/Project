def calculate_nutritional_data(goal, activity_type, activity_level, daily_calories):
    """
    Calculate carbohydrates, sugars, fats, and proteins.

    Parameters:
        goal (str): "lose", "maintain", or "gain" weight.
        activity_type (list): Activities performed, e.g., ["cardio", "strength"].
        activity_level (str): "sedentary", "lightly active", "moderately active", "active", "extremely active".
        daily_calories (float): Recommended daily calorie intake (C).

    Returns:
        dict: Macronutrient percentages {carbohydrates, sugars, fats, proteins}.
    """
    
    x1, x2, x3, x4 = 0.55, 0.05, 0.3, 0.225

    if goal == "lose":
        x3 = 0.25  
        x4 = 0.25  
    elif goal == "gain":
        x3 = 0.35  
        x4 = 0.3    

    if "cardio" in activity_type:
        x1 = 0.6  
        x3 = max(x3 - 0.05, 0.25) 
    if "strength" in activity_type or "muscle" in activity_type:
        x1 = max(x1 - 0.05, 0.45)  
        x4 = min(x4 + 0.1, 0.35)  

    if activity_level == "sedentary":
        x1 = max(x1 - 0.05, 0.45)
        x3 = max(x3 - 0.05, 0.25)
    elif activity_level == "lightly active":
        x1 = min(x1 + 0.02, 0.55)
        x3 = min(x3 + 0.02, 0.275)
    elif activity_level == "moderately active":
        x1 = min(x1 + 0.05, 0.6)
        x3 = min(x3 + 0.05, 0.3)
    elif activity_level == "active":
        x1 = min(x1 + 0.07, 0.62)
        x3 = min(x3 + 0.07, 0.325)
    elif activity_level == "extremely active":
        x1 = min(x1 + 0.1, 0.65)
        x3 = min(x3 + 0.1, 0.35)

    x1 = max(0.45, min(x1, 0.65))
    x2 = max(0, min(x2, 0.1))
    x3 = max(0.25, min(x3, 0.35))
    x4 = max(0.1, min(x4, 0.35))

    sum = x1 + x2 + x3 + x4
    x1 /= sum
    x2 /= sum
    x3 /= sum
    x4 /= sum

    carbohydrates = (daily_calories * x1) / 4
    sugars = (daily_calories * x2) / 4
    fats = (daily_calories * x3) / 9
    proteins = (daily_calories * x4) / 4

    carbohydrates = round(carbohydrates, 2)
    sugars = round(sugars, 2)
    fats = round(fats, 2)
    proteins = round(proteins, 2)
    
    return {
        "carbohydrates": carbohydrates,
        "sugars": sugars,
        "fats": fats,
        "proteins": proteins
    }

# Example
result = calculate_nutritional_data(
    goal="gain",
    activity_type=["cardio", "muscle", "strength"],
    activity_level="active",
    daily_calories=2500
)

print(result)
