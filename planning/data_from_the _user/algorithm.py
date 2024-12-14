def calculate_macros(goal, activity_type, activity_level, daily_calories):
    """
    Calculate x1 (carbohydrates), x2 (sugars), x3 (fats), and x4 (proteins).

    Parameters:
        goal (str): "lose", "maintain", or "gain" weight.
        activity_type (list): Activities performed, e.g., ["running", "strength"].
        activity_level (str): "sedentary", "lightly active", "moderately active", "active", "extremely active".
        daily_calories (float): Recommended daily calorie intake (C).

    Returns:
        dict: Macronutrient percentages {x1, x2, x3, x4}.
    """
    # Default macronutrient values (midpoints of ranges)
    x1, x2, x3, x4 = 0.55, 0.05, 0.3, 0.2

    # Step 1: Adjust macronutrients based on activity type
    if "running" in activity_type or "endurance" in activity_type:
        x1 = min(x1 + 0.1, 0.65)  # Increase carbs
        x3 = max(x3 - 0.05, 0.25)  # Decrease fats

    if "strength" in activity_type or "muscle" in activity_type:
        x4 = min(x4 + 0.1, 0.35)  # Increase protein
        x1 = max(x1 - 0.05, 0.45)  # Decrease carbs

    # Step 2: Adjust macronutrients based on activity level
    if activity_level == "sedentary":
        x1, x2, x3, x4 = 0.45, 0.03, 0.25, 0.15  # Favor lower bounds
    elif activity_level == "lightly active":
        x1 = min(x1 + 0.02, 0.55)
        x3 = min(x3 + 0.02, 0.32)
    elif activity_level == "active":
        x1 = min(x1 + 0.05, 0.6)
        x4 = min(x4 + 0.05, 0.3)
    elif activity_level == "extremely active":
        x1 = 0.65
        x4 = 0.35
        x3 = 0.25

    # Ensure values stay within valid ranges
    x1 = max(0.45, min(x1, 0.65))
    x2 = max(0, min(x2, 0.1))
    x3 = max(0.25, min(x3, 0.35))
    x4 = max(0.1, min(x4, 0.35))

    # Return results
    return {
        "x1 (carbohydrates)": x1,
        "x2 (sugars)": x2,
        "x3 (fats)": x3,
        "x4 (proteins)": x4,
        "Calories": daily_calories
    }

# Example usage
result = calculate_macros(
    goal="gain",
    activity_type=["muscle", "strength"],
    activity_level="moderately active",
    daily_calories=2500
)

print(result)
