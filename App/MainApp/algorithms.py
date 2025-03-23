def calculate_nutritional_data(goal, activity_type, activity_level, daily_calories):    
    x1, x2, x3, x4 = 0.55, 0.05, 0.3, 0.225

    if goal == "Lose weight":
        x3 = 0.25  
        x4 = 0.25  
    elif goal == "Gain weight":
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
    
    return [carbohydrates, sugars, fats, proteins]

def bmi(weight, height):
    height /= 100
    return round(weight / (height ** 2), 2)

def check_bmi(weight, height):
    b = bmi(weight, height)
    if b < 16:
        return "Severely underweight"
    elif b < 18.5:
        return "Underweight"
    elif b < 25:
        return "Healthy"
    elif b < 30:
        return "Overweight"
    elif b < 40:
        return "Obese"
    else:
        return "Extremely obese"

def bmr(weight, height, age, gender):
    if(gender == "Male"):
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161
    
def amr(weight, height, age, gender, activity_level):
    bmr_value = bmr(weight, height, age, gender)
    if activity_level == "sedentary":
        return bmr_value * 1.2
    elif activity_level == "lightly active":
        return bmr_value * 1.375
    elif activity_level == "moderately active":
        return bmr_value * 1.55
    elif activity_level == "active":
        return bmr_value * 1.725
    elif activity_level == "extremely active":
        return bmr_value * 1.9

def ideal_body_weight(height, gender):
    inch = 0.3937
    height = height * inch

    if(gender == "Male"):
        return 50 + 2.3 * (height - 60)
    else:
        return 45.5 + 2.3 * (height - 60)

def weight_change(current_weight, goal_weight):
    return abs(current_weight - goal_weight)

def time_of_change(current_weight, goal_weight):
    c = weight_change(current_weight, goal_weight)
    return round(260 * c / current_weight)

def weekly_change(current_weight, goal_weight, time):
    c = weight_change(current_weight, goal_weight)
    return round(1000 * c / time)

def daily_calories_change(current_weight, goal_weight):
    t = weekly_change(current_weight, goal_weight)
    return round(9 * t / 7)

def calculate_calories(current_weight, goal_weight, height, age, gender, activity_level):
    a = amr(current_weight, height, age, gender, activity_level)
    p = daily_calories_change(current_weight, goal_weight)
    i = 1 if current_weight <= goal_weight else -1
    return a + i * p

def get_vector(current_weight, goal_weight, height, age, gender, goal, cardio, strength, muscle, activity, vegeterian, vegan, eggs, milk, nuts, fish, sesame, soy, gluten):
    activity_type = []
    if cardio == "1":
        activity_type.append("cardio")
    if strength == "1":
        activity_type.append("strength")
    if muscle == "1":
        activity_type.append("muscle")

    c = calculate_calories(current_weight, goal_weight, height, age, gender, activity)

    result = calculate_nutritional_data(goal, activity_type, activity, c)

    vec = [c, result[0], result[1], result[2], result[3]]
    
    if vegeterian == "1":
        vec.append(1)
    else:
        vec.append(0)
    if vegan == "1":
        vec.append(1)
    else:
        vec.append(0)
    if eggs == "1":
        vec.append(0)
    else:
        vec.append(1)
    if milk == "1":
        vec.append(0)
    else:
        vec.append(1)
    if nuts == "1":
        vec.append(0)
    else:
        vec.append(1)
    if fish == "1":
        vec.append(0)
    else:
        vec.append(1)
    if sesame == "1":
        vec.append(0)
    else:
        vec.append(1)
    if soy == "1":
        vec.append(0)
    else:
        vec.append(1)
    if gluten == "1":
        vec.append(0)
    else:
        vec.append(1)
        
    return vec

