import json

with open('../../food_data/FoodData.json', 'r') as file1:
    data = json.load(file1)

with open('../../food_data/FoodDataSodium.json', 'r') as file2:
    data_sodium = json.load(file2)

for food in data:
    if(food in data_sodium):
        amount = data_sodium[food]
        data[food]['Sodium'] = amount
    else:
        print(food + ", not found in sodium data")

with open('../../food_data/FoodData.json', 'w') as file1:
    json.dump(data, file1, indent=4)

file1.close()
file2.close()

