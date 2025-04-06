import json

words = set()
d = {}

fd = open("FoodData.json", "r")
foods_dict = json.load(fd)
fd.close()

for food in foods_dict:
    name = food.replace(",", "").replace("(", "").replace(")", "").replace("-", "").lower()
    words = name.split()
    for word in words:
        if word in d:
            d[word].append(food)
        else:
            d[word] = [food]

nf = open("food_keys.json", "w")
json.dump(d, nf, indent=4)
nf.close()
print("done")