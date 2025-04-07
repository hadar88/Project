import json

# words = set()
# d = {}

# fd = open("FoodData.json", "r")
# foods_dict = json.load(fd)
# fd.close()

# for food in foods_dict:
#     name = food.replace(",", "").replace("(", "").replace(")", "").replace("-", "").lower()
#     words = name.split()
#     for word in words:
#         if word in d:
#             d[word].append(food)
#         else:
#             d[word] = [food]

# nf = open("food_keys.json", "w")
# json.dump(d, nf, indent=4)
# nf.close()
# print("done")

# with open("food_keys.json", "r") as f:
#     data = json.load(f)

# def distance(a, b):
#     if (len(a) == 0):
#         return len(b)
#     if (len(b) == 0):
#         return len(a)
#     if(a[0] == b[0]):
#         return distance(a[1:], b[1:])
    
#     sub_ = distance(a[1:], b[1:])
#     del_ = distance(a[1:], b)
#     ins_ = distance(a, b[1:])
#     return 1 + min(min(sub_, del_), ins_)

