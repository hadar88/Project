import json

db = open("../../layouts/menusByName.json")
dict = json.load(db)
db.close()

# convert the json file so each day will be in one line

onelinedb = open("../../layouts/menusByNameOneLine.json", "w")

onelinedb.writelines("{\n")

for id in dict:
    onelinedb.writelines('\t"' + id + '": {\n')
    for day in dict[id]:
        onelinedb.writelines('\t"' + day + '": ')
        onelinedb.writelines(json.dumps(dict[id][day]))
        if day != "saturday":
            onelinedb.writelines(",\n")
        else:
            onelinedb.writelines("\n")
    onelinedb.writelines("\t},\n")

onelinedb.writelines("}")
onelinedb.close()



# onelinedb.write('"' + food + '": ')
# onelinedb.write(json.dumps(dictionary[food]["ID"]))
# onelinedb.write(',\n')