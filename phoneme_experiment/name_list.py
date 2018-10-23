import json
import unidecode

data = None
with open("../users.json", "r") as read_file:
    data = json.load(read_file)

for screen_name, user in data.items():
	output = []
	if len(user["twitter_profile"]["name"]) < 2:
		continue
	decoded = unidecode.unidecode(user["twitter_profile"]["name"]).split()
	if len(decoded) == 0:
		continue
	print(unidecode.unidecode(screen_name))
	print(unidecode.unidecode(user["twitter_profile"]["name"]))
	print(decoded[0])