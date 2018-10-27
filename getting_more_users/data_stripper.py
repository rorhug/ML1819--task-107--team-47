
original = "C:/Users/Ed/Downloads/TwitterFriends/data.csv"
save_to = "usernames.txt"

with open(original, "r") as f:
    lines = f.read().split("\n")

usernames = [l.split(",")[1].replace("\"", "") for l in lines]
unduped = list(set(usernames))

print("Removed", len(usernames) - len(unduped), "duplicated usernames")

with open(save_to, "w") as f:
    f.write("\n".join(unduped))
