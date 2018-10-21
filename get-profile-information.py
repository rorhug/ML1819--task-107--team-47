# settings.py
import os
import csv
import json


from dotenv import load_dotenv
import twitter

load_dotenv()

api = twitter.Api(
    consumer_key=os.getenv("TWITTER_CONSUMER_KEY"),
    consumer_secret=os.getenv("TWITTER_CONSUMER_SECRET"),
    access_token_key=os.getenv("TWITTER_ACCESS_TOKEN_KEY"),
    access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def get_profiles_from_usernames(file_path):
    users = {}

    # <screen_name>: { gender: g, twitter_profile: {}, tweets: {} }

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            users[row[0].lower()] = { "gender": row[1] }

    for usernames_chunk in chunks(list(users.keys()), 100):
        response = api.UsersLookup(screen_name=usernames_chunk, include_entities=True, return_json=True)

        for user_dict in response:
            users[user_dict["screen_name"].lower()]["twitter_profile"] = user_dict

    for screen_name in list(users.keys()):
        if not 'twitter_profile' in users[screen_name]:
            users.pop(screen_name, None)

    return users


def write_users(users):
    with open('users.json', 'w') as outfile:
        json.dump(users, outfile)



