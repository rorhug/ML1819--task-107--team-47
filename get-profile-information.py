# settings.py
import os
import sys
import csv
import json
import pdb

from dotenv import load_dotenv
import twitter
import gender_guesser.detector as gender

# DATASET DEFINITION

# JSON file, key as their screen_name/user_name, e.g.
# {
#     screen_name: {
#         gender: "male|female|mostly_male|brand|unknown",
#         gender_source: "namesfile|",
#         twitter_profile: {},
#         tweets: [{}]
#     }
# }

# maybe sleep_on_rate_limit=True



load_dotenv()

api = twitter.Api(
    consumer_key=os.getenv("TWITTER_CONSUMER_KEY"),
    consumer_secret=os.getenv("TWITTER_CONSUMER_SECRET"),
    access_token_key=os.getenv("TWITTER_ACCESS_TOKEN_KEY"),
    access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
    sleep_on_rate_limit=True
)

DATASET_FILE = "users.json"

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def lowered_screen_names(users):
    return map(lambda screen_name: screen_name.lower(), list(users.keys()))

def extend_users(users, csv_file_path):
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # pdb.set_trace()

        # added_count = 0

        for row in reader:
            screen_name = row[0].lower()
            gender_column_present = len(row) > 1
            if screen_name not in users:
                users[screen_name] = { "gender": row[1], "gender_source": "" } if gender_column_present else {}

    return True


def populate_twitter_profiles(users):
    screen_names_to_query = [screen_name for screen_name in lowered_screen_names(users) if "twitter_profile" not in users[screen_name]]

    page_count = 0

    for screen_names_chunk in chunks(screen_names_to_query, 100):
        page_count += 1
        print(f"getting page: {page_count} of profiles")
        try:
            user_responses = api.UsersLookup(screen_name=screen_names_chunk, include_entities=True, return_json=True)
            print(f"got {len(user_responses)}/{len(screen_names_chunk)} profiles")
            for user_dict in user_responses:
                users[user_dict["screen_name"].lower()]["twitter_profile"] = user_dict
        except twitter.error.TwitterError as e:
            print("can't get users ")
            print(e)

    for screen_name in list(users.keys()):
        if not 'twitter_profile' in users[screen_name]:
            users.pop(screen_name, None)

    return True

def populate_user_tweets(users):
    screen_names_to_query = [screen_name for screen_name in lowered_screen_names(users) if "tweets" not in users[screen_name]]

    print(f"getting tweets for {len(screen_names_to_query)} users")
    user_count = 0

    for screen_name in screen_names_to_query:
        user_count += 1
        try:
            if user_count % 100 is 0: print(f"getting tweets for user {user_count}")
            tweets = api.GetUserTimeline(screen_name=screen_name, count=200, include_rts=False, trim_user=True, exclude_replies=False)
            users[screen_name]["tweets"] = [tweet.AsDict() for tweet in tweets]
        except twitter.error.TwitterError as e:
            print("can't get timeline for " + screen_name)
            print(e)


def populate_genders(users):
    d = gender.Detector()
    for screen_name in list(users.keys()):
        user = users[screen_name]
        if ("gender" not in user) and ("twitter_profile" in user) and ("name" in user["twitter_profile"]):
            user["gender"] = d.get_gender(user["twitter_profile"]["name"].split(" ")[0])
            user["gender_source"] = "gender-guesser"

def write_dataset(users, file):
    file.truncate(0)
    json.dump(users, file)


def main():
    with open(DATASET_FILE, "a+") as dataset_file:
        dataset_file.seek(0)
        users = json.load(dataset_file)

        # pdb.set_trace()
        print(sys.argv)
        if len(sys.argv) > 1:
            extend_users(users, sys.argv[1])

        populate_twitter_profiles(users)
        write_dataset(users, dataset_file)

        # populate_user_tweets(users)
        # write_dataset(users, dataset_file)

        populate_genders(users)
        write_dataset(users, dataset_file)

if __name__== "__main__":
    main()
