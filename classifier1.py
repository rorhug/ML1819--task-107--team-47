import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
# import matplotlib.pyplot as plt

import json
from dateutil import parser

GENDER_MAP = {
    "male": 0,
    "female": 1,
    "brand": 2,
    "unknown": 3
}

def make_features(user):
    profile = user["twitter_profile"]
    return [
        profile["id"],
        profile["followers_count"],
        profile["friends_count"],
        profile["listed_count"],
        profile["statuses_count"],
        profile["favourites_count"],
        int(profile["verified"]),
        int(profile["profile_use_background_image"]),
        int(parser.parse(profile["created_at"]).strftime("%s")),
    ]

def all_features_present(features):
    for f in features:
        if type(f) is not int:
            return False
    return True


data = None
with open("users.json", "r") as read_file:
    data = json.load(read_file)


features = []
labels = []

for screen_name, user in data.items():
    user_features = make_features(user)
    user_label = GENDER_MAP[user["gender"]]

    if user_label < 3 and all_features_present(user_features):
        features.append(user_features)
        labels.append(user_label)



train_features = np.array(features[1000:])
train_labels = np.array(labels[1000:])

test_features = np.array(features[0:999])
test_labels = np.array(labels[0:999])


model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(9,)),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_features, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_features, test_labels)

print('Test accuracy:', test_acc)
