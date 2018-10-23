import tensorflow as tf
import numpy as np
import json
import unidecode
from tensorflow import keras
from random import shuffle

TESTING_GENDS = 2
GENDER_MAP = {
    "male": 0,
    "female": 1,
    "brand": 2,
    "unknown": 3
}

def normalize_name(name):
    arr = []
    # map ascii ordinals to -1 to 1
    for c in name:
        shift = ord(c) - 64.0
        normal = shift / 64.0
        arr.append(normal)
    return arr

def generate_feature(user):
    feats = []
    profile = user["twitter_profile"]
    person_name = unidecode.unidecode(profile["name"]).split(" ")

    # in case no last name supplied
    if len(person_name) == 1:
        person_name.append("")
    # first name, last name, full name and screen name are test features
    names = [person_name[0], person_name[1], person_name[0] + person_name[1], profile["screen_name"]]

    for name in names:
        norm_name = normalize_name(name)
        #feats.append(len(name))
        for i in range(10):
            if len(name) > i:
                # add ordinal of nth letter
                feats.append(norm_name[i])
                # add ordinal of nth last letter
                feats.append(norm_name[i - len(name)])
            else:
                feats.append(0) # one for nth letter
                feats.append(0) # another for nth last letter

    # insert rgb of each customizable color of profile
    colornames = ["profile_background_color", "profile_link_color", "profile_sidebar_border_color", "profile_sidebar_fill_color", "profile_text_color"]

    for colorname in colornames:
        color = profile[colorname]
        channels = [color[0:2], color[2:4], color[4:6]]
        for channel in channels:
            value = int(channel, 16) / 255.0 # 0-1 rgb value of channel
            feats.append(value)

    return feats

def has_required_fields(user):
    profile = user["twitter_profile"]
    return len(profile["name"]) > 1




Features = []
Labels = []

# LOAD DATA
data = None
with open("users.json", "r") as read_file:
    data = json.load(read_file)
for screen_name, user in data.items():
    user_label = GENDER_MAP[user["gender"]]
    if user_label < TESTING_GENDS and has_required_fields(user):
        user_features = generate_feature(user)
        Features.append(user_features)
        Labels.append(user_label)

Features = np.array(Features)
Labels = np.array(Labels)

TestFeats, ValidFeats, TrainFeats = np.split(Features, [500, 1000])
TestLabels, ValidLabels, TrainLabels = np.split(Labels, [500, 1000])

print("Input Layer Size: ", Features.shape[1:])

# define neural net
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape = Features.shape[1:]))
model.add(keras.layers.Dense(Features.shape[1] // 1, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(TESTING_GENDS, activation='softmax'))
model.summary()

model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate=0.005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    TrainFeats,
    TrainLabels,
    epochs=100,
    batch_size=1024,
    validation_data=(ValidFeats, ValidLabels),
    #verbose=1
)

results = model.evaluate(TestFeats, TestLabels)
print(results)