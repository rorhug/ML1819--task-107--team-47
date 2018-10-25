import tensorflow as tf
import numpy as np
import json, unidecode
from tensorflow import keras

TESTING_GENDS = 4
GENDER_MAP = {
    "male": 0,
    "female": 1,
    "brand": 2,
    "unknown": 3
}


def generate_feature(user):
    feats = []

    # insert rgb of each customizable color of profile
    colornames = ["profile_background_color", "profile_link_color", "profile_sidebar_border_color", "profile_sidebar_fill_color", "profile_text_color"]
    for colorname in colornames:
        color = user[colorname]
        channels = [color[0:2], color[2:4], color[4:6]]
        for channel in channels:
            value = int(channel, 16) / 255.0 # 0-1 rgb value of channel
            feats.append(value)

    names = [user["name_phonemes"], user["screen_name_phonemes"], user["first_name_phonemes"]]

    for phonemes in names:
        contains = [0 for i in range(40)]
        for sound in phonemes:
            contains[sound] += 1
        feats += contains

        feats.append(phonemes[0])
        feats.append(phonemes[-1])
        feats.append(len(phonemes))

    return feats

Features = []
Labels = []

# LOAD DATA
data = None
with open("users.min.json", "r") as read_file:
    data = json.load(read_file)
for screen_name, user in data.items():
    user_label = GENDER_MAP[user["gender"]]
    if user_label < TESTING_GENDS:
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
model.add(keras.layers.Dense(Features.shape[1] // 2, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(TESTING_GENDS, activation='softmax'))
model.summary()

model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    TrainFeats,
    TrainLabels,
    epochs=100,
    batch_size=2048,
    validation_data=(ValidFeats, ValidLabels),
    #verbose=1
)

results = model.evaluate(TestFeats, TestLabels)
print(results)
