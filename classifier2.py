import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import unidecode
# import matplotlib.pyplot as plt

import json
from dateutil import parser

GENDER_MAP = {
    "male": 0,
    "female": 1,
    "brand": 2,
    "unknown": 3
}

HOW_MANY_GENDERS = 2

def make_features(user):
    profile = user["twitter_profile"]
    first_name = unidecode.unidecode(profile["name"].split(" ")[0]).lower()

    return [ ord(c) for c in list(first_name) ]

def all_features_present(features):
    return features and len(features) > 1

data = None
with open("users.json", "r") as read_file:
    data = json.load(read_file)


features = []
labels = []

for screen_name, user in data.items():
    user_features = make_features(user)
    user_label = GENDER_MAP[user["gender"]]

    if user_label < HOW_MANY_GENDERS and all_features_present(user_features):
        features.append(user_features)
        labels.append(user_label)


padded_features = keras.preprocessing.sequence.pad_sequences(
    np.array(features),
    value=0,
    padding='post',
    maxlen=20
)
np_labels = np.array(labels)

train_features = padded_features[1000:]
test_features = padded_features[:1000]

train_labels = np_labels[1000:]
test_labels = np_labels[:1000]

# test_features = keras.preprocessing.sequence.pad_sequences(
#     np.array(features[0:1000]),
#     value=0,
#     padding='post',
#     maxlen=20
# )
# test_labels = np.array(labels[0:1000])

x_val = train_features[:1500]
partial_x_train = train_features[1500:]

y_val = train_labels[:1500]
partial_y_train = train_labels[1500:]

vocab_size = 128

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 32))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(HOW_MANY_GENDERS, activation=tf.nn.softmax))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.05),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=100,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

results = model.evaluate(test_features, test_labels)

print(results)



# model.fit(train_features, train_labels, epochs=5)

# test_loss, test_acc = model.evaluate(test_features, test_labels)

# print('Test accuracy:', test_acc)
