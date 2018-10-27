import tensorflow as tf
import numpy as np
import json, unidecode
from tensorflow import keras

TESTING_GENDS = 2
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
    return feats

Features = []
NameInputs = []
NickInputs = []
DescInputs = []
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
        NameInputs.append(user["name_phonemes"])
        NickInputs.append(user["screen_name_phonemes"])
        DescInputs.append(user["description"])
        Labels.append(user_label)

NameInputs = keras.preprocessing.sequence.pad_sequences(
    np.array(NameInputs),
    value=40,
    padding='post',
    maxlen=24
)
NickInputs = keras.preprocessing.sequence.pad_sequences(
    np.array(NickInputs),
    value=40,
    padding='post',
    maxlen=24
)
Features = np.array(Features)
Labels = np.array(Labels)

TestSize = int(len(Labels) * .15)
DataSplit = [TestSize, TestSize * 2]

TestFeats, ValidFeats, TrainFeats = np.split(Features, DataSplit)
TestNames, ValidNames, TrainNames = np.split(NameInputs, DataSplit)
TestNicks, ValidNicks, TrainNicks = np.split(NickInputs, DataSplit)
TestLabels, ValidLabels, TrainLabels = np.split(Labels, DataSplit)
TestDescs, ValidDescs, TrainDescs = np.split(DescInputs, DataSplit)

vocabsize = 16
t = keras.preprocessing.text.Tokenizer(num_words=vocabsize)
t.fit_on_texts(TrainDescs)
TestDescTokens = t.texts_to_matrix(TestDescs, mode='count')
ValidDescTokens = t.texts_to_matrix(ValidDescs, mode='count')
TrainDescTokens = t.texts_to_matrix(TrainDescs, mode='count')

name_input = keras.layers.Input(shape=NameInputs.shape[1:], dtype='int32', name='name_input');
name_embed = keras.layers.Embedding(input_dim=41, output_dim=16, input_length=NameInputs.shape[1])(name_input);
name_lstm  = keras.layers.LSTM(16)(name_embed);

nick_input = keras.layers.Input(shape=NickInputs.shape[1:], dtype='int32', name='nick_input');
nick_embed = keras.layers.Embedding(input_dim=41, output_dim=12, input_length=NickInputs.shape[1])(nick_input);
nick_lstm  = keras.layers.LSTM(12)(nick_embed);

desc_input = keras.layers.Input(shape=TrainDescTokens.shape[1:], name='desc_input');
desc_dense = keras.layers.Dense(4, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(desc_input);

auxi_input = keras.layers.Input(shape=Features.shape[1:], name='auxi_input');
model_merge = keras.layers.concatenate([name_lstm, nick_lstm, desc_dense, auxi_input]);
model_drop1 = keras.layers.Dropout(0.2)(model_merge);
model_dense = keras.layers.Dense(32, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(model_drop1);
model_drop2 = keras.layers.Dropout(0.2)(model_dense);
model_dense2 = keras.layers.Dense(16, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(model_drop2);
model_drop3 = keras.layers.Dropout(0.2)(model_dense2);
model_output = keras.layers.Dense(3, activation='softmax', name='model_output')(model_drop3);

model = keras.models.Model(inputs=[name_input, nick_input, desc_input, auxi_input], outputs=[model_output]);

model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    [TrainNames, TrainNicks, TrainDescTokens, TrainFeats],
    [TrainLabels],
    epochs=75,
    batch_size=2048,
    validation_data=([ValidNames, ValidNicks, ValidDescTokens, ValidFeats], [ValidLabels]),
    #verbose=1
)

results = model.evaluate([TestNames, TestNicks, TestDescTokens, TestFeats], [TestLabels])
print(results)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()