import os
import tensorflow as tf
import numpy as np
import json
import unidecode
from tensorflow import keras
from random import shuffle
import colorsys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

MALE = "male"
FEMALE = "female"
GENDERMAP = {
	MALE: 0,
	FEMALE: 1
}

file_writer = tf.summary.FileWriter('/path/to/logs', sess.graph)

# given a dataset, return a filtered dataset which removes undesirables
# also sets male + female to same amount
def prune(dataset):
	males = []
	females = []
	for screen_name, user in dataset.items():
		gender = user["gender"]
		profile = user["twitter_profile"]
		name = profile["name"]
		if " " not in name:
			continue
		if gender == MALE:
			males.append([screen_name, user])
		elif gender == FEMALE:
			females.append([screen_name, user])
	numeach = min(len(males), len(females))
	print(f"Pruned dataset down to {numeach} males and {numeach} females.")
	prunset = {}
	for i in range(int(numeach * 0.5)):
		prunset[males[i][0]] = males[i][1]
		prunset[females[i][0]] = females[i][1]
	return prunset


# transform name string into 20 numbers (first 10 and last 10 character ordinals) and add
def addname(features, name):
	count = 0
	for i in range(3):
		count += 2
		if len(name) > i:
			features.append(ord(name[i]))
			features.append(ord(name[-i-1]))
		else:
			features.append(0)
			features.append(0)
	return count

# transform colour hexstring into R, G and B and add it
def addcolor(features, hexstring):
	channels = [hexstring[0:2], hexstring[2:4], hexstring[4:6]]
	rgb = []
	count = 0
	for channel in channels:
		rgb.append(int(channel, 16) / 255)
		count += 1

	features += colorsys.rgb_to_hsv(*rgb)
	return count

# given a dataset, return a feature set
# unicode is converted to ascii
# first and last 10 characters of each padded name used to represent name
# r, g, b used to represent colours
def featurify(dataset):
	featset = []
	labelset = []
	for screen_name, user in dataset.items():
		features = []
		profile = user["twitter_profile"]
		personnames = unidecode.unidecode(profile["name"]).lower().split(" ", 1)
		firstname = personnames[0]
		lastname = personnames[1]

		featcount = 0

		featcount += addname(features, firstname)
		featcount += addname(features, lastname)
		featcount += addname(features, screen_name)

		colornames = [
			"profile_background_color",
			"profile_link_color",
			"profile_sidebar_border_color",
			"profile_sidebar_fill_color",
			"profile_text_color"
		]
		for colorname in colornames:
			featcount += addcolor(features, profile[colorname])

		featcount += 3
		features.append(profile["followers_count"])
		features.append(profile["friends_count"])
		features.append(profile["favourites_count"])

		featset.append(features)

		label = GENDERMAP[user["gender"]]
		labelset.append(label)

	return (featset, labelset), [i for i in range(featcount)]

# standardize a feature set
def standardize(featxy):
	featset = featxy[0]
	labelset = featxy[1]

	rows = len(featset)
	feats = len(featset[0])
	means = [0 for i in range(feats)]
	maxes = [-9999 for i in range(feats)]
	mins  = [9999 for i in range(feats)]

	for i in range(rows):
		row = featset[i]
		for j in range(feats):
			means[j] += row[j] / rows
			maxes[j] = max(maxes[j], row[j])
			mins[j] = min(mins[j], row[j])

	newfeatset = []
	for i in range(rows):
		newfeatset.append([])
		row = featset[i]
		for j in range(feats):
			newfeatset[i].append( (row[j] - means[j]) / (maxes[j] - mins[j]) )

	return (newfeatset, labelset)

# split data into feature set, validation set and test set
def holdout(featxy):
	total = len(featxy[0])
	test = int(total * 0.2)
	valid = int(total * 0.35)

	feats = np.array(featxy[0])
	labels = np.array(featxy[1])

	TestFeats, ValidFeats, TrainFeats = np.split(feats, [test, valid])
	TestLabels, ValidLabels, TrainLabels = np.split(labels, [test, valid])

	return (TrainFeats, TrainLabels), (ValidFeats, ValidLabels), (TestFeats, TestLabels)

# run a tensorflow model on given sets and return accuracy
def evaluate(trainxy, validxy, testxy):
	keras.backend.clear_session()

	model = keras.Sequential()
	features = np.array(trainxy[0])
	model.add(keras.layers.InputLayer(input_shape = features.shape[1:]))
	model.add(keras.layers.Dense(features.shape[1], activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
	model.add(keras.layers.Dropout(0.3))
	model.add(keras.layers.Dense(2, activation='softmax'))
	model.compile(
		optimizer=tf.train.AdamOptimizer(learning_rate=0.002),
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

	# tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

	history = model.fit(
		trainxy[0], trainxy[1], None, 20, validation_data = validxy #, callbacks=[tbCallBack]
	)

	results = model.evaluate(testxy[0], testxy[1])
	return results[1] # test accuracy

# remove a feature from the set
def featstrip(trainxy, validxy, testxy, featid, n):
	feats = len(featid)

	newtrainfeat = []
	trainfeat = trainxy[0]
	for i in range(len(trainfeat)):
		row = []
		for j in range(feats):
			if j == n:
				continue
			row.append(trainfeat[i][j])
		newtrainfeat.append(row)
	newtrainxy = (np.array(newtrainfeat), trainxy[1])

	newvalidfeat = []
	validfeat = validxy[0]
	for i in range(len(validfeat)):
		row = []
		for j in range(feats):
			if j == n:
				continue
			row.append(validfeat[i][j])
		newvalidfeat.append(row)
	newvalidxy = (np.array(newvalidfeat), validxy[1])

	newtestfeat = []
	testfeat = testxy[0]
	for i in range(len(testfeat)):
		row = []
		for j in range(feats):
			if j == n:
				continue
			row.append(testfeat[i][j])
		newtestfeat.append(row)
	newtestxy = (np.array(newtestfeat), testxy[1])

	newfeatid = []
	for i in range(feats):
		if i == n:
			continue
		newfeatid.append(featid[i])

	return newtrainxy, newvalidxy, newtestxy, newfeatid

# repeatedly remove the worst feature. worst feature is highest acc model when feature removed
def featselect(trainxy, validxy, testxy, featid):
	print(f"Selecting from {len(featid)} features.")
	bestacc = evaluate(trainxy, validxy, testxy)
	bestidx = None

	for i in range(len(featid)):
		id = featid[i]
		print(f"Evaluating removal of feature {id}")
		trainxy2, validxy2, testxy2, featid2 = featstrip(trainxy, validxy, testxy, featid, i)
		stripacc = evaluate(trainxy2, validxy2, testxy2)
		if stripacc > bestacc:
			bestidx = i
			bestacc = stripacc
	if bestidx == None:
		print(f"Final accuracy: {bestacc}")
	else:
		trainxy2, validxy2, testxy2, featid2 = featstrip(trainxy, validxy, testxy, featid, bestidx)
		print(f"Removed feature: {featid2[bestidx]}")
		print(f"New accuracy: {bestacc}")
		if len(featid2) > 1:
			featselect(trainxy2, validxy2, testxy2, featid2)
dataset = None
with open("../users.json", "r") as datafile:
	dataset = json.load(datafile)

print("Dataset loaded.")

prunset = prune(dataset)

print("Dataset pruned.")

featxy, featid = featurify(prunset)
featxy = standardize(featxy)

print("Features extracted")

trainxy, validationxy, testxy = holdout(featxy)

featselect(trainxy, validationxy, testxy, featid)
