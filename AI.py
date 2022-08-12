import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, AveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time

#
# NOTE: Until you don't have your own dataset with game images, I don't recommend to touch anything in that file!
#

labels = {"down": 1, "up": 2, "left": 3, "right": 4}
labels2 = {"n": 0, "d": 1, "u": 2, "l": 3, "r": 4}
training_data = []

# Image width and height
WIDTH, HEIGHT = 90, 90


# Creating dataset from folders of labels
def create_training_data():

    print("Creating a data...")

    # For each label
    for label in labels.keys():

        print(f"Getting a {label} images...")

        # For each image
        for img in os.listdir(f"{label}/"):

            # Reading image as values
            img_array = cv2.imread(f"{label}/{img}")
            # Changing color of the image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            # Resizing the image
            new_array = cv2.resize(img_array, (WIDTH, HEIGHT))
            # Adding image values to the dataset
            training_data.append([new_array, labels[label]])
    print("Done!")


# Creating dataset from "images" folder
def create_training_data2():

    # For each game
    for game in os.listdir(f"images/"):
        print(f"Getting images from {game[4:]} game...")

        # For each image
        for img in os.listdir(f"images/{game}/"):

            # Reading image as values
            img_array = cv2.imread(f"images/{game}/{img}")
            # Changing color of the image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            # Resizing the image
            new_array = cv2.resize(img_array, (WIDTH, HEIGHT))
            # Adding image values to the dataset
            training_data.append([new_array, labels2[img[-5]]])


# Balancing data
def balance_data(X_data, y_data, max_size):
    bal_X = []
    bal_y = []
    nones, downs, ups, lefts, rights = 0, 0, 0, 0, 0

    for X, y in zip(X_data, y_data):
        if y == 0:
            nones += 1
            if nones <= max_size:
                bal_X.append(X)
                bal_y.append(y)
        elif y == 1:
            downs += 1
            if downs <= max_size:
                bal_X.append(X)
                bal_y.append(y)
        elif y == 2:
            ups += 1
            if ups <= max_size:
                bal_X.append(X)
                bal_y.append(y)
        elif y == 3:
            lefts += 1
            if lefts <= max_size:
                bal_X.append(X)
                bal_y.append(y)
        elif y == 4:
            rights += 1
            if rights <= max_size:
                bal_X.append(X)
                bal_y.append(y)

    bal_X, bal_y = np.array(bal_X), np.array(bal_y)

    return bal_X, bal_y


# Saving dataset to the file
def save_data(data, file):
    print("Saving data...")
    pickle_out = open(file, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    print("Done!")


# Reading dataset from the file
def read_data(file):
    pickle_in = open(file, "rb")
    data = pickle.load(pickle_in)
    return data


# Preparing data to training
def prepare_data(dataName):
    train_data = read_data(dataName)

    # Shuffling samples
    train_data = sklearn.utils.shuffle(train_data)

    X = []
    y = []

    # Diving dataset on images and labels
    for sample in train_data:
        X.append(sample[0])
        y.append(sample[1])

    X = np.array(X)
    y = np.array(y)

    # Scaling values
    X = X / 255.0

    # Size of the testing dataset
    test_size = 15000

    # Max size of the label samples in the training data
    maxLabelSize = 15000

    # Diving dataset on training dataset and testing dataset
    X_train = X[:-test_size]
    y_train = y[:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    # How many samples from each label are in the training dataset
    for x in range(5):
        print(f"{x}: {np.count_nonzero(y_train == x)}")

    # Balancing training data
    X_train, y_train = balance_data(X_train, y_train, maxLabelSize)

    return X_train, y_train, X_test, y_test


# create_training_data2()
# save_data(training_data, "dataset-filename") / Change "dataset-filename" in order to save dataset

# X_train, y_train, X_test, y_test = prepare_data("data-name") / Change "data-name" on dataset filename in order to prepare data
# save_data(X_train, "filename") / Change "filename" in order to save X_train
# save_data(y_train, "filename") / Change "filename" in order to save y_train
# save_data(X_test, "filename") / Change "filename" in order to save X_test
# save_data(y_test, "filename") / Change "filename" in order to save y_test

# If you've saved X's and y's read it!
# X_train, y_train, X_test, y_test = read_data("filename"), read_data("filename"), read_data("filename"), read_data("filename")

# Neural Network
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(WIDTH, HEIGHT, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(5, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=7e-4, decay=5e-6)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


NAME = f"Subway_Surfers_AI-{int(time.time())}"

# Logs
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

# Checkpoints of model during the training
# Change "folder-name", best on the model architecture
filepath = "folder-name/Subway_Surfers_AI-{epoch:02d}-{val_accuracy:.3f}"
checkpoint_dir = os.path.dirname(filepath)
checkpoint = ModelCheckpoint(filepath=filepath, save_weights_only=True, verbose=1)

# Neural Network architecture etc.
print(model.summary())

# Uncomment the following line in order to train model
# |
# |
# V

# model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32, callbacks=[tensorboard, checkpoint])





















