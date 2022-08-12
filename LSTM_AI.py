import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, TimeDistributed, Input, ConvLSTM2D, MaxPooling3D, BatchNormalization, MaxPool3D, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time

#
# NOTE: Until you don't have your own dataset with game images, I don't recommend to touch anything in that file!
#

labels = {"none": 0, "down": 1, "up": 2, "left": 3, "right": 4}
labels2 = {"n": 0, "d": 1, "u": 2, "l": 3, "r": 4}
training_data = []


def create_training_data():
    print("Creating a data...")
    for label in labels.keys():
        print(f"Getting a {label} images...")
        for img in os.listdir(f"{label}/"):
            img_array = cv2.imread(f"{label}/{img}")
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            new_array = cv2.resize(img_array, (90, 90))
            training_data.append([new_array, labels[label]])


def create_training_data2():
    print("Creating a data...")
    for game in os.listdir(f"images/"):
        print(f"Getting images from {game[4:]} game...")
        cycle = 0
        for img in os.listdir(f"images/{game}/"):
            if cycle == 0:
                img_array = cv2.imread(f"images/{game}/{img}")
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (90, 90))
                training_data.append([new_array, labels2[img[-5]]])
            cycle += 1
            if cycle == 2:
                cycle = 0
        training_data.append("next game")


def save_data(data, file):
    print("Saving data...")
    pickle_out = open(file, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    print("Done!")


def read_data(file):
    pickle_in = open(file, "rb")
    data = pickle.load(pickle_in)
    return data


def prepare_data():
    train_data = read_data("training_data2.pickle")
    train_data = sklearn.utils.shuffle(train_data)

    X = []
    y = []

    for sample in train_data:
        X.append(sample[0])
        y.append(sample[1])

    X = np.array(X)
    y = np.array(y)
    X = (X - 127.5) / 127.5
    test_size = 2000

    X_train = X[:-test_size]
    y_train = y[:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    return X_train, y_train, X_test, y_test


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


def prepare_data2():
    train_data = read_data("image_data90k.pickle")

    X = []
    y = []

    for x in range(len(os.listdir(f"images/"))):
        batch = []
        batch_size = 0
        print(f"Taking data from game  {x}...")
        for sample in train_data[:train_data.index("next game")]:
            batch.append(sample[0])
            batch_size += 1
            if batch_size == 2:
                X.append(np.array(batch))
                y.append(sample[1])
                batch = []
                batch_size = 0
        train_data = train_data[(train_data.index("next game") + 1):]

    X = np.array(X)
    y = np.array(y)
    X = (X - 127.5) / 127.5
    print(X.shape)

    X = sklearn.utils.shuffle(X, random_state=0)
    y = sklearn.utils.shuffle(y, random_state=0)

    test_size = 4500

    X_train = X[:-test_size]
    y_train = y[:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    for x in range(5):
        print(f"{x}: {np.count_nonzero(y_train == x)}")

    X_train, y_train = balance_data(X_train, y_train, 6000)

    for x in range(5):
        print(f"{x}: {np.count_nonzero(y_train == x)}")

    return X_train, y_train, X_test, y_test


# create_training_data2()
# save_data(training_data, "image_data90k.pickle")

# X_train, y_train, X_test, y_test = prepare_data2()
# save_data(X_train, "X_train90k-lstm2.pickle")
# save_data(y_train, "y_train90k-lstm2.pickle")
# save_data(X_test, "X_test90k-lstm2.pickle")
# save_data(y_test, "y_test90k-lstm2.pickle")

X_train, y_train, X_test, y_test = read_data("X_train90k-lstm2.pickle"), read_data("y_train90k-lstm2.pickle"), read_data("X_test90k-lstm2.pickle"), read_data("y_test90k-lstm2.pickle")

model = Sequential()
model.add(ConvLSTM2D(16, (3, 3), activation="tanh", input_shape=(2, 90, 90, 3), return_sequences=True, dropout=0.2))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2), padding="same"))

model.add(ConvLSTM2D(32, (3, 3), activation="tanh", padding="same", dropout=0.2, return_sequences=True))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2), padding="same"))

model.add(ConvLSTM2D(64, (3, 3), activation="tanh", padding="same", dropout=0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dense(5, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=7e-4, decay=5e-6)

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

NAME = f"Subway_Surfers_AI-{int(time.time())}"

tensorboard = TensorBoard(log_dir=f"logs2/{NAME}")

filepath = "convlstm-models-2img/Subway_Surfers_AI-{epoch:02d}-{val_accuracy:.3f}"
checkpoint_dir = os.path.dirname(filepath)
checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_weights_only=True)

print(model.summary())
model.fit(X_train, y_train, epochs=9, validation_data=(X_test, y_test), batch_size=32, callbacks=[tensorboard, checkpoint])




