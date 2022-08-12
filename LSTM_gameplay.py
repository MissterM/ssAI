import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ConvLSTM2D, MaxPooling3D, BatchNormalization
import keyboard
import pyautogui
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import mouse

#
# NOTE: Since LSTM models weren't doing well, I recommend you using "gameplay.py" file to use that AI!
#

def create_model():
    model = Sequential()
    model.add(ConvLSTM2D(16, (3, 3), activation="tanh", input_shape=(2, 90, 90, 3), return_sequences=True))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding="same"))

    model.add(ConvLSTM2D(32, (3, 3), activation="tanh", padding="same", return_sequences=True))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding="same"))

    model.add(ConvLSTM2D(64, (3, 3), activation="tanh", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))

    model.add(Dense(5, activation="softmax"))

    opt = tf.keras.optimizers.Adam(learning_rate=7e-4, decay=5e-6)

    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def acceleration(dif):
    acc = -0.00000029 * (dif - 1000) ** 2 + 0.29

    if acc > 0.29:
        acc = 0.29

    return acc


filepath = "convlstm-models-2img/Subway_Surfers_AI-07-0.786"

model = create_model()
model.load_weights(filepath)

labels = {0: "None", 1: "Down", 2: "Up", 3: "Left", 4: "Right"}
moves = {1: "down_arrow", 2: "up_arrow", 3: "left_arrow", 4: "right_arrow"}
playing = False
moved = False
first = True
second = True
cycle = 0
acc = 0


# batch_size == 3
# |
# V

# while True:
#     if pyautogui.pixelMatchesColor(703, 85, (90, 170, 210), tolerance=20):
#         moved = False
#         acc = 0
#         if not playing:
#             start = time.time()
#             time.sleep(3)
#             playing = True
#
#         if not cycle == 0:
#             new_array1, new_array2 = new_array2, new_array3
#
#         if first:
#             pyautogui.screenshot("screenshot1.png", region=(678, 33, 564, 1007))
#             img_array1 = cv2.imread("screenshot1.png")
#             img_array1 = cv2.cvtColor(img_array1, cv2.COLOR_BGR2RGB)
#             new_array1 = cv2.resize(img_array1, (90, 90))
#             first = False
#             time.sleep(0.2)
#
#         if second:
#             pyautogui.screenshot("screenshot2.png", region=(678, 33, 564, 1007))
#             img_array2 = cv2.imread("screenshot2.png")
#             img_array2 = cv2.cvtColor(img_array2, cv2.COLOR_BGR2RGB)
#             new_array2 = cv2.resize(img_array2, (90, 90))
#             second = False
#             time.sleep(0.2)
#
#         pyautogui.screenshot("screenshot3.png", region=(678, 33, 564, 1007))
#         img_array3 = cv2.imread("screenshot3.png")
#         img_array3 = cv2.cvtColor(img_array3, cv2.COLOR_BGR2RGB)
#         new_array3 = cv2.resize(img_array3, (90, 90))
#
#         X = np.array([new_array1, new_array2, new_array3])
#         X = (X - 127.5)/127.5
#         X = np.expand_dims(X, axis=0)
#         np.set_printoptions(suppress=True)
#         y = model.predict(X)
#         cycle += 1
#         print("\n"*25)
#         print(f"Choice: {labels[int(np.argmax(y))]}, Confidence: {round((np.max(y)*100), 2)}%")
#         y = np.argmax(y)
#         end = time.time()
#         difference = end - start
#         if not int(y) == 0:
#             keyboard.send(moves[int(y)])
#             if difference >= 90:
#                 acc = acceleration(difference)
#             time.sleep(0.24 - acc)
#
#     else:
#         playing = False

# batch_size == 2
# |
# V

while True:
    if pyautogui.pixelMatchesColor(703, 85, (90, 170, 210), tolerance=20):
        moved = False
        if not playing:
            start = time.time()
            time.sleep(1)
            playing = True

        if not cycle == 0:
            new_array1 = new_array2

        if first:
            pyautogui.screenshot("screenshot1.png", region=(678, 33, 564, 1007))
            img_array1 = cv2.imread("screenshot1.png")
            img_array1 = cv2.cvtColor(img_array1, cv2.COLOR_BGR2RGB)
            new_array1 = cv2.resize(img_array1, (90, 90))
            first = False
            time.sleep(0.2)

        pyautogui.screenshot("screenshot2.png", region=(678, 33, 564, 1007))
        img_array2 = cv2.imread("screenshot2.png")
        img_array2 = cv2.cvtColor(img_array2, cv2.COLOR_BGR2RGB)
        new_array2 = cv2.resize(img_array2, (90, 90))

        X = np.array([new_array1, new_array2])
        X = (X - 127.5)/127.5
        X = np.expand_dims(X, axis=0)
        np.set_printoptions(suppress=True)
        y = model.predict(X)
        cycle += 1
        print("\n"*25)
        print(f"Choice: {labels[int(np.argmax(y))]}, Confidence: {round((np.max(y)*100), 2)}%")
        y = np.argmax(y)
        end = time.time()
        difference = end - start
        if not int(y) == 0:
            keyboard.send(moves[int(y)])
            if difference >= 90:
                acc = acceleration(difference)
            time.sleep(0.3 - acc)
    else:
        playing = False





