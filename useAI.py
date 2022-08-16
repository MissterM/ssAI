import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import keyboard
import pyautogui
import cv2
import numpy as np
import time
import mouse

#
# NOTE: In order to use AI, you have to change some values next to comments!
# NOTE: Best way to use this AI is to play Subway Surfers on BlueStacks on maximized window and hidden right sidebar.
#


class Model:
    def __init__(self, width, height, first, second, third, fourth, filepath):
        self.model = None
        self.width = width
        self.height = height
        self.first = first
        self.second = second
        self.third = third
        self.fourth = fourth
        self.filepath = filepath

    def create_model(self):

        self.model = Sequential()

        self.model.add(Conv2D(self.first, (3, 3), activation="relu", input_shape=(self.width, self.height, 3)))
        self.model.add(MaxPooling2D(2, 2))

        self.model.add(Conv2D(self.second, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D(2, 2))

        self.model.add(Conv2D(self.third, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D(2, 2))

        self.model.add(Flatten())

        self.model.add(Dense(self.fourth, activation="relu"))

        self.model.add(Dense(5, activation="softmax"))

        opt = tf.keras.optimizers.Adam(learning_rate=7e-4, decay=5e-6)

        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        self.model.load_weights(self.filepath)

        return self.model


Alex = Model(90, 90, 32, 32, 32, 128, "Alex/Subway_Surfers_AI-13-0.830")
John = Model(100, 100, 32, 32, 32, 128, "John/Subway_Surfers_AI-10-0.832")
Andrew = Model(100, 100, 16, 32, 64, 128, "Andrew/Subway_Surfers_AI-08-0.825")
Max = Model(90, 90, 16, 32, 64, 128, "Max/Subway_Surfers_AI-14-0.830")
FirstGame = Model(90, 90, 32, 32, 16, 256, "Model from First Game/Subway_Surfers_AI-09-0.826")
SecondGame = Model(90, 90, 32, 32, 32, 128, "Model from Second Game/Subway_Surfers_AI-05-0.803")





# Choose what model do you want to use, default: "Andrew"
modelName = Andrew







model = modelName.create_model()


# Don't change anything there
def acceleration(dif):
    acc = -0.0000017 * (dif - 500) ** 2 + 0.29

    if acc > 0.23:
        acc = 0.23

    return acc


labels = {0: "None", 1: "Down", 2: "Up", 3: "Left", 4: "Right"}
moves = {1: "down_arrow", 2: "up_arrow", 3: "left_arrow", 4: "right_arrow"}
playing = False
moved = True
sswait = time.time()
acc = 0

#
# NOTE: If your AI doesn't work, change following values
#

while True:

    #
    # If program is pressing buttons while you are not in game, change values "703" and "85" on coordinates of any pixel in the blue pause button in the up-left corner.
    #

    if pyautogui.pixelMatchesColor(703, 85, (90, 170, 210), tolerance=20):
        moved = False
        acc = 0
        if not playing:
            start = time.time()
            time.sleep(2.5)
            playing = True

        #
        # If you are not using that AI on the 1920x1080 screen, change "678, 33, 564, 1007" values like on the "subway_surfers_size.png" file
        #

        pyautogui.screenshot("screenshot.png", region=(678, 33, 564, 1007))
        img_array = cv2.imread("screenshot.png")
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        new_array = cv2.resize(img_array, (modelName.width, modelName.height))
        X = np.array(new_array)
        X = (X-127.5)/127.5
        X = np.expand_dims(X, axis=0)
        np.set_printoptions(suppress=True)
        y = model.predict(X)
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
        start = 0

        # Mouse automation to start games by itself

        # if not moved:
        #     mouse.move(0, -150, absolute=False, duration=2.5)
        #     mouse.click("left")
        #     mouse.move(0, 150, absolute=False, duration=2)
        #
        #     mouse.move(0, -150, absolute=False, duration=1.5)
        #     mouse.click("left")
        #     mouse.move(0, 150, absolute=False, duration=1)
        #
        #     mouse.move(0, -150, absolute=False, duration=2)
        #     mouse.click("left")
        #     mouse.move(0, 150, absolute=False, duration=1)
        #
        #     mouse.move(0, -150, absolute=False, duration=1.5)
        #     mouse.click("left")
        #     mouse.move(0, 150, absolute=False, duration=1)
        #
        #     mouse.move(0, -150, absolute=False, duration=2)
        #     mouse.click("left")
        #     mouse.move(0, 150, absolute=False, duration=1)
        #
        #     pyautogui.screenshot(f"scores/{model_architecture}_{time.time()}.png", region=(678, 33, 564, 1007))
        #
        #     mouse.move(0, -50, absolute=False, duration=0.3)
        #     mouse.click("left")
        #     mouse.move(0, 50, absolute=False, duration=0.3)
        #
        #     moved = True






