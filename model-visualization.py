import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, AveragePooling2D
from tensorflow.keras import Model

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(90, 90, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(5, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=7e-4, decay=5e-6)

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

ixs = [0, 3, 6]
outputs = [model.layers[i].output for i in ixs]

model = Model(inputs=model.inputs, outputs=outputs)
print(model.summary())

x = cv2.imread(f"right/r4.png")
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
x = cv2.resize(x, (90, 90))
plt.imshow(x)
plt.show()

x = np.expand_dims(x, axis=0)

x = (x - 127.5) / 127.5

feature_maps = model.predict(x)
square = 4

for fmap in feature_maps:
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(fmap[0, :, :, ix - 1])
            ix += 1
