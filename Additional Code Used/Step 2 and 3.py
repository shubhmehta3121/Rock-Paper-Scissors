import os
import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Activation, Convolution2D
# Define the path to the dataset folder
folder_path = 'Dataset'

# Define class labels for mapping
label = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "restart": 3
}

def get_model():
    # Create a SqueezeNet-based model with custom architecture
    model = Sequential([
        SqueezeNet(input_shape=(227, 227, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(len(label), (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model

dataset = []

# Load and preprocess images from the dataset folder
for directory in os.listdir(folder_path):
    path = os.path.join(folder_path, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

data, labels = zip(*dataset)
labels = list(label[i] for i in labels)
labels = to_categorical(labels, len(label))

model = get_model()

# Compile the model with appropriate optimizer and loss function
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fit the model to the preprocessed image data and labels
model.fit(np.array(data), np.array(labels), epochs=10)

# Save the trained model
model.save("rock-paper-scissors-model.h5")
