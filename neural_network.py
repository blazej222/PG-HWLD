import os

import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.saving.legacy.model_config import model_from_json
from keras.utils import load_img, img_to_array
import main


class Network:
    def __init__(self, cnn, trainSet, testSet):
        self.cnn = cnn
        self.trainSet = trainSet
        self.testSet = testSet

    def create_cnn(self):
        cnn = Sequential(
            [Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 3)),
             MaxPooling2D(2, 2),
             Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 3)),
             MaxPooling2D(2, 2),
             Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 3)),
             MaxPooling2D(2, 2),
             Flatten(),
             Dense(units=512, activation='relu'),
             Dense(units=26, activation='softmax')])
        cnn.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        cnn.summary()
        self.cnn = cnn

    def training_testing_set(self):
        trainDataGen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        self.trainSet = trainDataGen.flow_from_directory(
            directory='train-images',
            target_size=(28, 28),
            batch_size=64,
            class_mode='categorical')

        testDataGen = ImageDataGenerator(rescale=1. / 255)
        self.testSet = testDataGen.flow_from_directory(
            directory='test-images',
            target_size=(28, 28),
            batch_size=64)


    def train(self):
        #if json exist
        if os.path.exists("cnn.json") and os.path.exists("cnn.h5"):
            with open("cnn.json", "r") as json_file:
                loaded_model_json = json_file.read()
                self.cnn = model_from_json(loaded_model_json)

            self.cnn.load_weights("cnn.h5")
            return

        #json does not exist
        self.cnn.fit(self.trainSet,
                steps_per_epoch=1950,
                epochs=3,
                validation_data=self.testSet)

        model_json = self.cnn.to_json()
        with open("cnn.json", "w") as json_file:
            json_file.write(model_json)
        self.cnn.save_weights("cnn.h5")

    def test(self):
        file = r'test-images/train_0.bmp'
        img = load_img(file, target_size=(28, 28))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = np.vstack([img])
        letter = self.cnn.predict(img)
        letter = main.find_letter(letter)
        print("Letter: " + letter)