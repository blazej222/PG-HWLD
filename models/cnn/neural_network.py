import ntpath
import os

import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.saving.legacy.model_config import model_from_json
from keras.utils import load_img, img_to_array
from keras.regularizers import l2
import multiprocessing as mp

import sys  # required to make imports from another directory work
# sys.path.append('../../utils')
from utils.DataSetConverter import main as dsc


def test_file(catalog, file, isLabelFile, test_by_folder, labelArray, label, doPrint, cnn, i):
    correctCount = 0
    incorrectCount = 0
    tmpath = os.path.join(catalog, file)
    tmpath = os.path.normpath(tmpath)
    img = load_img(tmpath, target_size=(28, 28))
    imgArray = img_to_array(img)
    imgArray = np.expand_dims(imgArray, axis=0)
    imgArray = np.vstack([imgArray])
    predictedLetter = cnn.predict(imgArray, verbose=0)
    predictedLetter = find_letter(predictedLetter)
    if isLabelFile:
        actualLetter = chr(labelArray[i] + 97)
    elif test_by_folder:
        actualLetter = ntpath.basename(catalog)[0]
    else:
        actualLetter = label

    if doPrint:
        print("Predicted: " + predictedLetter + " Actual: " + actualLetter)

    if predictedLetter == actualLetter:
        correctCount += 1
    else:
        # img.show()
        incorrectCount += 1

    return [correctCount, incorrectCount]


def find_letter(letter):
    max_prob = 0
    index = -1
    for i in range(0, 26):
        if letter[0][i] > max_prob:
            max_prob = letter[0][i]
            index = i
    if index == -1:
        print("find_letter() error")
    return chr(index + 97)


class Network:
    def __init__(self, cnn, trainSet, testSet):
        self.cnn = cnn
        self.trainSet = trainSet
        self.testSet = testSet

    def create_cnn(self):
        cnn = Sequential(
            [Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
             MaxPooling2D(2, 2),
             Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 3)),
             MaxPooling2D(2, 2),
             Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 3)),
             MaxPooling2D(2, 2),
             Flatten(),
             Dense(units=512, activation='relu'),
             Dense(units=26, activation='softmax')])

        cnn.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        cnn.summary()
        self.cnn = cnn

    def training_testing_set(self, train_catalog, test_catalog):
        trainDataGen = ImageDataGenerator(rescale=1. / 255)
        self.trainSet = trainDataGen.flow_from_directory(
            directory=train_catalog,
            target_size=(28, 28),
            batch_size=64,
            class_mode='categorical')

        testDataGen = ImageDataGenerator(rescale=1. / 255)
        self.testSet = testDataGen.flow_from_directory(
            directory=test_catalog,
            target_size=(28, 28),
            batch_size=64)

    def train(self):
        # if json exists
        if os.path.exists("cnn.json") and os.path.exists("cnn.h5"):
            with open("cnn.json", "r") as json_file:
                loaded_model_json = json_file.read()
                self.cnn = model_from_json(loaded_model_json)

            self.cnn.load_weights("cnn.h5")
            return

        # json does not exist
        self.cnn.fit(self.trainSet,
                     # steps_per_epoch=1950,
                     epochs=3,
                     # validation_data=self.testSet
                     )

        model_json = self.cnn.to_json()
        with open("cnn.json", "w") as json_file:
            json_file.write(model_json)
        self.cnn.save_weights("cnn.h5")

    def test(self, fileName):
        img = load_img(fileName)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = np.vstack([img])
        predictedLetter = self.cnn.predict(img, verbose=0)
        predictedLetter = find_letter(predictedLetter)
        print("Predicted letter: " + predictedLetter)

    #  TODO: Check if there's a cleaner way to implement this
    def test_catalog(self, catalog, labelFileName=None, label=None, doPrint=False, test_by_folder=False):
        i = 0
        correctCount = 0
        incorrectCount = 0
        isLabelFile = None
        labelArray = 0

        files = sorted(os.listdir(catalog), key=len)

        if labelFileName is not None:
            labelArray = dsc.txt_labelFile_to_array(labelFileName, len(files))
            isLabelFile = True
        elif label is not None or test_by_folder is True:
            isLabelFile = False
        else:
            print("Invalid arguments to test_catalog")
            return

        print("Starting catalog tests")

        pool = mp.Pool()

        if test_by_folder is True:
            for address, directories, files in os.walk(catalog):
                process = pool.starmap_async(test_file, [(address, x, isLabelFile, test_by_folder, labelArray, label,
                                                          doPrint, self.cnn, files.index(x)) for x in files])
        else:
            process = pool.starmap_async(test_file, [(catalog, x, isLabelFile, test_by_folder, labelArray, label,
                                                      doPrint, self.cnn, files.index(x)) for x in files])

        pool.close()
        pool.join()

        result = process.get()

        for a, b in result:
            correctCount += a
            incorrectCount += b

        print("Accuracy: " + str(correctCount / (correctCount + incorrectCount)))
