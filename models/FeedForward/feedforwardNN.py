import os
from tkinter import Image
from PIL import Image
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


class FNN:
    def __init__(self):
        self.b1 = None
        self.w1 = None
        self.b2 = None
        self.w2 = None
        self.w3 = None
        self.b3 = None
        self.error = None
        self.epochs = 6000
        self.h1_output = None
        self.h2_output = None
        self.predicted_output = None
        self.train_labels = []
        self.test_labels = []
        self.train_set = []
        self.test_set = []
        self.current_set = None
        self.learning_rate = 0.00001

    def initialize_parameters(self):
        input_size = self.train_set.shape[1]
        h1_size = 128
        h2_size = 64
        output_size = 26

        self.w1 = np.random.randn(input_size, h1_size)
        self.b1 = np.zeros(h1_size)
        self.w2 = np.random.randn(h1_size, h2_size)
        self.b2 = np.zeros(h2_size)
        self.w3 = np.random.randn(h2_size, output_size)
        self.b3 = np.zeros(output_size)

    def forward(self):
        self.h1_output = sigmoid(np.dot(self.current_set, self.w1) + self.b1)
        self.h2_output = sigmoid(np.dot(self.h1_output, self.w2) + self.b2)
        self.predicted_output = sigmoid(np.dot(self.h2_output, self.w3) + self.b3)

    def backward(self):
        output_error = self.train_labels - self.predicted_output
        self.w3 += self.learning_rate * np.dot(self.h2_output.T, output_error)
        self.b3 += self.learning_rate * np.sum(output_error, axis=0)

        h2_error = np.dot(output_error, self.w3.T)
        self.w2 += self.learning_rate * np.dot(self.h1_output.T, h2_error)
        self.b2 += self.learning_rate * np.sum(h2_error, axis=0)

        h1_error = np.dot(h2_error, self.w2.T)
        self.w1 += self.learning_rate * np.dot(self.train_set.T, h1_error)
        self.b1 += self.learning_rate * np.sum(h1_error, axis=0)

    def train(self):
        self.current_set = self.train_set
        for epoch in range(self.epochs):
            self.forward()
            self.backward()
            if epoch % 100 == 0:
                self.predict()
                print(epoch)
                # tu zrobic wykres

    def predict(self):
        accuracy = 0
        self.current_set = self.test_set
        self.forward()
        predicted_labels = np.argmax(self.predicted_output, axis=1)
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == self.test_labels[i]:
                accuracy += 1
        print("Accuracy: ", accuracy / len(predicted_labels))

        self.current_set = self.train_set

    def load_sets(self, directory, is_train_set):
        imagesArr = []
        labels = []
        image_size = (28, 28)
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                image = Image.open(file_path)
                image = image.resize(image_size)
                image = image.convert("L")
                image = np.array(image) / 255.0
                image_array = np.array(image).flatten()
                imagesArr.append(image_array)
                label = ord(os.path.basename(root)) - ord('a')
                labels.append(label)

        if is_train_set:
            self.train_set = np.array(imagesArr)
            self.train_labels = np.array(labels)
            self.initialize_parameters()
            self.prepare_train_labels()
        else:
            self.test_set = np.array(imagesArr)
            self.test_labels = np.array(labels)

    def prepare_train_labels(self):
        encoded_labels = np.zeros((len(self.train_labels), 26))
        for i, label in enumerate(self.train_labels):
            encoded_labels[i, label] = 1
        self.train_labels = encoded_labels
