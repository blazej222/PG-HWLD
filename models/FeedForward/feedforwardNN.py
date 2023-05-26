import os
from tkinter import Image
from PIL import Image
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class FNN:
    def __init__(self):
        self.bias1 = None
        self.weights1 = None
        self.bias2 = None
        self.weights2 = None
        self.weights3 = None
        self.bias3 = None
        self.error = None
        self.epochs = 2352
        self.hidden1_output = None
        self.hidden2_output = None
        self.predicted_output = None
        self.trainLabels = []
        self.testLabels = []
        self.trainSet = []
        self.testSet = []
        self.current_set = None
        self.learning_rate = 0.0001

    def initialize_parameters(self):
        input_size = self.trainSet.shape[1]
        hidden1_size = 130
        hidden2_size = 65
        output_size = 26

        self.weights1 = np.random.randn(input_size, hidden1_size)
        self.bias1 = np.zeros(hidden1_size)
        self.weights2 = np.random.randn(hidden1_size, hidden2_size)
        self.bias2 = np.zeros(hidden2_size)
        self.weights3 = np.random.randn(hidden2_size, output_size)
        self.bias3 = np.zeros(output_size)

    def forward(self):
        self.hidden1_output = sigmoid(np.dot(self.current_set, self.weights1) + self.bias1)
        self.hidden2_output = sigmoid(np.dot(self.hidden1_output, self.weights2) + self.bias2)
        self.predicted_output = sigmoid(np.dot(self.hidden2_output, self.weights3) + self.bias3)

    def backward(self):
        error = self.trainLabels - self.predicted_output

        self.weights3 += self.learning_rate * np.dot(self.hidden2_output.T, error)
        self.bias3 += self.learning_rate * np.sum(error, axis=0)

        hidden2_error = np.dot(error, self.weights3.T)
        self.weights2 += self.learning_rate * np.dot(self.hidden1_output.T, hidden2_error)
        self.bias2 += self.learning_rate * np.sum(hidden2_error, axis=0)

        hidden1_error = np.dot(hidden2_error, self.weights2.T)
        self.weights1 += self.learning_rate * np.dot(self.trainSet.T, hidden1_error)
        self.bias1 += self.learning_rate * np.sum(hidden1_error, axis=0)

    def train(self):
        self.current_set = self.trainSet
        for epoch in range(self.epochs):
            self.forward()
            self.backward()
            print("{} done.".format(epoch))

    def predict(self):
        accuracy = 0
        self.current_set = self.testSet
        self.forward()
        predicted_labels = np.argmax(self.predicted_output, axis=1)
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == self.testLabels[i]:
                accuracy += 1
        print("Accuracy: ", accuracy/len(predicted_labels))

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
            self.trainSet = np.array(imagesArr)
            self.trainLabels = np.array(labels)
            self.initialize_parameters()
            self.prepare_train_labels()
        else:
            self.testSet = np.array(imagesArr)
            self.testLabels = np.array(labels)

    def prepare_train_labels(self):
        encoded_labels = np.zeros((len(self.trainLabels), 26))
        for i, label in enumerate(self.trainLabels):
            encoded_labels[i, label] = 1
        self.trainLabels = encoded_labels

