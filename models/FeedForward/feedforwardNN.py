import os
from tkinter import Image
from PIL import Image
import numpy as np
from numba import jit

@jit(parallel=True)
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
        self.bias4 = None
        self.weights4 = None
        self.weights5 = None
        self.bias5 = None
        self.bias6 = None
        self.weights6 = None
        self.error = None
        self.epochs = 2352
        self.hidden1_output = None
        self.hidden2_output = None
        self.hidden3_output = None
        self.hidden4_output = None
        self.hidden5_output = None
        self.predicted_output = None
        self.trainLabels = []
        self.testLabels = []
        self.trainSet = []
        self.testSet = []
        self.current_set = None
        self.learning_rate = 0.0001

    def initialize_parameters(self):
        input_size = self.trainSet.shape[1]
        hidden1_size = 960
        hidden2_size = 480
        hidden3_size = 240
        hidden4_size = 120
        hidden5_size = 60
        output_size = 26

        self.weights1 = np.random.randn(input_size, hidden1_size)
        self.bias1 = np.zeros(hidden1_size)
        self.weights2 = np.random.randn(hidden1_size, hidden2_size)
        self.bias2 = np.zeros(hidden2_size)
        self.weights3 = np.random.randn(hidden2_size, hidden3_size)
        self.bias3 = np.zeros(hidden3_size)
        self.weights4 = np.random.randn(hidden3_size, hidden4_size)
        self.bias4 = np.zeros(hidden4_size)
        self.weights5 = np.random.randn(hidden4_size, hidden5_size)
        self.bias5 = np.zeros(hidden5_size)
        self.weights6 = np.random.randn(hidden5_size, output_size)
        self.bias6 = np.zeros(output_size)

    @jit(parallel=True)
    def forward(self):
        self.hidden1_output = sigmoid(np.dot(self.current_set, self.weights1) + self.bias1)
        self.hidden2_output = sigmoid(np.dot(self.hidden1_output, self.weights2) + self.bias2)
        self.hidden3_output = sigmoid(np.dot(self.hidden2_output, self.weights3) + self.bias3)
        self.hidden4_output = sigmoid(np.dot(self.hidden3_output, self.weights4) + self.bias4)
        self.hidden5_output = sigmoid(np.dot(self.hidden4_output, self.weights5) + self.bias5)
        self.predicted_output = sigmoid(np.dot(self.hidden5_output, self.weights6) + self.bias6)

    @jit(parallel=True)
    def backward(self):
        output_error = self.trainLabels - self.predicted_output
        self.weights6 += self.learning_rate * np.dot(self.hidden5_output.T, output_error)
        self.bias6 += self.learning_rate * np.sum(output_error, axis=0)

        hidden5_error = np.dot(output_error, self.weights6.T)
        self.weights5 += self.learning_rate * np.dot(self.hidden4_output.T, hidden5_error)
        self.bias5 += self.learning_rate * np.sum(hidden5_error, axis=0)

        hidden4_error = np.dot(hidden5_error, self.weights5.T)
        self.weights4 += self.learning_rate * np.dot(self.hidden3_output.T, hidden4_error)
        self.bias4 += self.learning_rate * np.sum(hidden4_error, axis=0)

        hidden3_error = np.dot(hidden4_error, self.weights4.T)
        self.weights3 += self.learning_rate * np.dot(self.hidden2_output.T, hidden3_error)
        self.bias3 += self.learning_rate * np.sum(hidden3_error, axis=0)

        hidden2_error = np.dot(hidden3_error, self.weights3.T)
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
            print(epoch)

    def predict(self):
        accuracy = 0
        self.current_set = self.testSet
        self.forward()
        predicted_labels = np.argmax(self.predicted_output, axis=1)
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == self.testLabels[i]:
                accuracy += 1
        print("Accuracy: ", accuracy / len(predicted_labels))

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
