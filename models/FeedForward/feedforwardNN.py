import os
import string
from tkinter import Image
from PIL import Image
import numpy as np
from matplotlib import pyplot


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


def softmax(values):
    exp_values = np.exp(values)
    exp_values_sum = np.sum(exp_values)
    return exp_values / exp_values_sum


class FNN:
    def __init__(self):
        self.b1 = None
        self.w1 = None
        self.b2 = None
        self.w2 = None
        self.w3 = None
        self.b3 = None
        self.error = None
        self.epochs = 7000
        self.h1_output = None
        self.h2_output = None
        self.predicted_output = None
        self.train_labels = []
        self.test_labels = []
        self.train_set = []
        self.test_set = []
        self.current_set = None
        self.learning_rate = 0.00001
        self.accuracy_arr = []
        self.epoch_arr = []
        self.loss = []

    def initialize_parameters(self):
        if len(self.train_labels) <= 10000:
            self.learning_rate = 0.00001
        elif 10000 < len(self.train_labels) <= 100000:
            self.learning_rate = 0.000001
        else:
            self.learning_rate = 0.0000001

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

        self.cross_entropy()

    def train(self):
        self.current_set = self.train_set
        for epoch in range(self.epochs):
            self.forward()
            self.backward()
            if epoch % 100 == 0:
                print("Epoch: ", epoch)
                self.predict(epoch, None)

    def cross_entropy(self):
        if np.array_equal(self.current_set, self.train_set):
            labels = self.train_labels
        else:
            labels = self.test_labels

        softmax(self.predicted_output)
        loss = 0
        for i in range(len(self.predicted_output)):
            loss += -1 * labels[i] * np.log(self.predicted_output[i])
        total_loss = loss.sum()
        self.loss.append(total_loss)

    def predict(self, epoch, dir_name):
        if epoch == -1:
            # print accuracy
            self.current_set = self.test_set
            self.forward()
            pyplot.plot(self.epoch_arr, self.accuracy_arr)
            pyplot.ylabel("Dokładność")
            pyplot.xlabel("Epoki")
            pyplot.grid()
            pyplot.title("Dokładność rozpoznania liter w zależności od epoki dla zbioru " + dir_name)
            pyplot.savefig(dir_name + "_accuracy_plot.png")
            pyplot.close()
            # print loss
            pyplot.plot(self.loss)
            pyplot.ylabel("Strata")
            pyplot.xlabel("Epoki")
            pyplot.grid()
            pyplot.title("Funkcja straty w zależności od epoki dla zbioru " + dir_name)
            pyplot.savefig(dir_name + "_loss_plot.png")
            pyplot.close()
            # print accuracy for every letter
            letters_accuracy = []
            predicted_labels = np.argmax(self.predicted_output, axis=1)
            correct = 0
            nb_of_letters = 0
            for i in range(len(predicted_labels)):
                nb_of_letters += 1
                if predicted_labels[i] == self.test_labels[i]:
                    correct += 1
                if i == (len(predicted_labels) - 1):
                    letters_accuracy.append(correct / nb_of_letters)
                else:
                    if self.test_labels[i] != self.test_labels[i + 1]:
                        letters_accuracy.append(correct / nb_of_letters)
                        correct = 0
                        nb_of_letters = 0

            alphabet = list(string.ascii_lowercase)
            pyplot.bar(alphabet, letters_accuracy)
            pyplot.title("Dokładność rozpoznania liter dla zbioru " + dir_name)
            pyplot.savefig(dir_name + "_letters_accuracy_plot.png")

        else:
            correct = 0
            self.current_set = self.test_set
            self.forward()
            predicted_labels = np.argmax(self.predicted_output, axis=1)
            for i in range(len(predicted_labels)):
                if predicted_labels[i] == self.test_labels[i]:
                    correct += 1
            accuracy = correct/len(predicted_labels)
            print("Accuracy: ", accuracy)
            self.accuracy_arr.append(accuracy)
            self.epoch_arr.append(epoch)
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
