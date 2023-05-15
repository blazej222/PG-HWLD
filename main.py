from neural_network import *


def find_letter(letter):
    for i in range(0, 26):
        if letter[0][i] == 1:
            return chr(i + 97)
    return '0'


if __name__ == '__main__':
    network = Network(None, None, None)
    network.create_cnn()
    network.training_testing_set()
    network.train()
    # network.testCatalog(".\\test-images", ".\\test-labels.txt")
    network.testCatalog(".\\imageTransformer/dataset/Testing/a", label='a', doPrint=True)
    # network.test(".\\uploaded-images/testImage.bmp")
    # network.test(".\\uploaded-images/testImage2.bmp")
