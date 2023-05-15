from neural_network import *


def find_letter(letter):
    if letter[0][0] == 1:
        return 'a'
    elif letter[0][1] == 1:
        return 'b'
    elif letter[0][2] == 1:
        return 'c'
    elif letter[0][3] == 1:
        return 'd'
    elif letter[0][4] == 1:
        return 'e'
    elif letter[0][5] == 1:
        return 'f'
    elif letter[0][6] == 1:
        return 'g'
    elif letter[0][7] == 1:
        return 'h'
    elif letter[0][8] == 1:
        return 'i'
    elif letter[0][9] == 1:
        return 'j'
    elif letter[0][10] == 1:
        return 'k'
    elif letter[0][11] == 1:
        return 'l'
    elif letter[0][12] == 1:
        return 'm'
    elif letter[0][13] == 1:
        return 'n'
    elif letter[0][14] == 1:
        return 'o'
    elif letter[0][15] == 1:
        return 'p'
    elif letter[0][16] == 1:
        return 'q'
    elif letter[0][17] == 1:
        return 'r'
    elif letter[0][18] == 1:
        return 's'
    elif letter[0][19] == 1:
        return 't'
    elif letter[0][20] == 1:
        return 'u'
    elif letter[0][21] == 1:
        return 'v'
    elif letter[0][22] == 1:
        return 'w'
    elif letter[0][23] == 1:
        return 'x'
    elif letter[0][24] == 1:
        return 'y'
    elif letter[0][25] == 1:
        return 'z'


# XD
def find_letter_2(letter):
    for i in range(0, 25):
        if letter[0][i] == 1:
            return chr(i + 97)
    return '0'


if __name__ == '__main__':
    network = Network(None, None, None)
    network.create_cnn()
    network.training_testing_set()
    network.train()
    # network.testCatalog(".\\test-images", ".\\test-labels.txt)
    network.test(".\\uploaded-images/testImage.bmp")
    network.test(".\\uploaded-images/testImage2.bmp")
