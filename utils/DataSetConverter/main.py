import struct
import numpy as np
from PIL import Image
import os


def array_to_images(imageFileName, location):
    imageFile = open(imageFileName, 'rb')
    buf = imageFile.read()

    if not os.path.exists(location):
        os.makedirs(location)

    index = 0
    # read 4 unsigned int with big-endian format
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')  # move the cursor

    for image in range(0, numImages):
        # the image is 28*28=784 unsigned chars
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')  # move the cursor

        # create a np array to save the image
        im = np.array(im, dtype='uint8')

        # flip colors
        for pixel_no in range(im.size):
            im[pixel_no] = 255 - im[pixel_no]

        im = im.reshape(28, 28)

        im = Image.fromarray(im)
        im = im.rotate(270)
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im.save(location + "/train_%s.bmp" % image, "bmp")

    imageFile.close()

    return numImages


def array_to_images_sorted(imageFileName, location, labelFileName):
    imageFile = open(imageFileName, 'rb')
    buf = imageFile.read()

    if not os.path.exists(location):
        os.makedirs(location)

    for letter in range(26):
        folder = (location + "/%s" % chr(letter + 97))
        if not os.path.exists(folder):
            os.makedirs(folder)

    index = 0
    # read 4 unsigned int with big-endian format
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')  # move the cursor

    txtFileName = location + "/labels.txt"

    labelFile_to_txt(labelFileName, txtFileName, numImages)
    labelArray = txt_labelFile_to_array(txtFileName, numImages)

    for image in range(0, numImages):
        # the image is 28*28=784 unsigned chars
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')  # move the cursor

        # create a np array to save the image
        im = np.array(im, dtype='uint8')

        # flip colors
        for pixel_no in range(im.size):
            im[pixel_no] = 255 - im[pixel_no]

        im = im.reshape(28, 28)

        im = Image.fromarray(im)
        im = im.rotate(270)
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

        label = chr(labelArray[image] + 97)

        im.save(location + "/%s" % label + "/train_%s.bmp" % image, "bmp")

    imageFile.close()

    return numImages


def labelFile_to_txt(labelFileName, outputFileName, numImages):
    labelFile = open(labelFileName, 'rb')
    labelFile.read(8)  # discard header info
    labelArray = np.array([], dtype='uint8')

    for label in range(0, numImages):
        letter = ord(labelFile.read(1))
        labelArray = np.append(labelArray, letter - 1)

    labelFile.close()
    np.savetxt(outputFileName, X=labelArray, fmt="%d")


def txt_labelFile_to_array(txtFileName, numImages):
    txtFile = open(txtFileName, 'rb')
    labelArray = np.array([], dtype=int)

    for label in range(0, numImages):
        letter = int(txtFile.readline())
        labelArray = np.append(labelArray, letter)

    txtFile.close()
    return labelArray


def main():
    filename = "../../resources/datasets/archives/EMNIST-binary/emnist-letters-train-images-idx3-ubyte"
    labelFile = "../../resources/datasets/archives/EMNIST-binary/emnist-letters-train-labels-idx1-ubyte"
    location = "../../resources/datasets/dataset-EMNIST/train-images"
    #array_to_images_sorted(filename, location, labelFile)

    filename = "../../resources/datasets/archives/EMNIST-binary/emnist-letters-test-images-idx3-ubyte"
    labelFile = "../../resources/datasets/archives/EMNIST-binary/emnist-letters-test-labels-idx1-ubyte"
    location = "../../resources/datasets/dataset-EMNIST/test-images"
    #numImages = array_to_images(filename, location)
    #labelFile_to_txt(labelFile, "../../resources/datasets/dataset-EMNIST/test-labels.txt", numImages)
    array_to_images_sorted(filename, location, labelFile)


if __name__ == '__main__':
    main()
