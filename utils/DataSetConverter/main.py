import struct
import numpy as np
from PIL import Image
import os
import argparse


def array_to_images(imageFileName, location, flip_colors):
    """
    Converts an array of images to actual image files.

    Args:
    - imageFileName (str): The file name of the array of images.
    - location (str): The directory where the images will be saved.
    - flip_colors (bool): The flag to flip image colors.

    Returns:
    - numImages (int): Number of images processed.
    """
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
        if flip_colors:
            for pixel_no in range(im.size):
                im[pixel_no] = 255 - im[pixel_no]

        im = im.reshape(28, 28)

        im = Image.fromarray(im)
        im = im.rotate(270)
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im.save(location + "/%s" % label + "/" + imageFileName.split(os.path.sep)[-1] + "_%s.png" % image, "png")

    imageFile.close()

    return numImages


def array_to_images_sorted(imageFileName, location, labelFileName, flip_colors):
    """
    Converts an array of images to actual image files sorted into labeled folders.

    Args:
    - imageFileName (str): The file name of the array of images.
    - location (str): The directory where the images will be saved.
    - labelFileName (str): The file name containing labels for the images.
    - flip_colors (bool): The flag to flip image colors.

    Returns:
    - numImages (int): Number of images processed.
    """
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
        if flip_colors:
            for pixel_no in range(im.size):
                im[pixel_no] = 255 - im[pixel_no]

        im = im.reshape(28, 28)

        im = Image.fromarray(im)
        im = im.rotate(270)
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

        label = chr(labelArray[image] + 97)

        im.save(location + "/%s" % label + "/" + imageFileName.split(os.path.sep)[-1] + "_%s.png" % image, "png")

    imageFile.close()

    return numImages


def labelFile_to_txt(labelFileName, outputFileName, numImages):
    """
    Converts a label file to a text file.

    Args:
    - labelFileName (str): The file name containing labels.
    - outputFileName (str): The output file name for the converted labels.
    - numImages (int): Number of labels to process.
    """
    labelFile = open(labelFileName, 'rb')
    labelFile.read(8)  # discard header info
    labelArray = np.array([], dtype='uint8')

    for label in range(0, numImages):
        letter = ord(labelFile.read(1))
        labelArray = np.append(labelArray, letter - 1)

    labelFile.close()
    np.savetxt(outputFileName, X=labelArray, fmt="%d")


def txt_labelFile_to_array(txtFileName, numImages):
    """
    Converts a text file containing labels to a numpy array.

    Args:
    - txtFileName (str): The file name containing labels.
    - numImages (int): Number of labels to process.

    Returns:
    - labelArray (numpy.array): Numpy array containing labels.
    """
    txtFile = open(txtFileName, 'rb')
    labelArray = np.array([], dtype=int)

    for label in range(0, numImages):
        letter = int(txtFile.readline())
        labelArray = np.append(labelArray, letter)

    txtFile.close()
    return labelArray


def main():
    parser = argparse.ArgumentParser(
        description='Convert dataset from idx3-ubyte format to extracted_undivided format.')
    parser.add_argument('--source_dataset', type=str, required=True,
                        help='Dataset source file in idx3-ubyte format.')
    parser.add_argument('--source_labels', type=str, required=True,
                        help='Labels file in idx1-ubyte format.')
    parser.add_argument('--destination', type=str, required=True,
                        help='Dataset destination directory.')
    parser.add_argument('--flip_colors', action='store_true',
                        help='Flip colors (on EMINST results in black-on-white).')
    args = parser.parse_args()

    filename = args.source_dataset
    labelFile = args.source_labels
    location = args.destination
    flip_colors = args.flip_colors

    array_to_images_sorted(filename, location, labelFile, flip_colors)


if __name__ == '__main__':
    main()
