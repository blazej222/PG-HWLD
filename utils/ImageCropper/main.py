import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_black_letter(image, margin, threshold):
    # Invert the image (black letter on white background)
    inverted = cv2.bitwise_not(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary mask
    _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find the non-zero pixels in the thresholded image
    points = cv2.findNonZero(thresholded)

    # Create a rectangular bounding box around the contour
    x, y, w, h = cv2.boundingRect(points)

    # Calculate the maximum side length for the square
    max_side = max(w, h)

    # Calculate the center coordinates of the square
    cx = x + w // 2
    cy = y + h // 2

    # Calculate half the side length of the square
    half_side = max_side // 2

    # Calculate the new bounding box coordinates for the square
    x = cx - half_side
    y = cy - half_side
    w = h = max_side

    # Add margin to the bounding rectangle
    x -= margin
    y -= margin
    w += 2 * margin
    h += 2 * margin

    # Fix overflows
    if x + w > image.shape[0]:
        x -= (x + w - image.shape[0])
    if y + h > image.shape[1]:
        y -= (y + h - image.shape[1])

    x = max(x, 0)
    y = max(y, 0)

    if x + w > image.shape[0] or y + h > image.shape[1]:
        x = 0
        y = 0
        w = min(image.shape[0], image.shape[1])
        h = w

    # Crop the image using the bounding box coordinates
    cropped = image[y:y + h, x:x + w]

    if cropped.shape[0] != cropped.shape[1]:
        print(cropped.shape)
        print(w, h)
        imgplot = plt.imshow(cropped)
        plt.show()

    return cropped


def crop_black_letters_catalog(catalog, margin, threshold):
    for address, dirs, files in os.walk(catalog):
        for file in files:
            image = cv2.imread(os.path.join(address, file))
            cropped_image = crop_black_letter(image, margin, threshold)
            cv2.imwrite(os.path.join(address, file), cropped_image)

        for directory in dirs:
            crop_black_letters_catalog(directory, margin, threshold)


def main():
    #FIXME: Change paths
    crop_black_letters_catalog("..\\dataset-black-marker", margin=5, threshold=100)
    print("Cropped images saved successfully.")


if __name__ == '__main__':
    main()
