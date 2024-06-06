import ntpath
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from time import time
import argparse


def EMNISTify(image, threshold, verbose=False):
    """
    Processes an image to resemble EMNIST format by inverting colors, thresholding,
    blurring, cropping, centering, padding, and resizing.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        threshold (int): Threshold value for binary segmentation.
        verbose (bool): If True, prints debug information.

    Returns:
        numpy.ndarray: Processed image in EMNIST format.
    """
    # Invert the image (black on white to white on black)
    inverted = cv2.bitwise_not(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary mask
    _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Apply a Gaussian filter
    blurred = cv2.GaussianBlur(thresholded, (0, 0), 1, _, 1, cv2.BORDER_DEFAULT)

    # Find the non-zero pixels in the thresholded image
    points = cv2.findNonZero(blurred)

    # Create a rectangular bounding box around the contour
    x, y, w, h = cv2.boundingRect(points)
    if verbose:
        print(f"Cropped image boundary: {x, y, w, h}")

    # Extract ROI
    cropped = blurred[y:y + h, x:x + w]

    # Center image in the frame
    top_border = 0
    bottom_border = 0
    left_border = 0
    right_border = 0

    # Fit into a square if not ROI was not even-sided
    if w > h:
        to_add = w - h
        top_border = to_add // 2
        bottom_border = to_add // 2
        if to_add % 2 != 0:
            top_border += 1
    elif h > w:
        to_add = h - w
        left_border = to_add // 2
        right_border = to_add // 2
        if to_add % 2 != 0:
            left_border += 1

    if verbose:
        print(f"Width: {w}, Height: {h}, Top border: {top_border}, Bottom border: {bottom_border}, Left border: {left_border}, Right border: {right_border}")

    new_image = np.zeros((w + left_border + right_border, h + top_border + bottom_border, 1), dtype=np.uint8)
    for i in range(0, w + left_border + right_border):
        for j in range(0, h + top_border + bottom_border):
            if left_border <= i < left_border + w and top_border <= j < top_border + h:
                new_image[j][i] = cropped[j - top_border][i - left_border]
            else:
                new_image[j][i] = 0

    # Add 2 pixel padding
    padded = np.zeros((new_image.shape[0] + 4, new_image.shape[1] + 4, 1), dtype=np.uint8)
    for i in range(0, new_image.shape[0] + 4):
        for j in range(0, new_image.shape[0] + 4):
            if 2 <= i < new_image.shape[0] + 2 and 2 <= j < new_image.shape[1] + 2:
                padded[j][i] = new_image[j - 2][i - 2]
            else:
                padded[j][i] = 0

    # Resize the image to 28 x 28
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_CUBIC)

    # Scale intensity to [0, 255]
    processed = cv2.normalize(resized, _, 0, 255, cv2.NORM_MINMAX)

    return processed


def EMNISTify_file(address, file, threshold, destination, location, verbose=False):
    """
    Processes a single image file to apply the EMNISTify transformation.

    Args:
        address (str): Directory of the image file.
        file (str): Name of the image file.
        threshold (int): Threshold value for binary segmentation.
        destination (str): Path to the destination directory.
        location (str): Path to the source directory.
        verbose (bool): If True, prints debug information.
    """
    image = cv2.imread(str(os.path.join(address, file)))
    processed_image = EMNISTify(image, threshold,verbose)
    cv2.imwrite(str(os.path.join(destination + address.replace(location, ''), file)), processed_image)


def EMNISTify_catalog(location, destination, threshold=100, verbose=False):
    """
    Processes all image files in a directory to apply the EMNISTify transformation.

    Args:
        location (str): Path to the source directory.
        destination (str): Path to the destination directory.
        threshold (int, optional): Threshold value for binary segmentation. Default is 100.
        verbose (bool, optional): If True, prints debug information. Default is False.
    """
    if not os.path.exists(destination):
        os.makedirs(destination)

    for address, dirs, files in os.walk(location):
        for directory in dirs:
            subdirectory = address.replace(location, "")
            temp = destination + subdirectory
            if not os.path.exists(os.path.join(temp, directory)):
                os.makedirs(os.path.join(temp, directory))

        pool = mp.Pool()
        pool.starmap_async(EMNISTify_file, [(address, x, threshold, destination, location, verbose) for x in files])
        pool.close()
        pool.join()

    print(f"Cropping finished for catalog {location}")


def main():
    parser = argparse.ArgumentParser(description='Apply the EMNIST image processing steps to each sample.')
    parser.add_argument('--source', type=str, required=True,
                        help='Dataset source directory.')
    parser.add_argument('--destination', type=str, required=True,
                        help='Processed dataset destination directory.')
    parser.add_argument('--threshold', type=int, default=100,
                        help='Threshold value.')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Print debug info.')
    args = parser.parse_args()

    location = args.source
    destination = args.destination
    threshold = args.threshold
    verbose = args.verbose
    start = time()

    EMNISTify_catalog(location, destination, threshold, verbose)

    end = time()
    print(f"Finished in {end - start}")


if __name__ == '__main__':
    main()
