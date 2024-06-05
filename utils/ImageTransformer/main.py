import ntpath
import numpy as np
import cv2
import os
import multiprocessing as mp
import argparse


def image_transform(imageFile, denoise=True):
    """
    Transforms an image by converting it to grayscale, normalizing,
    optionally denoising, and resizing to 28x28 pixels.

    Args:
        imageFile (numpy.ndarray): Input image in BGR format.
        denoise (bool): Flag to determine whether to denoise the image.

    Returns:
        numpy.ndarray: Transformed image.
    """
    imageFile = cv2.cvtColor(imageFile, cv2.COLOR_BGR2GRAY)
    imageFile = cv2.normalize(imageFile, imageFile, 0, 255, cv2.NORM_MINMAX)
    if denoise:
        # imageFile = remove_shadows(imageFile)
        # imageFile = cv2.convertScaleAbs(imageFile, 2, 1);
        # imageFile = cv2.fastNlMeansDenoising(imageFile, imageFile, 60.0, 7, 21)
        imageFile = flat_denoise(imageFile, 190)
    imageFile = cv2.resize(imageFile, dsize=(28, 28), interpolation=cv2.INTER_NEAREST)
    return imageFile


def remove_shadows(image):
    """
    Removes shadows from an image by applying dilation and median blur,
    followed by normalization.

    Args:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Image with shadows removed.
    """
    transformedImg = cv2.dilate(image, np.ones((7, 7), np.uint8))
    transformedImg = cv2.medianBlur(transformedImg, 21)
    transformedImg = 255 - cv2.absdiff(image, transformedImg)
    cv2.normalize(transformedImg, transformedImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return transformedImg


# TODO: Check if eliminating reshapes speeds things up
def flat_denoise(image, threshold):
    """
    Denoises an image by setting pixels above a certain threshold to 255.

    Args:
        image (numpy.ndarray): Input grayscale image.
        threshold (int): Threshold above which pixels are set to 255.

    Returns:
        numpy.ndarray: Denoised image.
    """
    width = image.shape[1]
    height = image.shape[0]
    image = image.reshape(image.size)
    for pixel in range(image.size):
        if image[pixel] > threshold:
            image[pixel] = 255
    image = image.reshape(height, width)
    return image


def sig(x, parameter):
    """
    Computes the sigmoid function value for a given input x and parameter.

    Args:
        x (float): Input value.
        parameter (float): Scaling parameter for the sigmoid function.

    Returns:
        float: Sigmoid function value.
    """
    return 1 / (1 + np.exp((-parameter) * (x - 127)))


# TODO : Check if faster MP approach exists
def transformSingle(file, address, denoise, destination, location, removeOriginals):
    """
    Transforms a single image file and saves the result to a new location.

    Args:
        file (str): Name of the file to transform.
        address (str): Path to the directory containing the file.
        denoise (bool): Flag to determine whether to denoise the image.
        destination (str): Path to the destination directory.
        location (str): Path to the source directory.
        removeOriginals (bool): Flag to determine whether to remove the original files.
    """
    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".bmp"):
        image = cv2.imread(os.path.join(address, file))
        transformed_image = image_transform(image, denoise)
        transformed_image_name = f"{os.path.join(destination + address.replace(location, ''), os.path.splitext(file)[0])}.bmp"
        cv2.imwrite(transformed_image_name, transformed_image)
        if removeOriginals:
            os.remove(os.path.join(address, file))


def transformAll(location, destination, removeOriginals=False, denoise=True):
    """
    Transforms all image files in a given directory and saves the results to a new location.

    Args:
        location (str): Path to the source directory.
        destination (str): Path to the destination directory.
        removeOriginals (bool, optional): Flag to determine whether to remove the original files. Default is False.
        denoise (bool, optional): Flag to determine whether to denoise the images. Default is True.
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
        process = pool.starmap_async(transformSingle,
                                     [(x, address, denoise, destination, location, removeOriginals) for x in files])
        pool.close()
        pool.join()

    print(f"Image transformation completed for catalog {location}")


def main():
    parser = argparse.ArgumentParser(description='Channel reduction to grayscale, normalization, shadow removal '
                                                 'and image scaling to 28x28px.')
    parser.add_argument('--source', type=str, required=True,
                        help='Dataset source directory.')
    parser.add_argument('--destination', type=str, required=True,
                        help='Processed dataset destination directory.')

    args = parser.parse_args()

    location = args.source
    destination = args.destination
    # TODO: Append transformed suffix?
    transformAll(location, destination)


if __name__ == '__main__':
    main()
