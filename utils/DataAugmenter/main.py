import ntpath
import numpy as np
from scipy import ndimage
import cv2
import os


def image_rotate(imageFile, angle):
    imageFile = ndimage.rotate(imageFile, angle, mode='constant', cval=255, reshape=False)
    return imageFile


def rotateAll(location, destination, angle):
    if not os.path.exists(destination):
        os.makedirs(destination)

    for address, dirs, files in os.walk(location):
        for directory in dirs:
            subdirectory = address.replace(location, "")
            temp = destination + subdirectory
            if not os.path.exists(os.path.join(temp, directory)):
                os.makedirs(os.path.join(temp, directory))

        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".bmp"):
                image = cv2.imread(os.path.join(address, file))
                rotated_image = image_rotate(image, angle)
                rotated_image_name = f"{os.path.join(destination + address.replace(location, ''), os.path.splitext(file)[0])}_{angle}.png"
                cv2.imwrite(rotated_image_name, rotated_image)

    print(f"Dataset augmentation completed for catalog {location}, angle: {angle}")


def main():
    location = "../../resources/datasets/unpacked/dataset-single-person"
    destination = "../../resources/datasets/augmented/dataset-single-person-augmented"
    rotateAll(location, destination, 15)
    rotateAll(location, destination, 10)
    rotateAll(location, destination, 5)
    rotateAll(location, destination, 0)
    rotateAll(location, destination, -5)
    rotateAll(location, destination, -10)
    rotateAll(location, destination, -15)

    location = "../../resources/datasets/unpacked/dataset-single-person-cropped-10"
    destination = "../../resources/datasets/augmented/dataset-single-person-cropped-10-augmented"
    rotateAll(location, destination, 15)
    rotateAll(location, destination, 10)
    rotateAll(location, destination, 5)
    rotateAll(location, destination, 0)
    rotateAll(location, destination, -5)
    rotateAll(location, destination, -10)
    rotateAll(location, destination, -15)


if __name__ == '__main__':
    main()
