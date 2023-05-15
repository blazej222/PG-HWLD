import numpy as np
import cv2
import os


def image_transform(filename):
    imageFile = cv2.imread(filename)
    imageFile = cv2.cvtColor(imageFile, cv2.COLOR_BGR2GRAY)
    imageFile = cv2.normalize(imageFile, imageFile, 0, 255, cv2.NORM_MINMAX)
    # imageFile = remove_shadows(imageFile)
    # imageFile = cv2.convertScaleAbs(imageFile, 2, 1);
    # imageFile = cv2.fastNlMeansDenoising(imageFile, imageFile, 60.0, 7, 21)
    imageFile = flat_denoise(imageFile, 220)
    imageFile = cv2.resize(imageFile, dsize=(28, 28), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite("%s.bmp" % os.path.splitext(filename)[0], imageFile)
    os.remove(filename)


def remove_shadows(image):
    transformedImg = cv2.dilate(image, np.ones((7, 7), np.uint8))
    transformedImg = cv2.medianBlur(transformedImg, 21)
    transformedImg = 255 - cv2.absdiff(image, transformedImg)
    cv2.normalize(transformedImg, transformedImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return transformedImg


def flat_denoise(image, threshold):
    width = image.shape[1]
    height = image.shape[0]
    image = image.reshape(image.size)
    for pixel in range(image.size):
        if image[pixel] > threshold:
            image[pixel] = 255
    image = image.reshape(height, width)
    return image


def sig(x, parameter):
    return 1 / (1 + np.exp((-parameter) * (x - 127)))


def transformAll(location):
    for address, dirs, files in os.walk(location):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                image_transform(os.path.join(address, file))

        for directory in dirs:
            transformAll(os.path.join(address, directory))


def main():
    folderName = ".\\dataset"
    transformAll(folderName)
    transformAll("..\\uploaded-images")


if __name__ == '__main__':
    main()
