import ntpath
import numpy as np
import cv2
import os
import multiprocessing as mp

def image_transform(imageFile, denoise=True):
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
    transformedImg = cv2.dilate(image, np.ones((7, 7), np.uint8))
    transformedImg = cv2.medianBlur(transformedImg, 21)
    transformedImg = 255 - cv2.absdiff(image, transformedImg)
    cv2.normalize(transformedImg, transformedImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return transformedImg

#TODO: Check if eliminating reshapes speeds things up
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

# TODO (Blazej): Check if faster MP approach exists
def transformSingle(file,address,denoise,destination,location,removeOriginals):
    if file.endswith(".png") or file.endswith(".jpg"):
        image = cv2.imread(os.path.join(address, file))
        transformed_image = image_transform(image, denoise)
        transformed_image_name = f"{os.path.join(destination + address.replace(location, ''), os.path.splitext(file)[0])}.bmp"
        cv2.imwrite(transformed_image_name, transformed_image)
        if removeOriginals:
            os.remove(os.path.join(address, file))

def transformAll(location, destination, removeOriginals=False, denoise=True):
    if not os.path.exists(destination):
        os.makedirs(destination)

    for address, dirs, files in os.walk(location):
        for directory in dirs:
            subdirectory = address.replace(location, "")
            temp = destination + subdirectory
            if not os.path.exists(os.path.join(temp, directory)):
                os.makedirs(os.path.join(temp, directory))

        pool = mp.Pool()
        process = pool.starmap_async(transformSingle,[(x,address,denoise,destination,location,removeOriginals) for x in files])
        pool.close()
        pool.join()

    print(f"Image transformation completed for catalog {location}")


def main():
    location = "../../resources/datasets/unpacked"
    destination = "../../resources/datasets/transformed"
    transformAll(location, destination)
    # transformAll("../../resources/uploaded-images", "../../resources/uploaded-images", removeOriginals=False)


if __name__ == '__main__':
    main()
