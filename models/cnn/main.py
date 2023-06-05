import os
import shutil
import utils.ImageCropper.main as ImageCropper
import utils.DataAugmenter.main as DataAugmenter
import utils.ImageTransformer.main as ImageTransformer
from neural_network import *

train_images_path = "../../resources/datasets/dataset-EMNIST/train-images"
# train_images_path = "../../resources/datasets/augmented/dataset-multi-person-augmented"
test_images_path = "../../resources/datasets/dataset-EMNIST/test-images"
test_labels_path = "../../resources/datasets/dataset-EMNIST/test-labels.txt"

def prediction_loop():
    print(f"Catalog test for EMNIST/test-images")
    for i in range(0, 26):
        print(f"Testing for {chr(97 + i)}:", end=" ")
        network.test_catalog(
            f"../../resources/datasets/dataset-EMNIST/test-images/{chr(97 + i)}",
            label=chr(97 + i), echo=False)

    print(f"Catalog test for dataset-single-person")
    for i in range(0, 26):
        print(f"Testing for {chr(97 + i)}:", end=" ")
        network.test_catalog(
            f"../../resources/datasets/transformed/dataset-single-person/{chr(97 + i)}",
            label=chr(97 + i), echo=False)

    print(f"Catalog test for dataset-multi-person")
    for i in range(0, 26):
        print(f"Testing for {chr(97 + i)}:", end=" ")
        network.test_catalog(
            f"../../resources/datasets/transformed/dataset-multi-person/{chr(97 + i)}",
            label=chr(97 + i), echo=False)

    for c in range(1, 7):
        margin = c * 5

        print(f"Catalog test for margin={margin}, single-person")
        for i in range(0, 26):
            print(f"Testing for {chr(97 + i)}:", end=" ")
            network.test_catalog(
                f"../../resources/datasets/transformed/dataset-single-person-cropped-{margin}/{chr(97 + i)}",
                label=chr(97 + i), echo=False)

        print(f"Catalog test for margin={margin}, multi-person")
        for i in range(0, 26):
            print(f"Testing for {chr(97 + i)}:", end=" ")
            network.test_catalog(
                f"../../resources/datasets/transformed/dataset-multi-person-cropped-{margin}/{chr(97 + i)}",
                label=chr(97 + i), echo=False)

    print(f"Catalog test for dataset-single-person-cropped-20-augmented")
    for i in range(0, 26):
        print(f"Testing for {chr(97 + i)}:", end=" ")
        network.test_catalog(
            f"../../resources/datasets/transformed/dataset-single-person-cropped-20-augmented/{chr(97 + i)}",
            label=chr(97 + i), echo=False)

    print(f"Catalog test for dataset-multi-person-cropped-20-augmented")
    for i in range(0, 26):
        print(f"Testing for {chr(97 + i)}:", end=" ")
        network.test_catalog(
            f"../../resources/datasets/transformed/dataset-multi-person-cropped-20-augmented/{chr(97 + i)}",
            label=chr(97 + i), echo=False)


# FIXME: Something repeatedly raises exceptions when trying to call train and then test methods in 1 go (I think it
#  doesn't close the filestream properly before trying to perform tests, therefore doesn't load the model properly)
if __name__ == '__main__':
    network = Network(None, None, None)
    network.create_simple_cnn()
    # network.create_cnn()
    network.training_testing_set(train_catalog=train_images_path, test_catalog=test_images_path)
    network.train()

    margin = 0
    color_threshold = 100
    rotation_angle = 0

    ImageCropper.crop_black_letters_catalog("../../resources/uploaded-images", "../../resources/cropped-images",
                                            margin, color_threshold)
    DataAugmenter.rotateAll("../../resources/cropped-images", "../../resources/rotated-images", rotation_angle)
    ImageTransformer.transformAll("../../resources/rotated-images/", "../../resources/transformed-images")

    for file in os.listdir("../../resources/transformed-images"):
        network.test(os.path.join("../../resources/transformed-images", file))

    shutil.rmtree("../../resources/cropped-images")
    shutil.rmtree("../../resources/rotated-images")
    shutil.rmtree("../../resources/transformed-images")

    # prediction_loop()

    # for i in range(0, 26):
    #     print(f"Testing for {chr(97 + i)}:", end=" ")
    #     network.test_catalog(
    #         f"../../resources/datasets/divided/ultimate_dataset_3000/test-images/{chr(97 + i)}",
    #         label=chr(97 + i), echo=False)

    # network.test_catalog(f"../../resources/datasets/transformed/dataset-single-person-cropped-10-augmented", test_by_folder=True)
    # network.test_catalog(f"../../resources/datasets/transformed/dataset-single-person-cropped-20", test_by_folder=True)

    # network.test_catalog(f"../../resources/datasets/dataset-EMNIST/test-images", test_by_folder=True)
    # network.test_catalog(f"../../resources/datasets/transformed/dataset-multi-person-cropped-5/b", label='b')

    # network.test_catalog(f"../../resources/datasets/transformed/dataset-multi-person-cropped-10", test_by_folder=True)

    # for i in range(0, 26):
    #     print(f"Testing for {chr(97 + i)}")
    #     network.test_catalog(f"../../resources/datasets/augmented/dataset-multi-person-cropped-10-augmented/{chr(97 + i)}", label=chr(97 + i))