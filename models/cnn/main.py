from neural_network import *

train_images_path = "../../resources/datasets/dataset-EMNIST/train-images"
# train_images_path = "../../resources/datasets/augmented/dataset-multi-person-augmented"
test_images_path = "../../resources/datasets/dataset-EMNIST/test-images"
test_labels_path = "../../resources/datasets/dataset-EMNIST/test-labels.txt"

if __name__ == '__main__':
    network = Network(None, None, None)
    network.create_cnn()
    network.training_testing_set(train_catalog=train_images_path, test_catalog=test_images_path)
    network.train()
    network.test_catalog(test_images_path, test_labels_path)

    # for i in range(0, 26):
    #     print(f"Testing for {chr(97 + i)}")
    #     network.test_catalog(f"../../resources/datasets/transformed/dataset-multi-person-cropped-10/{chr(97 + i)}",
    #                          label=chr(97 + i))

    # network.test_catalog(f"../../resources/datasets/transformed/dataset-multi-person-cropped-10", test_by_folder=True)

    # for i in range(0, 26):
    #     print(f"Testing for {chr(97 + i)}")
    #     network.test_catalog(f"../../resources/datasets/augmented/dataset-multi-person-cropped-10-augmented/{chr(97 + i)}", label=chr(97 + i))

    # network.test_catalog(".\\imageTransformer/dataset/Training/e", label='e', doPrint=True)
    # network.test(".\\uploaded-images/testImage.bmp")
    # network.test(".\\uploaded-images/testImage2.bmp")
    # network.test_catalog(".\\dataset-black-marker/x", test_by_folder=True, doPrint=True)
