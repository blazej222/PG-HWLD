from neural_network import *

if __name__ == '__main__':
    network = Network(None, None, None)
    network.create_cnn()
    network.training_testing_set(train_catalog=".\\train-images", test_catalog=".\\test-images")
    network.train()
    network.test_catalog(".\\test-images", ".\\test-labels.txt")
    # network.test_catalog(".\\imageTransformer/dataset/Training/e", label='e', doPrint=True)
    # network.test(".\\uploaded-images/testImage.bmp")
    # network.test(".\\uploaded-images/testImage2.bmp")
    # network.test_catalog(".\\dataset-black-marker/g", test_by_folder=True, doPrint=True)
