import feedforwardNN

if __name__ == '__main__':
    fnn = feedforwardNN.FNN()
    fnn.load_sets("../../resources/datasets/unpacked/dataset-emnist/train-images", True)
    fnn.load_sets("../../resources/datasets/unpacked/dataset-emnist/test-images", False)
    fnn.train()
    predictions = fnn.predict()