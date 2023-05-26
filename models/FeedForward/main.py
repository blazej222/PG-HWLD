import feedforwardNN
#from timeit import default_timer as timer
import time

# train_path = "../../resources/datasets/unpacked/dataset-emnist/train-images"
# test_path = "../../resources/datasets/unpacked/dataset-emnist/test-images"

train_path = "C:/tmp/dataset-emnist/train-images"
test_path = "C:/tmp/dataset-emnist/test-images"

if __name__ == '__main__':
    fnn = feedforwardNN.FNN()
    fnn.load_sets(train_path, True)
    fnn.load_sets(test_path, False)
    print("Starting training process")
    start = time.time()
    fnn.train()
    end = time.time()
    print("Training finished in {}".format(end-start))
    predictions = fnn.predict()