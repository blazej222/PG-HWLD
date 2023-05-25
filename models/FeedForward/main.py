import feedforwardNN

# FIXME : Check paths

if __name__ == '__main__':
    fnn = feedforwardNN.FNN()
    fnn.load_sets("Training", True)
    fnn.load_sets("Testing", False)
    fnn.train()
    predictions = fnn.predict()