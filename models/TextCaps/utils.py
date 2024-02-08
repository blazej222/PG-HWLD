import numpy as np
import math

def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def load_emnist_balanced(cnt, train_load_path, test_load_path):
    from scipy import io as spio
    from keras.utils import to_categorical
    import numpy as np
    import matplotlib.pyplot as plt
    train_file = spio.loadmat(train_load_path)
    test_file = spio.loadmat(test_load_path)
    
    classes = 26
    cnt = cnt
    lim_train = cnt*classes
    
    x_train = train_file["dataset"][0][0][0][0][0][0]
    x_train = x_train.astype(np.float32)
    y_train = train_file["dataset"][0][0][0][0][0][1]
    if y_train.min() == 1:
        y_train -= 1
    x_test = test_file["dataset"][0][0][1][0][0][0]
    x_test = x_test.astype(np.float32)
    y_test = test_file["dataset"][0][0][1][0][0][1]
    if y_test.min() == 1:
        y_test -= 1

    x_train_test = train_file["dataset"][0][0][1][0][0][0]
    x_train_test = x_train_test.astype(np.float32)
    y_train_test = train_file["dataset"][0][0][1][0][0][1]
    if y_train_test.min() == 1:
        y_train_test -= 1

    # SHOW FIRST LETTER OF EACH DATASET
    # train_sample = np.array(train_file['dataset'][0][0][0][0][0][0][0]).reshape(28,28)
    # test_sample = np.array(test_file['dataset'][0][0][0][0][0][0][0]).reshape(28,28)
    # train_label = train_file['dataset'][0][0][0][0][0][1][0]
    # test_label = test_file['dataset'][0][0][0][0][0][1][0]
    #
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(train_sample, cmap='gray')
    # axs[0].set_title(train_label[0])
    # axs[1].imshow(test_sample, cmap='gray')
    # axs[1].set_title(test_label[0])
    # plt.show()
    # input("Press any key to continue...")

    #-------------------------------------------------

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1, order="A").astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1, order="A").astype('float32') / 255.
    x_train_test = x_train_test.reshape(x_train_test.shape[0],28,28,1, order="A").astype('float32') / 255
 
    y_train = (y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    y_train_test = to_categorical(y_train_test.astype('float32'))
    
    #Append equal number of training samples from each class to x_train and y_train
    x_tr = []
    y_tr = []
    count = [0] * classes
    for i in range(0,x_train.shape[0]):
        if (sum(count)==classes*cnt):
            break
        name = (y_train[i])
        if (count[int(name)]>=cnt):
            continue
        count[int(name)] = count[int(name)]+1
        x_tr.append(x_train[i])
        y_tr.append(name)
    x_tr = np.asarray(x_tr)
    y_tr = np.asarray(y_tr)
    y_tr = to_categorical(y_tr.astype('float32'))
    
    print(x_tr.shape,y_tr.shape,x_test.shape,y_test.shape)
    return (x_tr, y_tr), (x_test, y_test), (x_train_test, y_train_test)
    #return (x_tr, y_tr), (x_test, y_test), (x_test, y_test)