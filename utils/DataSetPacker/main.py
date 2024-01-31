import numpy as np
import os
from PIL import Image,ImageOps
from scipy import io as spio
import multiprocessing as mp

dataset_directory = "../../resources/datasets/dataset-EMNIST"
destination = "../../resources/datasets/packed/dataset-EMNIST"
train_path = dataset_directory + "/train-images"
test_path = dataset_directory + "/test-images"
reverse_colors = True
filename = "/ouremnist.mat"

def load_directory(address,directory):
    images = []
    files = os.listdir(os.path.join(address,directory))
    for file in files:
        tmp = Image.open(os.path.join(address,directory,file))
        if reverse_colors:
            tmp = ImageOps.invert(tmp)
        tmp = np.array(tmp, order='F').reshape(-1)
        tmp = tmp.tolist()
        images.append(tmp)
    return [images,ord(directory)-97]

def load_classes(path,dirs,x,y):
    for directory in dirs:
        result = load_directory(path, directory)
        x.extend(result[0])
        y_tmp = [[result[1]]] * len(result[0])
        y.extend(y_tmp)

dirs = os.listdir(train_path)
if not os.path.exists(destination):
    os.makedirs(destination)

x_train = []
y_train = []
x_test = []
y_test = []

load_classes(train_path,dirs,x_train,y_train)
load_classes(test_path,dirs,x_test,y_test)


x_train = np.array(x_train,dtype="uint8")
y_train = np.array(y_train,dtype="uint8")
writers = np.array([[0]] * y_train.size, dtype="uint8")

x_test = np.array(x_test,dtype="uint8")
y_test = np.array(y_test,dtype="uint8")

third_selector_choice_one = np.array([[(x_train,y_train,writers)]], dtype=[("images", "O"), ("labels", "O"), ("writers", "O")]) # dtype is actually important here!
# print(third_selector_choice_one)

third_selector_choice_two = np.array([[(x_test,y_test,writers)]], dtype=[("images", "O"), ("labels", "O"), ("writers", "O")])
# print(third_selector_choice_two)

maps = [[i,i+65,i+97] for i in range(len(dirs))]
# print(maps)

third_selector_choice_three = np.array(maps, dtype="uint8")
# print(third_selector_choice_three)

tpl = (third_selector_choice_one,third_selector_choice_two,third_selector_choice_three)
lst = [tpl]
mainarray = np.array([lst], dtype=[("train","O"),("test","O"),("map","O")]) # TODO:CORRECT DTYPE
#print(mainarray)

data_to_save = {"dataset": mainarray}
spio.savemat(destination + filename, data_to_save)


    # pool = mp.Pool()
    # pool.starmap_async(crop_black_letters_file,[(address,x,margin,threshold,destination,location) for x in files])
    # pool.close()
    # pool.join()
    #
    # print(f"Cropping finished for catalog {location}")