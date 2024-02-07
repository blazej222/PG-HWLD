import numpy as np
import os
from PIL import Image,ImageOps
from matplotlib import pyplot as plt
from scipy import io as spio
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse


# source_dataset/
# ├── train-images/
# │   ├── a/
# │   ├── b/
# │   ├── c/
# │   └── ...
# └── test-images/
#     ├── a/
#     ├── b/
#     ├── c/
#     └── ...

parser = argparse.ArgumentParser(description='Pack dataset into .mat format')
parser.add_argument('--source', type=str, required=True,
                    help='Dataset source directory')
parser.add_argument('--destination', type=str, required=True,
                    help='.Mat file destination directory')
parser.add_argument('--reverse_colors', action='store_true', default=False,
                    help='Change black to white and vice versa')
parser.add_argument('--filename', type=str, default='packed_dataset.mat',
                    help='Output filename')
args = parser.parse_args()
print(args)

source = args.source
destination = args.destination
reverse_colors = args.reverse_colors
output_filename = args.filename
train_path = source + "/train-images"
test_path = source + "/test-images"

def load_directory(address,directory):
    images = []
    files = os.listdir(os.path.join(address,directory))
    for file in files:
        tmp = Image.open(os.path.join(address,directory,file))
        if reverse_colors:
            tmp = ImageOps.invert(tmp)
        tmp = np.array(tmp).reshape(-1,order='F')
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
if len(dirs) != 0:
    load_classes(train_path,dirs,x_train,y_train)
else:
    dirs = os.listdir(test_path)
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
mainarray = np.array([lst], dtype=[("train","O"),("test","O"),("mapping","O")])
#print(mainarray)

data_to_save = {"dataset": mainarray}
spio.savemat(destination + output_filename, data_to_save)


    # pool = mp.Pool()
    # pool.starmap_async(crop_black_letters_file,[(address,x,margin,threshold,destination,location) for x in files])
    # pool.close()
    # pool.join()
    #
    # print(f"Cropping finished for catalog {location}")