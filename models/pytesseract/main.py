import pytesseract
from PIL import Image
import os
tesseractPath = "./bin/tesseract.exe"
dataset_test = "../../resources/datasets/unpacked/dataset-emnist/test-images" # temporary test path

pytesseract.pytesseract.tesseract_cmd = tesseractPath

directories = sorted(os.listdir(dataset_test), key=len)

flawlessly_recongized = 0
ok_recognized = 0
dataset_size = 0

for dir in directories:
    files = sorted(os.listdir(os.path.join(dataset_test, dir)))  # contains files
    for file in files:
        dataset_size += 1
        fullpath = os.path.join(dataset_test, dir, file)
        #  Refer to https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc
        letter = pytesseract.image_to_string(fullpath, config='--psm 10 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        #  letter = pytesseract.image_to_string(Image.open(fullpath), config='--psm 10 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz')
        #  To recognize between uppercase and lowercase, add ABCDEFGHIJKLMNOPQRSTUVWXYZ to whitelist
        letter = letter.replace("\n","") #  get rid of the newline character

        if len(letter) == 1 and (letter == dir or letter == dir.upper()): flawlessly_recongized += 1  #  if we recognized one letter and its the correct one
        elif len(letter) > 1:
            if letter[0] == dir or letter[0] == dir.upper(): ok_recognized += 1
        else:
            pass
    print("Currently done directory {}".format(dir))

        #print("Image {}/{} recognized as {}".format(dir,file,letter))
percentage = (flawlessly_recongized + ok_recognized) / dataset_size
print("Percentage was {}".format(percentage))