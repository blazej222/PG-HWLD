# TODO:Check other MP approaches (jit,chunksize etc.)
import pytesseract
from PIL import Image
import os
import multiprocessing as mp
import time
tesseractPath = "./bin/tesseract.exe"
dataset_test = "../../resources/datasets/transformed/dataset-multi-person-cropped-30" # temporary test path

pytesseract.pytesseract.tesseract_cmd = tesseractPath

directories = sorted(os.listdir(dataset_test), key=len)

flawlessly_recongized = 0
ok_recognized = 0
dataset_size = 0

def recognize(file,dir):
    current_flawlessly_recongized = 0
    current_ok_recognized = 0
    #current_dataset_size = len(files)
    #for file in files:
    fullpath = os.path.join(dataset_test, dir, file)
    #  Refer to https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc
    letter = pytesseract.image_to_string(fullpath, config='--psm 10 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    #  letter = pytesseract.image_to_string(Image.open(fullpath), config='--psm 10 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz')
    #  To recognize between uppercase and lowercase, add ABCDEFGHIJKLMNOPQRSTUVWXYZ to whitelist
    letter = letter.replace("\n","") #  get rid of the newline character

    if len(letter) == 1 and (letter == dir or letter == dir.upper()): current_flawlessly_recongized += 1  #  if we recognized one letter and its the correct one
    elif len(letter) > 1:
        if letter[0] == dir or letter[0] == dir.upper(): current_ok_recognized += 1
    else:
        pass
        #if len(letter) == 0:    letter = "null"
        #print(f"{dir};{letter};{file}")
        #print(f"Letter {dir} from file {file} wrongly recognized as {letter}")
    return [current_flawlessly_recongized,current_ok_recognized]

if __name__ == '__main__':
    print(dataset_test)

    start = time.time()

    for dir in directories:
        this_dir_flawless = 0
        this_dir_ok = 0
        files = sorted(os.listdir(os.path.join(dataset_test, dir)))  # contains files
        pool = mp.Pool()

        processmap = pool.starmap_async(recognize,[(file,dir) for file in files]) # Async currently works only on directory level

        pool.close()
        pool.join()

        results = processmap.get()
        dataset_size += len(files)

        for b,c in results:
            this_dir_flawless += b
            this_dir_ok += c
        print(f"Done directory {dir}. Accuracy was {(this_dir_flawless + this_dir_ok) / len(files)}")
        flawlessly_recongized +=this_dir_flawless
        ok_recognized += this_dir_ok

    end = time.time()
    percentage = (flawlessly_recongized + ok_recognized) / dataset_size
    print("Percentage was {}. Finished in {}".format(percentage,end-start))