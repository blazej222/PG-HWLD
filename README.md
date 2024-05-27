# The state of machine learning models in handwritten character recognition

## Introduction

This repository contains models and utilities used to achieve results described in paper.
"The state of machine learning models in handwritten character recognition". 
Link to the article is presented [here](example.org).

## Description of directory structure

This repository consists of the following directories:
- models - source code of models tested in the article.
- resources - python virtual environments and datasets used in model training and/or testing.
- utils - utilities used mainly for cutting, cropping, transforming and augmenting datasets.
- scripts - scripts used for automatic training and testing of the models.

## Dataset formats

The following dataset formats are being used in the project:

- `idx-ubyte` - binary files containing encoded labels and dataset images.
- `.mat` - dataset images and labels stored inside matlab file.
- `extracted` - dataset images extracted into `train` and `test` directories, with separate directories for each class.
Example - letter a sample used for training can be stored as `train/a/xxxx.png`.

## Getting started

Two separate virtual environments are required in order to run all the models and utilities. 
They should be created under `resources/python/`.

Python `3.6.8` is required to run `TextCaps` model, and Python `3.11.7` is required to run 
all the other models/utilities.

Every model and utility directory should contain a file called `requirements.txt`.

Create appropriate virtual environments and activate the one you'll
be using for now (`3.6.8` for TextCaps, `3.11.7` for everything else).

Once inside model/utility directory, use:

`pip install -r requirements.txt`

to install all requirements.

This repository contains project run configurations for JetBrains Pycharm IDE.

## Using models

### VGG-5
When run with no arguments, the model will train itself on `EMNIST-letters` dataset, and then perform a test 
against `EMNIST-letters` test subset.

Resulting weights will be saved to `./saved_models` directory under `model1.pth` and `model2.pth` 
for VGG and VGG-Spinal respectively.

Model behavior can be modified by using the following arguments:

* `--train_path` - Path to the training subset of dataset in `extracted` format (`dataset/train`). 
* `--test_path` - Path to the testing subset of dataset in `extracted` format (`dataset/test`).
* `--test` - If this option is specified, the model will only be tested (training is disabled).
* `--saved_model_path` - Directory in which `.pth` weight files are stored. Used both for loading and saving files. Default value is `./saved_models`. 
* `--model1_filename` - Weights filename of VGG model. Default value is `model1.pth`.
* `--model2_filename` - Weights filename of VGG-Spinal. Default value is `model2.pth`.
* `--verbose` - Whether to show some examples of data loaded to the user. Not specified by default.
* `--rotate_images` - If specified rotates images from test dataset by 90 degrees left, then flips them vertically to match those from default emnist training set. Not specified by default.

### WaveMix
When run with no arguments, the model will train itself on `EMNIST-letters` dataset, and then perform a test 
against `EMNIST-letters` test subset.

Resulting weights will be saved to `./saved_models` directory under `model.pth`.

Model behavior can be modified by using the following arguments:

* `--train_path` - Path to the training subset of dataset in `extracted` format (`dataset/train`). 
* `--test_path` - Path to the testing subset of dataset in `extracted` format (`dataset/test`).
* `--test` - If this option is specified, the model will only be tested (training is disabled).
* `--saved_model_path` - Directory in which `.pth` weight file is stored. Used both for loading and saving files. Default value is `./saved_models`. 
* `--model_filename` - Weights filename of VGG model. Default value is `model1.pth`.

### TextCaps
This model requires the following parameters in order to run:
* `--train_path` - Path to the training subset of dataset in `extracted` format (`dataset/train`). 
* `--test_path` - Path to the testing subset of dataset in `extracted` format (`dataset/test`).

If no other arguments are specified, the model will train itself on specified dataset, and then perform a test 
against specified test dataset.

Model behavior can be modified by using the following optional arguments:
* `--epochs` - Amount of epochs the model uses. Default is `60`.
* `--verbose` - Print additional debug information about model. Default is not specified.
* `--cnt` - Amount of samples taken from each class. Default is `200`.
* `--num_cls` - Amount of iterations the model performs. Default is `47`.
* `--batch_size` - Batch size the model uses. Default is `32`.
* `--lr` - Initial learning rate. Default is `0.001`.
* `--lr_decay` - The value multiplied by lr at each epoch. Set a larger value for larger epochs. Default is `0.9`.
* `--lam_recon` - The coefficient for the loss of decoder. Default is `0.392`.
* `--routings` - Number of iterations used in routing algorithm. Should be > 0. Default is `3`.
* `--shift_fraction` - Fraction of pixels to shift at most in each direction. Default is `0.1`.
* `--save_dir` - Directory where snapshots of the model will be saved.
* `--weights` - The path of the saved weights. Should be specified when testing.  Default is empty.
* `--data_generate` - If specified will generate new data with pre-trained model. Requires `--weights` to be specified. Not specified by default.
* `--samples_to_generate` - this option is used only when `--data_generate` is specified.
* `--test` - If this option is specified, the model will only be tested (training is disabled). Requires `--weights` to be specified.

## Description of utilities

### DataSetPacker 
DataSetPacker provides an ability to pack the dataset from by-class directory format to `.mat` (used by TextCaps model).

### DirectorySplitter
DirectorySplitter splits a dataset into `train` and `test` directories by moving files to appropriate folders, respecting `split_ratio`.
 
### DataAugmenter
Data augmenter creates multiple variants of the source dataset by rotating each sample by -15, -10, -5, 5, 10, 15 degrees.

### DataSetConverter
Utility that extracts sample images from dataset in `.idx-ubyte` format to separate `.png` images. 
Also creates `.txt` file / numpy array with labels for each letter.

### DirectorySplitter
Splits source dataset into `train` and `test` subsets according to `split_ratio`. Results in creation of `extracted` dataset.

### EMNISTifier
Transforms source dataset into a dataset more familiar to EMNIST dataset by inverting image color scale, 
applying Gaussian Filter and centering the images. More details are available in our paper.

### ImageCropper
Automatically detect contour of dark image on white background, center it and add white margin to image from
dataset.

### ImageCutter
Divide previously fitted scans of letter sheets collected as part of collecting data sets into a rectangular grid.

### ImageTransformer
Channel reduction to grayscale, normalization, shadow removal and image scaling to 28x28px.
