# The state of machine learning models in handwritten character recognition

## Introduction

This repository contains models and utilities used to achieve results described in paper.
"The state of machine learning models in handwritten character recognition". 

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
- `extracted` - dataset images extracted into `train-images` and `test-images` directories, with separate directories for each class.
Example - letter `a` sample used for training can be stored as `train-images/a/xxxx.png`.
- `extracted_undivided` - same as above but samples aren't divided into `train` and `test` directories.

## Getting started

### CUDA Requirements
- Nvidia CUDA toolkit 8.0.61
- Nvidia CUDNN library 5.1

### Repository location (Windows)
Due to an unaddressed issue in the required version of `keras`, some Windows path strings dependent on the disk location
cause crashes of model scripts. We recommend placing this repository in the root directory of your volume. 

### Usage of bundled virtual environments (recommended, Windows)
Due to the growing difficulty in obtaining specific versions of dependency packages required to run the bundled models, 
we have included pre-configured virtual environments for usage with Windows under `resources/python/`.

Two separate virtual environments are required in order to run all the models and utilities.

Python `3.6.8` (`venv-3.6.8`) is required to run `TextCaps` model, and Python `3.11.7` (`venv`) is required to run 
all the other models/utilities.

We highly recommend using the bundled virtual environments for performing testing or validation, as some of the packages
required for the models to operate are no longer easily obtainable from official repositories.

Before running any python code using said environments, please run the `install.ps1` script located in the main directory
to set up the necessary absolute paths for your system.

In order to switch between the virtual environments, please invoke the corresponding `activate.ps1` Powershell script 
from the `Scripts` directory of a given environment. The Powershell version does not use absolute paths, and as such no 
further machine-specific modification is required. Please see `scripts/test.ps1` for usage examples.

The bundled virtual environments have been tested and proved functional using:

- Nvidia RTX 3000 and 4000 series graphics cards
- Nvidia GeForce Game Ready Driver 576.88

### Creating your own virtual environments (optional, Linux)
Create appropriate virtual environments and activate the one you'll
be using for now (`3.6.8` for TextCaps, `3.11.7` for everything else).

Every model and utility directory should contain a file called `requirements.txt`. Once inside model/utility directory, use:

`pip install -r requirements.txt`

to install all requirements.

In order to use TextCaps `tensorflow-gpu 1.2.1` must be installed, with `cuda_8.0.61` and `cudnn-5.1`.

If some of the packages fail to install or throw errors, try copying them from the `Lib` directories of virtual 
environments bundled with this repository under `resources/python/`.

## Using models

### Scripts
Powershell scripts for training, testing and cross-validation have been provided inside this repository.

`cross_validate.ps1` - split dataset into subsets for cross validation, then perform model training on subsets.

`train.ps1` - trains all models thrice on Emnist-letters dataset.

`test.ps1` - performs testing on all models.

`test_cross_validate.ps1` - performs cross-validation testing on subsets created and models trained by `cross_validate.ps1`.

`pack_dataset.ps1` - packs dataset `extracted` to `.mat` format by calling `DataSetPacker`.

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
* `--do_not_rotate_images` - Do not perform automatic rotation of images when no train dataset is specified. Not specified by default.
* `--cmsuffix` - By default, a confusion matrix image is generated by the model. This allows appending a suffix to the
filename in case we are running multiple tests from script file. Default is empty.

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
* `--cmsuffix` - By default, a confusion matrix image is generated by the model. This allows appending a suffix to the
filename in case we are running multiple tests from script file. Default is empty.
* `--verbose` - Whether to show model summary. Not specified by default.
* `--do_not_rotate_images` - Do not perform automatic rotation of images when no train dataset is specified. Not specified by default.

### TextCaps
This model requires the following parameters in order to run:
* `--train_path` - Path to the training subset of dataset in `extracted` format (`dataset/train`). 
Required only when training. 
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
* `--samples_to_generate` - This option is used only when `--data_generate` is specified.
* `--test` - If this option is specified, the model will only be tested (training is disabled). Requires `--weights` to be specified.
* `--cmsuffix` - By default, a confusion matrix image is generated by the model. This allows appending a suffix to the
filename in case we are running multiple tests from script file. Default is empty.

## Description of utilities

### DataSetPacker 
DataSetPacker provides an ability to pack the dataset from `extracted` to `.mat` format (used by TextCaps model).

The following arguments are accepted by this utility:

- `--source` - Source directory of dataset in `extracted` format (required).
- `--destination` - Destination directory where resulting dataset in `.mat` format will be placed (required).
- `--reverse_colors` - Whether colors of the images should be inverted. Helpful when converting from
  (white background,black letters) to (black background,white letters) format used by EMNIST. `False` by default.
- `--filename` - Name of the resulting `.mat` file containing converted dataset.

### DirectorySplitter
DirectorySplitter splits a dataset from `extracted_undivided` format
into `train` and `test` directories by moving files to appropriate folders, respecting `split_ratio`.
Resulting dataset is in `extracted` format.

The following arguments are accepted by this utility:

- `--source` - Source directory of dataset in `extracted_undivided` format (required).
- `--destination` - Destination directory where resulting dataset in `extracted` format will be placed (required).
- `--split_ratio` - Split ratio between training and testing sets. Default is `0.8`, meaning 80% of samples will 
go to training subset and 20% of samples will go to testing subset.
 
### DataAugmenter
Data augmenter creates multiple variants of the source dataset by rotating each sample by a certain amount of degrees.

The following arguments are accepted by this utility:

- `--source` - Source directory of dataset in `extracted_undivided` format (required).
- `--destination` - Destination directory where resulting rotated dataset in `extracted_undivided` format will be placed (required).
- `--angle` - Rotation angle of each sample. This parameter is required.

### DataSetConverter
Utility that extracts sample images from dataset in `.idx-ubyte` format to separate `.png` images 
(dataset in `extracted_undivided` format). 
Also creates `.txt` file / numpy array with labels for each letter.

The following arguments are accepted by this utility:

- `--source_dataset` - Source directory of dataset in `idx3-ubyte` format (required).
- `--source_labels` -  Labels file in `idx1-ubyte` format (required)
- `--destination` - Destination directory where resulting processed dataset in `extracted_undivided` format will be placed (required).


### EMNISTifier
Transforms source dataset in `extracted_undivided` format into a dataset more familiar to EMNIST dataset by inverting image color scale, 
applying Gaussian Filter and centering the images. More details are available in our paper.

The following arguments are accepted by this utility:

- `--source` - Source directory of dataset in `extracted_undivided` format (required).
- `--destination` - Destination directory where resulting processed dataset in `extracted_undivided` format will be placed (required).
- `--threshold` -  For every pixel, the same threshold value is applied. 
If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value. Default is `100`.
- `--verbose` - Whether debug data should be printed. `False` by default.

### ImageCropper
Automatically detect contour of dark image on white background, center it and add white margin to image from
dataset.

The following arguments are accepted by this utility:

- `--source` - Source directory of dataset in `extracted_undivided` format (required).
- `--destination` - Destination directory where resulting cropped dataset will be placed in `extracted_undivided` format (required).
- `--threshold` -  For every pixel, the same threshold value is applied. 
If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value. Default is `100`.
- `--margin` - Size of added margin. This argument is required.

### ImageCutter
Divide previously fitted scans of letter sheets collected as part of collecting data sets into a rectangular grid.

The following arguments are accepted by this utility:

- `--source` - Source directory of scanned letter sheets (required).
- `--destination` - Destination directory where resulting dataset will be placed in `extracted_undivided` format (required).
- `--left_margin` - Margin from the left side of the sheet/letter where cutting will start. Default is `36`px.
- `--upper_margin` - Margin from the top side of the sheet/letter row where cutting will start. Default is `48`px.
- `--crop_width` - How much each letter should be cropped horizontally. Default is `196`px.
- `--crop_height` - How much each letter should be cropped vertically. Default is `196`px.
- `--divisor` - Additional gap applied between each letter both vertically and horizontally. Default is `5`px.
- `--num_tiles_x` - Amount of tiles in x direction on scanned page (horizontally). Default is `12`.
- `--num_tiles_y` - Amount of tiles in y direction on scanned page (vertically). Default is `17`.
- `--num_files` - Amount of samples that should be obtained from each scan page. After extracting this 
number of samples extracting process will stop and utility will move to another scan file. 


### ImageTransformer
Channel reduction to grayscale, normalization, shadow removal and image scaling to 28x28px.

The following arguments are accepted by this utility:

- `--source` - Source directory of dataset in `extracted_undivided` format (required).
- `--destination` - Destination directory where processed dataset in `extracted_undivided` format will be 
placed (required).
