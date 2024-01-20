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

## Getting started

Two separate virtual environments are required in order to run all the models and utilities. 
They should be created under resources/python/.

Python `3.6.8` is required to run `TextCaps` model, and Python `3.11.7` is required to run 
all the other models/utilities.

Every model and utility directory should contain a file called `requirements.txt`.

Create appropriate virtual environments and activate the one you'll
be using for now (`3.6.8` for TextCaps, `3.11.7` for everything else).

Once inside model/utility directory, use:

`pip install -r requirements.txt`

to install all requirements.

This repository contains project run configurations for JetBrains Pycharm IDE.

## Description of utilities
Work in progress.

