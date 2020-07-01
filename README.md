# Multi-Class Segmentation of Side-Scan Sonar Data using a Neural Network

This code makes it possible to segment Side-Scan Sonar (SSS) acoustic images into three different classes: rock, sand and others. The segmentation is based on a Convolutional Neural Network (CNN) and it is aimed at on-line operation.

This is the kind of input our system deals with:
![Acoustic Image](https://github.com/aburguera/NNSSS/blob/master/DATASET/DATA/TRAN00.png)

This is the kind of output ir produces:
![Ground truth](https://github.com/aburguera/NNSSS/blob/master/DATASET/GT/TRAN00.png)

If you use this software, please cite the following paper:

Paper reference (to be posted soon)

Also, the whole process is carefully described in the paper. Please read the paper to understand how the system works.

## Credits

* Science : Antoni Burguera (antoni dot burguera at uib dot es) and Francisco Bonin-Font
* Coding : Antoni Burguera (antoni dot burguera at uib dot es)
* Data : The underwater equipment used to gather this dataset was provided by Unidad de Tecnología Marina-CSIC (http://www.utm.csic.es/auv/). The authors wish to thank Pablo Rodríguez Fornes, from UTM-CSIC, and Yvan Petillot, from Heriot-Watt University, for sharing their expertise with us and providing the data used in the experiments presented in this article. The authors are also grateful to Daniel Moreno Linares for his help with the XTF format.

## Understanding the system

The main modules are:

* DataGenerator : Keras data generator to feed the Neural Network.
* ModelWrapper : Simple wrapper to ease the creation, loading, saving, ... the Keras model.
* Tester : Tests and evaluates the system in different ways.

For each of these modules, there is an usage example available. To understand each of them just check the corresponding example_* file. There is an addition example, called main, which exemplifies the general usage of this software. All these examples are Jupyter Notebooks.

## The datasets

This code is provided together with one dataset, located in the DATASET folder. The dataset is divided in two subfolders:

* DATA: Contains the SSS data. The data is a set of informative acoustic images. Please refer to the paper to understand what are the informative images. Images are named TRANnn.png. Each even numbered nn corresponds to the port data and the subsequent odd numbered nn is the corresponding starboard data gathered in the same transect. Images are PNG in Grayscale.
* GT: Contains the hand labelled ground truth. Each ground truth file is related to the data file with the same name. They are PNG files. Possible values for pixels are 0, 127 and 255. Each value represents one of the three classes.

Please, read the paper (cited above) and check the provided examples to understand the dataset.


## The models

Two pre-trained models are provided in the MODELS folder:

* TEST_MODEL: This is the example model created in example_modelwrapper.ipynb. See the example for more info.
* MODEL01234567TS01: This has been trained with data from TRAN00 to TRAN07 with a training step of 1.

To use one of these models just use the ModelWrapper load member function. For example:

    from modelwrapper import ModelWrapper
    theModel=ModelWrapper()
    theModel.load('MODELS/TEST_MODEL')

Check example_modelwrapper.ipynb and example_tester.ipynb for more information.

## Using the system

Check main.ipynb for an example.

## Requirements

To execute this software, you will need:

* Python 3
* Keras
* Tensorflow
* NumPy
* Matplotlib
* Pickle
* SciKit-Image
* SciKit-Learn

## Disclaimer

The code is provided as it is. It may work in your computer, it may not work. It may even crash it or, eventually, create a time paradox, the result of which could cause a chain reaction that would unravel the very fabric of the space-time continuum and destroy the entire universe. Just be careful and try to understand everything before using it. If you have questions, please carefully read the code and the paper. If this doesn't help, contact us.
