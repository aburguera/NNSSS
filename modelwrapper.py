# -*- coding: utf-8 -*-
###############################################################################
# Name        : ModelWrapper
# Description : Simple wrapper to ease the acces to one specific Keras model.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-March-2020 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from pickle import dump,load
from os.path import splitext
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from time import time

class ModelWrapper:
    # Constructor. Please note that it does NOT create the model
    # Input  : inputShape - The size of input images.
    def __init__(self,inputShape=(83,83)):
        self.theModel=None
        self.trainHistory=None
        self.inputShape=inputShape
        self.trainTime=0

    # Creates the model    
    def create(self):
        inputShape=(*self.inputShape,1)
        self.theModel=models.Sequential([
            Conv2D(16, (3, 3), activation='relu', padding='same',input_shape=inputShape),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),

            Conv2D(8, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(8, (3, 3), activation='relu',padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(3, (3, 3), activation='softmax', padding='same'),
            # Change the cropping layer depending on the input size (only
            # required for odd sizes)
            Cropping2D(cropping=((3, 2), (3, 2)))
            ])
        self.theModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    # Just a helper to build filenames
    def __build_filenames__(self,fileName):
        baseName,theExtension=splitext(fileName)
        modelFName=baseName+'.h5'
        histFName=baseName+'_HISTORY.pkl'
        return modelFName,histFName
        
    # Saves the model (as a .h5 file) and the training history (by means of
    # pickle)
    def save(self,fileName):
        modelFName,histFName=self.__build_filenames__(fileName)
        self.theModel.save(modelFName)
        outData=[self.trainHistory,self.trainTime]
        with open(histFName,'wb') as histFile:
            dump(outData,histFile)
        
    # Loads the model and the training history
    def load(self,fileName):
        modelFName,histFName=self.__build_filenames__(fileName)
        self.theModel=load_model(modelFName)        
        with open(histFName,'rb') as histFile:
            inData=load(histFile)
        self.trainHistory=inData[0]
        self.trainTime=inData[1]
        self.inputShape=self.theModel.layers[0].input_shape[1:-1]

    # Plots the training history
    def plot_training_history(self,plotTitle='TRAINING EVOLUTION'):
        plt.figure()
        plt.plot(self.trainHistory['loss'])
        plt.plot(self.trainHistory['val_loss'])
        plt.title(plotTitle)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'], loc='upper left')
        plt.show()
        
        plt.figure()
        plt.title(plotTitle)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.plot(self.trainHistory['acc'])
        plt.plot(self.trainHistory['val_acc'])
        plt.legend(['train','validation'], loc='upper left')
        plt.show()

    # Trains the model. Only useable with data generators. Please use those
    # defined in datagenerators.py.
    def train(self,trainGenerator,valGenerator=None,nEpochs=100):
        tStart=time()
        self.trainHistory=self.theModel.fit_generator(trainGenerator,epochs=nEpochs,validation_data=valGenerator).history
        self.trainTime=time()-tStart
        
    # Evaluate the model
    def evaluate(self,testGenerator):
        return self.theModel.evaluate_generator(testGenerator)
        
    # Output the model predictions.
    def predict(self,theImages):
        return self.theModel.predict(theImages)