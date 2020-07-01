# -*- coding: utf-8 -*-
###############################################################################
# Name        : DataGenerator
# Description : Keras data generator for SSS transects
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-March-2020 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from keras.utils import Sequence
from skimage.io import imread
from keras.utils import to_categorical
import numpy as np
import os
import sys

class DataGenerator(Sequence):
    # Constructor. The parameters are:
    # - dataPath     : Path to the SSS informative images. Files within the
    #                  path must be named TRANnn.png, where nn is a two digit
    #                  number.
    # - gtPath       : Path to the ground truth images. Each ground truth file
    #                  must have the exact same name that the corresponding
    #                  SSS data file in dataPath.
    # - transectList : List of transects from which the data generator gathers
    #                  the data. A transect is specified by a number, which
    #                  corresponds to the "nn" in the associated transect
    #                  file name.
    # - patchMargin  : A patch is a swath together with the preceeding
    #                  and the subsequent patchMargin swaths.
    # - patchStep    : Distance (in swaths) between central swaths of consecu-
    #                  tive patches.
    # - batchSize    : The batch size. Fair enough :)
    # - doRandomize  : If False, patches are provided as they are in the
    #                  informative acoustic images. If True, they are provided
    #                  in random order. Advise: randomize for training, do not
    #                  randomize for testing (so it is easier to build the
    #                  segmented acoustic image).
    def __init__(self,dataPath,gtPath,transectList=[0,1,2,3,4,5,6,7,8,9],patchMargin=41,patchStep=41,batchSize=10,doRandomize=False):
        self.dataPath=dataPath
        self.gtPath=gtPath
        self.transectList=transectList
        self.patchMargin=patchMargin        
        self.patchStep=patchStep
        self.batchSize=batchSize
        self.doRandomize=doRandomize
        self._init_indexes_()
        self.lastLoadedTransect=-1
        self.loadedData=[]
        self.loadedGT=[]
        self.on_epoch_end()

    # Randomizes data at the end of every epoch
    def on_epoch_end(self):
        if self.doRandomize:
            np.random.shuffle(np.transpose(self.dataIndexes))
            
    # Number ot batches
    def __len__(self):
        return int(np.ceil(self.dataIndexes.shape[1]/float(self.batchSize)))

    # Provides a batch
    # Batch format:
    # - X : The data. Numpy array of shape (bs,ps,ps,1)
    # - y : The ground truth. Numpy array of shape (bs,ps,ps,3)
    # Where nb=batch size, ps=patch size (2*patchMargin+1).
    # "y" is provided in categorical format. Thus, the last dimension (3) is
    # the number of classes.
    def __getitem__(self,theIndex):
        X=[]
        y=[]
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.dataIndexes.shape[1])
        for i in range(bStart,bEnd):
            tranToLoad=self.dataIndexes[0,i]
            colToLoad=self.dataIndexes[1,i]
            if tranToLoad!=self.lastLoadedTransect:
                self.lastLoadedTransect=tranToLoad
                self.loadedData,self.loadedGT=self._load_image_pair_(tranToLoad)
            curData=self.loadedData[:,colToLoad-self.patchMargin:colToLoad+self.patchMargin+1]
            curGT=self.loadedGT[:,colToLoad-self.patchMargin:colToLoad+self.patchMargin+1]
            curGT=to_categorical(np.round(curGT*2.0).astype('int'),num_classes=3)
            curData=np.reshape(curData,(curData.shape[0],curData.shape[1],1))
            X.append(curData)
            y.append(curGT)
        return np.array(X),np.array(y)

    # Loads a pair of data and ground truth
    def _load_image_pair_(self,numTran):
        baseName='TRAN'+str(numTran).zfill(2)+'.png'
        dataImage=imread(os.path.join(self.dataPath,baseName))
        gtImage=imread(os.path.join(self.gtPath,baseName))
        dataImage=dataImage.astype('float')/255.0
        gtImage=gtImage.astype('float')/255.0
        return dataImage[:self.patchMargin*2+1,:],gtImage[:self.patchMargin*2+1,:]
        
    # Checks data and builds the vector of indices
    def _init_indexes_(self):
        self.dataIndexes=np.empty((2,0),int)
        for numTran in self.transectList:
            baseName='TRAN'+str(numTran).zfill(2)+'.png'
            dataImage=imread(os.path.join(self.dataPath,baseName))
            gtImage=imread(os.path.join(self.gtPath,baseName))
            if dataImage.shape!=gtImage.shape:
                sys.exit('[ERROR: '+baseName+'] ALL DATA AND GT IMAGES MUST HAVE THE SAME WIDTH AND HEIGHT.')
            if dataImage.shape[0]<self.patchMargin*2+1:
                sys.exit('[ERROR: '+baseName+'] ALL DATA AND GT IMAGES MUST HAVE HEIGHTS LARGER THAN '+str(1+2*self.patchMargin))
            if dataImage.shape[1]<self.patchMargin*2+1:
                sys.exit('[ERROR: '+baseName+'] ALL DATA AND GT IMAGES MUST HAVE WIDTHS LARGER THAN '+str(1+2*self.patchMargin))
            if len(dataImage.shape)!=2 or len(gtImage.shape)!=2:
                sys.exit('[ERROR: '+baseName+'] ALL DATA AND GT IMAGES MUST HAVE ONE SINGLE CHANNEL.')
            colIndexes=np.array(range(self.patchMargin,dataImage.shape[1]-self.patchMargin,self.patchStep))
            tranIndexes=np.repeat(numTran,colIndexes.shape[0])
            dataToAppend=np.stack((tranIndexes,colIndexes))
            self.dataIndexes=np.concatenate((self.dataIndexes,dataToAppend),1)
            
    # Outputs the number of items (not batches)
    def get_num_items(self):
        return self.dataIndexes.shape[1]