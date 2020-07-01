# -*- coding: utf-8 -*-
###############################################################################
# Name        : Tester
# Description : Evaluates segmented informative images
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-March-2020 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

import numpy as np
import sys
from skimage.io import imread
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import os
from time import time

class Tester:
    # Constructor
    # Input : theModel - model to evaluate (ModelWrapper class)
    #         dataPath, gtPath - Paths to SSS data and ground truth
    #         patchStep - Step between patches during evaluation (Delta t in
    #                     the paper (see README)
    #         multiClass - If False, the most probable class is selected for
    #                      each bin. If True, the three probabilities are
    #                      left.
    #         colorOrder - How color channels (R,G,B) are assigned to each
    #                      class. No effect other than visual.
    def __init__(self,theModel,dataPath='DATASET/DATA',gtPath='DATASET/GT',patchStep=1,multiClass=True,colorOrder=[1,0,2]):
        self._theModel_=theModel
        self._dataPath_=dataPath
        self._gtPath_=gtPath
        self._patchStep_=patchStep
        self._multiClass_=multiClass
        self._colorOrder_=colorOrder
        self._segmentedTransectNumber_=-1
        self.tPredict=-1
        self.theTransect=None
        self.segTransect=None
        self.theGT=None

    # Change patch step
    def set_patch_step(self,patchStep):
        self._patchStep_=patchStep
        self._segmentedTransectNumber_=-1
        self.theTransect=None
        self.segTransect=None
        self.theGT=None
        
    # Change multi class
    def set_multi_class(self,multiClass):
        self._multiClass_=multiClass
        self._segmentedTransectNumber_=-1
        self.theTransect=None
        self.segTransect=None
        self.theGT=None

    # Segments the specified transect according to the NN model
    def segment(self,transectNum):
        # Build the file names
        baseName='TRAN'+str(transectNum).zfill(2)+'.png'
        dataFileName=os.path.join(self._dataPath_,baseName)
        gtFileName=os.path.join(self._gtPath_,baseName)
        
        # Read the transect image
        self.theTransect=self._read_data_image_(dataFileName)
        # Patchify it and build a batch
        X=self._patchify_transect_(self.theTransect)
        # Segment the batch using the NN model
        tStart=time()
        X=self._segment_patches_(X)
        # Go back to an image
        self.segTransect=self._depatchify_transect_(X)
        self.tPredict=time()-tStart
        # Cut the original image to have the same size
        self.theTransect=self.theTransect[:,:self.segTransect.shape[1]]
        
        # Read the ground truth image and convert it to categorical
        self.theGT=self._read_gt_image_(gtFileName)
        # Cut the GT to have the same size as the data.
        self.theGT=self.theGT[:,:self.segTransect.shape[1],:]
        
        # Remember the segmented transect number
        self._segmentedTransectNumber_=transectNum
        
        return self.segTransect,self.theGT,self.theTransect
        
    # Evaluates the specified transect or the previously segmented one
    def evaluate(self,transectNum=None):
        # Check errors and if needed to segment
        if transectNum is None and self._segmentedTransectNumber_==-1:
            sys.exit('[ERROR] PLEASE SPECIFY A TRANSECT NUMBER OR SEGMENT IT BEFORE')
        if transectNum is not None and self._segmentedTransectNumber_!=transectNum:
            self.segment(transectNum)

        # Build the flattened, non categorical, vectors to compute accuracies
        flatSegmented=np.argmax(self.segTransect,axis=2).flatten()
        flatGT=np.argmax(self.theGT,axis=2).flatten()
        confusionMatrix=confusion_matrix(flatGT,flatSegmented)
        theAccuracy=np.sum(np.diag(confusionMatrix))/np.sum(confusionMatrix)
        
        return confusionMatrix,theAccuracy
        
    # Evaluates multiple transects
    def evaluate_multiple(self,transectList):
        tPredict=0
        confusionMatrix=np.zeros((3,3))
        for curTransect in transectList:
            curCM,stuff=self.evaluate(curTransect)
            confusionMatrix+=curCM
            tPredict+=self.tPredict
        theAccuracy=np.sum(np.diag(confusionMatrix))/np.sum(confusionMatrix)
        return confusionMatrix,theAccuracy,tPredict

    # Segments a full transect (i.e. two consecutive transects considered
    # board and starboard data)
    def segment_full_transect(self,transectNum,deadZone=80):
        # Compute top and bottom indexes, taking into account that a full
        # top part is even and bottom is odd.
        if transectNum%2!=0:
            transectNum-=1
        topTransect=transectNum
        bottomTransect=transectNum+1
        
        # Segment the transects
        segTop,gtTop,dataTop=self.segment(topTransect)
        segBottom,gtBottom,dataBottom=self.segment(bottomTransect)
        
        # Extract some parameters and check if top and bottom have the same
        # number of columns
        nRows=segTop.shape[0]
        nCols=segTop.shape[1]
        if nCols!=segBottom.shape[1]:
            sys.exit('[ERROR] TRANSECTS IN A FULL TRANSECT MUST HAVE THE SAME WIDTH')
        
        # Flip vertically the bottom one to have a proper representation
        segBottom=np.flip(segBottom,axis=0)
        gtBottom=np.flip(gtBottom,axis=0)
        dataBottom=np.flip(dataBottom,axis=0)
        
        # Create data structures
        fullSegmented=np.zeros((nRows*2+deadZone,nCols,3))
        fullGT=np.zeros((nRows*2+deadZone,nCols,3))
        fullTransect=np.zeros((nRows*2+deadZone,nCols))
        
        # Put the top and bottom at the proper positions
        fullSegmented[:nRows,:,:]=segTop
        fullSegmented[nRows+deadZone:,:,:]=segBottom
        fullGT[:nRows,:,:]=gtTop
        fullGT[nRows+deadZone:,:,:]=gtBottom
        fullTransect[:nRows,:]=dataTop
        fullTransect[nRows+deadZone:,:]=dataBottom
        
        return fullSegmented,fullGT,fullTransect

    # Builds an image from a set of patches
    def _depatchify_transect_(self,X):
        # Get the number of patches and patch size. Patches are assumed to be
        # square.
        nPatches=X.shape[0]
        patchSize=X.shape[1]
    
        # Compute the margin at each side of the central pixel
        patchMargin=self._get_patch_margin_(patchSize)
        
        # Create matrices to store the image and the overlap count
        nRows=nPatches*patchSize-(patchSize-self._patchStep_)*(nPatches-1)
        outImage=np.zeros((patchSize,nRows,3))
        outCount=np.zeros((patchSize,nRows,1))
        
        # Put the segmented patches in the proper position and count the
        # overlaps
        for i in range(nPatches):
            colStart=i*self._patchStep_
            colEnd=colStart+2*patchMargin+1
            outImage[:,colStart:colEnd]+=X[i]
            outCount[:,colStart:colEnd]+=1
            
        if 0 in outCount:
            sys.exit('[ERROR] UNEXPECTED ERROR DEPATCHIFYING TRANSECT. NON COVERED AREAS.')
            
        # Average the overlapping parts
        outImage/=outCount
        
        # Sort channels to have the proper representation and process labels
        # if necessary.
        return self._process_labels_(outImage[:,:,self._colorOrder_])
        
    # Segments a batch of patches using the provided model.
    def _segment_patches_(self,X):
        # Use the model to predict and process the labels if necessary
        return self._process_labels_(self._theModel_.predict(X))
        
    # Given a transect number, outputs the patches ready to be segmented by
    # theModel according to patchStep and the proper patchSize
    def _patchify_transect_(self,theTransect):
        # Patch size is assumed to be image height
        patchSize=theTransect.shape[0]
        
        # Compute the margin at each side of the central pixel
        patchMargin=self._get_patch_margin_(patchSize)

        # Build the patch array in keras batch format
        X=[]
        for i in range(patchMargin,theTransect.shape[1]-patchMargin-1,self._patchStep_):
            X.append(theTransect[:,i-patchMargin:i+patchMargin+1])
        X=np.array(X)[...,np.newaxis]
        return X
        
    # Helper function to get the patch margin from the patch size.
    def _get_patch_margin_(self,patchSize):
        # Patch size must be even
        if patchSize%2==0:
            sys.exit('[ERROR] IMAGE HEIGHT MUST BE ODD')
            
        # Compute the margin at each side of the central pixel
        return int((patchSize-1)/2)  

    # If necessary, converts the image from multi to single class
    def _process_labels_(self,theImage):
        labelAxis=len(theImage.shape)-1
        # If no probabilities wanted, convert to pure categorical
        if not self._multiClass_:
            # Select the class with the largest probability
            theImage=np.argmax(theImage,axis=labelAxis)
            # Convert to categorical
            theImage=to_categorical(theImage)
        return theImage
        
    # Reads a data image from file
    def _read_data_image_(self,fileName):
        theImage=imread(fileName)
        # Convert image if necessary
        if np.max(theImage)>1:
            theImage=theImage.astype('float')/255.0
        return theImage

    # Reads a ground truth image from file
    def _read_gt_image_(self,fileName):
        # Read the image from file
        theImage=self._read_data_image_(fileName)
        # Convert values to 0,1,2
        theImage=np.round(theImage*2.0).astype('int')
        # Convert to categorical
        theImage=to_categorical(theImage)
        # Resort channels according to desired visualization
        return theImage[:,:,self._colorOrder_]