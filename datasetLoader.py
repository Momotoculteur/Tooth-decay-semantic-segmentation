import keras
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
import sys

from utils import isMulticlassDataset


class DatasetLoader(keras.utils.Sequence):
    """
    Generateur custom pour charger, transformer et envoyer au DNN
    """

    def __init__(self, data, xLabel, yLabel, batchSize=2, shuffle=True, targetSize=(256, 256), nClass=1):
        self.xData = data[xLabel]
        self.yData = data[yLabel]
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.targetSize = targetSize
        self.nClass = nClass
        self.on_epoch_end()

    def __len__(self):
        """
        Nombre de batches par epoch
        @return:
        """
        return int(np.floor(len(self.xData) / self.batchSize))

    def on_epoch_end(self):
        """
        Appelé à chaque fin d'epoch.
        @return:
        """
        # [0,1,2,3,4... nb_image]
        self.indexes = np.arange(len(self.xData))

        # [2,4,1,3,0... nb_image]
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Génére un batch de data
        @param index:
        @return:
        """

        # Genere batchSize nombre d'ID de row de DATA (batchSize=2, [0,1])
        currentBatchIdsRow = self.indexes[index * self.batchSize:(index + 1) * self.batchSize]

        # Find list of IDs
        # list_IDs_temp = [self.xData[k] for k in indexes]

        # Generate data
        x, y = self.preprocessData(currentBatchIdsRow)

        return x, y

    def preprocessData(self, currentBatchIdsRow):
        """
        @param currentBatchIdsRow: ID des items du batch courant à créer, correspond au id du df dataset x/y
        @return:
        """

        xTrain = np.zeros((self.batchSize, *self.targetSize, 3), dtype=np.float32)
        yTrain = np.zeros((self.batchSize, *self.targetSize, 1), dtype=np.float32)

        # X = np.empty((self.batchSize, *self.targetSize, self.n_channels))
        # y = np.empty((self.batchSize), dtype=int)

        for i, rowId in enumerate(currentBatchIdsRow):
            xTmp = Image.open(self.xData.iloc[rowId]).convert('RGB')
            xTmp = xTmp.resize(size=self.targetSize, resample=Image.NEAREST)
            xTmp = np.asarray(xTmp, dtype=np.float32) / 255.
            xTrain[i, ] = xTmp

            yTmp = Image.open(self.yData.iloc[rowId]).convert('L')
            yTmp = yTmp.resize(size=self.targetSize, resample=Image.NEAREST)
            yTmp = np.asarray(yTmp, dtype=np.uint8)
            yTmp = np.expand_dims(yTmp, axis=-1)

            if isMulticlassDataset():
                yTmp = keras.utils.to_categorical(yTmp, num_classes=self.nClass)

            yTrain[i, ] = yTmp

        return xTrain, yTrain
