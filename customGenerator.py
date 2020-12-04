from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt


def trainGenerator(trainPath, imageDir, maskDir, batchSize, VAL_SPLIT, seed=1 ):
    xTrain = ImageDataGenerator(validation_split=VAL_SPLIT)
    yTrain = ImageDataGenerator(validation_split=VAL_SPLIT)

    xGenerator = xTrain.flow_from_directory(
        seed=seed,
        batch_size=batchSize,
        class_mode=None,
        directory=trainPath,
        classes=[imageDir],
        color_mode = "rgb",
        subset='training'
    )
    yGenerator = yTrain.flow_from_directory(
        seed=seed,
        batch_size=batchSize,
        class_mode=None,
        directory=trainPath,
        classes=[maskDir],
        color_mode = "grayscale",
        subset='training'
    )

    trainZip = zip(xGenerator, yGenerator)
    for (x,y) in trainZip:
        img,mask = adjustData(x,y)
        yield (img,mask)


def validationGenerator(trainPath, imageDir, maskDir, batchSize, VAL_SPLIT, seed=1):
    xValidation = ImageDataGenerator(validation_split=VAL_SPLIT)
    yValidation = ImageDataGenerator(validation_split=VAL_SPLIT)

    xGenerator = xValidation.flow_from_directory(
        seed=seed,
        batch_size=batchSize,
        class_mode=None,
        directory=trainPath,
        classes=[imageDir],
        color_mode = "rgb",
        subset='validation'
    )
    yGenerator = yValidation.flow_from_directory(
        seed=seed,
        batch_size=batchSize,
        class_mode=None,
        directory=trainPath,
        classes=[maskDir],
        color_mode = "grayscale",
        subset='validation'
    )

    validationZip = zip(xGenerator, yGenerator)
    for (x,y) in validationZip:
        img,mask = adjustData(x,y)
        yield (img,mask)



def adjustData(img,mask, flag_multi_class=True):
    img = img / 255
    #mask = mask / 255
    #mask[mask > 0.5] = 1
    #mask[mask <= 0.5] = 0

    #plt.imshow(mask[0])
    #plt.show()
    #img = cv2.imread(img)
    #img = cv2.resize(src=img,dsize=(256,256), interpolation=cv2.INTER_NEAREST)

    #mask = cv2.imread(mask)
    #mask = cv2.resize(src=mask,dsize=(256,256), interpolation=cv2.INTER_NEAREST)

    return (img,mask)