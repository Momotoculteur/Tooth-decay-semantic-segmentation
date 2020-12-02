from keras.preprocessing.image import ImageDataGenerator

def trainGenerator(trainPath, imageDir, maskDir, batchSize, VAL_SPLIT, seed=1 ):
    xTrain = ImageDataGenerator(rescale=1./255.,
                                validation_split=VAL_SPLIT,
                                rotation_range=0.2,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                horizontal_flip=True,
                                fill_mode='nearest'
                                )
    yTrain = ImageDataGenerator(rescale=1./255.,
                                validation_split=VAL_SPLIT,rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

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
        color_mode = "rgb",
        subset='training'
    )

    trainZip = zip(xGenerator, yGenerator)
    for (x,y) in trainZip:
        #img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (x,y)


def validationGenerator(trainPath, imageDir, maskDir, batchSize, VAL_SPLIT, seed=1):
    xValidation = ImageDataGenerator(rescale=1./255.,
                                     validation_split=VAL_SPLIT)
    yValidation = ImageDataGenerator(rescale=1./255.,
                                     validation_split=VAL_SPLIT)

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
        color_mode = "rgb",
        subset='validation'
    )

    validationZip = zip(xGenerator, yGenerator)
    for (x,y) in validationZip:
        #img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (x,y)