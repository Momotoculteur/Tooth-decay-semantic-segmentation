from keras.preprocessing.image import ImageDataGenerator

def trainGenerator(trainPath, imageDir, maskDir, batchSize, seed=1):
    image_datagen = ImageDataGenerator(rescale=1./255.)
    mask_datagen = ImageDataGenerator(rescale=1./255.)

    image_generator = image_datagen.flow_from_directory(
        seed=seed,
        batch_size=batchSize,
        class_mode=None,
        directory=trainPath,
        classes=[imageDir]

    )
    mask_generator = mask_datagen.flow_from_directory(
        seed=seed,
        batch_size=batchSize,
        class_mode=None,
        directory=trainPath,
        classes=[maskDir]

    )

    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        #img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)
