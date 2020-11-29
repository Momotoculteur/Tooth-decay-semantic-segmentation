from generators import trainGenerator
from segmentation_models import Xnet
from keras.callbacks import ModelCheckpoint
import os


BATCH_SIZE = 2
TRAIN_PATH = "data/img"
NUM_SAMPLES = len(os.listdir(os.path.join(os.getcwd(), TRAIN_PATH, "ori")))
EPOCH = 4
print("NB ITEMMMMMMMMMMM")
print(NUM_SAMPLES)

# COMPILATION MODEL
model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose', classes=3)
model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

# TRAIN GENERATOR
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])
trainGen = trainGenerator(TRAIN_PATH, 'ori', 'mask', BATCH_SIZE)
model.fit_generator(trainGen, steps_per_epoch=NUM_SAMPLES/BATCH_SIZE, epochs=EPOCH)


# TRAIN NUMPY FILES
#imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
#model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])