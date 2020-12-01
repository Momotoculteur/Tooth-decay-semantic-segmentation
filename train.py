from generators import trainGenerator
from segmentation_models import Xnet
from keras.callbacks import ModelCheckpoint, TensorBoard,CSVLogger
import os

# HYPER PARAMS
BATCH_SIZE = 2
TRAIN_PATH = "data/img"
IMG_DIR_NAME = "ori"
MASK_DIR_NAME = "mask"

DIR_MODEL = "result/model"
MODEL_NAME = "model.h5"

DIR_LOGS = "result/log/metric"
LOGS_FILE_NAME = "metrics.csv"

DIR_TRAINED_MODEL = os.path.join(DIR_MODEL, MODEL_NAME)
DIR_TRAINED_LOGS = os.path.join(DIR_LOGS, LOGS_FILE_NAME)

NUM_SAMPLES = len(os.listdir(os.path.join(os.getcwd(), TRAIN_PATH, IMG_DIR_NAME)))
EPOCH = 4

# CALLBACK sauvegarde model
savemodelCallback = ModelCheckpoint(DIR_TRAINED_MODEL,
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='auto',
                                      period=1,
                                      monitor='binary_accuracy')
                                      #monitor='val_acc')
# TENSORBOARD CALLBACL
#logsCallback = TensorBoard(log_dir=DIR_TRAINED_MODEL_LOGS, histogram_freq=0, write_graph=True, write_images=True)
# CSV METRICS CALLBACK
csv_logger = CSVLogger(DIR_TRAINED_LOGS, append=True, separator=',')


# COMPILATION MODEL
model = Xnet(backbone_name='resnet50',
             encoder_weights='imagenet',
             decoder_block_type='transpose',
             classes=3)
model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])



# TRAIN GENERATOR
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])
trainGen = trainGenerator(TRAIN_PATH, IMG_DIR_NAME, MASK_DIR_NAME, BATCH_SIZE)
model.fit_generator(trainGen,
                    steps_per_epoch=NUM_SAMPLES/BATCH_SIZE,
                    epochs=EPOCH,
                    callbacks=[csv_logger, savemodelCallback])


# TRAIN NUMPY FILES
#imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
#model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])