from segmentation_models import Xnet, Unet
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping
import os
from customLoss import bce_dice_loss
from customMetric import my_iou_metric
from customGenerator import trainGenerator, validationGenerator

# HYPER PARAMS
BATCH_SIZE = 2
TRAINSIZE_RATIO = 0.8
COLOR_MODE ="rgb"

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
EPOCH = 999
SAMPLE_TRAIN = int(NUM_SAMPLES * TRAINSIZE_RATIO)
SAMPLE_VALID = int(NUM_SAMPLES * (1-TRAINSIZE_RATIO))


STEP_SIZE_TRAIN = SAMPLE_TRAIN / BATCH_SIZE
STEP_SIZE_VALID = SAMPLE_VALID / BATCH_SIZE

#TRAIN_STEPS = len(os.listdir((os.path.join(train_path, "images")))) // batch_size
#VALIDATION_STEPS = len(os.listdir((os.path.join(val_path, "images")))) // batch_size

#metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
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
# EARLY STOP CALLBACK
earlyStopping = EarlyStopping(verbose=1,monitor='binary_accuracy', min_delta=0, patience=5, mode='auto')


# COMPILATION MODEL
model = Unet(backbone_name='vgg16',
             encoder_weights=None,
             decoder_block_type='transpose',
             classes=3)
#model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])



# TRAIN GENERATOR
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])
trainGen = trainGenerator(TRAIN_PATH, IMG_DIR_NAME, MASK_DIR_NAME, BATCH_SIZE, 0.2)
validationGen = validationGenerator(TRAIN_PATH, IMG_DIR_NAME, MASK_DIR_NAME, BATCH_SIZE, 0.2)



model.fit_generator(generator=trainGen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validationGen,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EPOCH,
                    callbacks=[csv_logger, savemodelCallback, earlyStopping])


# TRAIN NUMPY FILES
#imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
#model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])