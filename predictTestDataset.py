import json
from PIL import Image, ImageDraw
import numpy as np
from skimage.draw import polygon2mask
from utils import pairwise, plot_roc_curve, plot_confusion_matrix
import matplotlib.pyplot as plt
from keras.models import load_model
from visualizer import visualize
from metrics import mean_iou, my_iou_metric, get_iou_vector, plot_best_iou_threshold, mean_iou_simple
from sklearn.metrics import roc_curve, auc, confusion_matrix
from losses import dice_coef, np_dice_coef

TOTAL_IMAGE = 0
TOTAL_FP = 0
TOTAL_TP = 0
TOTAL_AUC = 0

FINAL_TRUE = np.array([])
FINAL_PRED = np.array([])

SCORES = {
    'GROUP_TYPE': ['PM', 'CP', 'IC', 'I', 'E', 'O', 'TOTAL'],
    'COUNT': {
        'TOTAL': 0,
        'PM': 0,
        'CP': 0,
        'IC': 0,
        'I': 0,
        'E': 0,
        'O': 0
    },
    'Y_TRUE': {
        'TOTAL': np.array([]),
        'PM': np.array([]),
        'CP': np.array([]),
        'IC': np.array([]),
        'I': np.array([]),
        'E': np.array([]),
        'O': np.array([])
    },
    'Y_PRED': {
        'TOTAL': np.array([]),
        'PM': np.array([]),
        'CP': np.array([]),
        'IC': np.array([]),
        'I': np.array([]),
        'E': np.array([]),
        'O': np.array([])
    }
}

json_raw_data_filepath = './dataTest/annotations.json'
image_test_dir = './dataTest/images/'

modelPath = 'C:\\model.h5'
model = load_model(modelPath, compile=False)

with open(json_raw_data_filepath) as raw_json_data:
    annotations_def = json.load(raw_json_data)

for image_name, images_values in annotations_def.items():
    image_path = image_test_dir + image_name
    img_loaded = Image.open(image_path).convert('RGB')
    # img_loaded = img_loaded_raw.resize(size=(256, 256))

    width, height = img_loaded.size
    mask = np.zeros(shape=(height, width), dtype=np.uint8)

    tooth_type = ''

    for carry in images_values:

        if carry['type'] == "tag":
            tooth_type = carry['name']

        if (carry['type'] == "polygon"):
            # print(carry['points'])
            res = []
            for a, b in pairwise(carry['points']):
                res.append((b, a))

            # METHODE CREER MASK V1
            tmpMask = polygon2mask((height, width), res).astype(int)
            tmpMask[tmpMask == 1] = int(carry['classId'])
            mask[:, :] = np.maximum(mask[:, :], tmpMask[:, :])

    # fin traitement img
    # faire predict

    # SHOW
    # plt.imshow(mask)
    # plt.show()

    img_loaded = img_loaded.resize(size=(256, 256))

    img_to_predict = np.asarray(img_loaded, dtype=np.float32) / 255.
    dimension = img_to_predict.shape
    img_to_predict = img_to_predict.reshape(1, dimension[0], dimension[1], dimension[2])
    prediction = model.predict(img_to_predict)
    res = np.asarray(prediction[0] * 100)
    res[res >= 0.95] = 1
    res[res < 0.95] = 0
    toSave = Image.fromarray(mask, 'L')
    toSave_resized = toSave.resize(size=(256, 256))
    mask_resize = np.asarray(toSave_resized)  # need convert en plus 'L' ?

    preed = res[np.newaxis, :, :]
    truue = mask_resize[np.newaxis, :, :]

    print("res  :  " + str(res.shape))
    print("mask_resize  :  " + str(mask_resize.shape))
    print("preed  :  " + str(preed.shape))
    print("truue  :  " + str(truue.shape))
    '''
    res: (256, 256, 1)
    mask_resize: (256, 256)
    preed: (1, 256, 256, 1)
    truue: (1, 256, 256)
    '''

    final_pred = res[:, :, 0]  # (256, 256)
    final_true = mask_resize  # (256, 256)

    print("DICE COEF")
    print(np_dice_coef(final_true.flatten(), final_pred.flatten()))

    # print(preed.shape)
    # (truue.shape)

    # print(another_iou_metric(res, tt))

    '''
    res_custom = res.reshape(1, res.shape[0], res.shape[1], 1)
    predict_custom = tt.reshape(1, tt.shape[0], tt.shape[1], 1)
    print(res_custom.shape)
    print(predict_custom.shape)
    print(mean_iou(predict_custom,res_custom))
    '''
    '''

    res_custom = res.reshape(1, 1, res.shape[0], res.shape[1])
    predict_custom = tt.reshape(1, 1, tt.shape[0], tt.shape[1])
    print(res_custom.shape)
    print(predict_custom.shape)
    print(get_iou_vector(predict_custom, res_custom))
    '''

    # VIZUALIIIIIZE
    # print(plot_best_iou_threshold(mask_resize, res[:,:,0]))
    # visualize(res, toSave_resized)

    '''
    ######  CALCULATE SCORES #############
    '''
    SCORES['COUNT']['TOTAL'] = SCORES['COUNT']['TOTAL'] + 1
    SCORES['COUNT'][tooth_type] = SCORES['COUNT'][tooth_type] + 1

    SCORES['Y_TRUE']['TOTAL'] = np.concatenate([SCORES['Y_TRUE']['TOTAL'], final_true.flatten()])
    SCORES['Y_PRED']['TOTAL'] = np.concatenate([SCORES['Y_PRED']['TOTAL'], final_pred.flatten()])

    SCORES['Y_TRUE'][tooth_type] = np.concatenate([SCORES['Y_TRUE'][tooth_type], final_true.flatten()])
    SCORES['Y_PRED'][tooth_type] = np.concatenate([SCORES['Y_PRED'][tooth_type], final_pred.flatten()])

    print('========')

'''
######  PLOT ROC/AUC CURVE #############
'''
for group_type in SCORES['GROUP_TYPE']:
    print("=============  GROUPE {} =====================".format(group_type))

    plot_roc_curve(SCORES['Y_TRUE'][group_type], SCORES['Y_PRED'][group_type], group_type)


    print('DICE : {}'.format(np_dice_coef(SCORES['Y_TRUE'][group_type], SCORES['Y_PRED'][group_type])))


    matrix_data = confusion_matrix(SCORES['Y_TRUE'][group_type], SCORES['Y_PRED'][group_type])
    plot_confusion_matrix(cm=matrix_data, normalize=True, target_names=['Background', 'Carry'], title='Confusion Matrix for {}'.format(group_type), cmap=plt.cm.Blues)

    tn, fp, fn, tp = confusion_matrix(SCORES['Y_TRUE'][group_type], SCORES['Y_PRED'][group_type]).ravel()

    sensivity = tp / (tp+fn)
    specificity = tn / (tn+fp)

    print("SENSIBILITE : {}".format(sensivity))
    print("SPECIFICITE : {}".format(specificity))
