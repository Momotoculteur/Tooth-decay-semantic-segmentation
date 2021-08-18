import json
from PIL import Image, ImageDraw
import numpy as np
from skimage.draw import polygon2mask
from utils import pairwise
import matplotlib.pyplot as plt
from keras.models import load_model
from visualizer import visualize
from metrics import mean_iou, my_iou_metric, get_iou_vector, plot_best_iou_threshold, mean_iou_simple


json_raw_data_filepath = './dataTest/annotations.json'
image_test_dir = './dataTest/images/'

modelPath = 'C:\\model.h5'
model = load_model(modelPath, compile=False)


with open(json_raw_data_filepath) as raw_json_data:
    annotations_def = json.load(raw_json_data)

for image_name, images_values in annotations_def.items():
    image_path = image_test_dir + image_name
    img_loaded = Image.open(image_path).convert('RGB')
    #img_loaded = img_loaded_raw.resize(size=(256, 256))

    width, height = img_loaded.size
    mask = np.zeros(shape=(height, width), dtype=np.uint8)


    for carry in images_values:

        if (carry['type'] == "polygon"):
            #print(carry['points'])
            res = []
            for a, b in pairwise(carry['points']):
                res.append((b, a))

            # METHODE CREER MASK V1
            tmpMask = polygon2mask((height, width), res).astype(int)
            tmpMask[tmpMask == 1] = int(carry['classId'])
            mask[:, :] = np.maximum(mask[:, :], tmpMask[:, :])


    # fin traitement img
    # faire predict
    plt.imshow(mask)
    plt.show()

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
    print('IOU_________\n')
    tt = np.asarray(toSave_resized) # need convert en plus 'L' ?

    preed = res[np.newaxis,:,:]
    truue = tt[np.newaxis,:,:]

    #print(preed.shape)
    #(truue.shape)

    #print(another_iou_metric(res, tt))

    print('shape')
    print(res[:,:,0].shape)
    print(tt.shape)
    #print(tt.shape)
    #(256, 256, 1)
    #(256, 256, )

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



    print(plot_best_iou_threshold(tt, res[:,:,0]))


    visualize(res, toSave_resized)

