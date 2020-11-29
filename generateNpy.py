import os
import numpy as np
import json
from PIL import Image
from skimage.draw import polygon2mask
from utils import pairwise, getPaletteColors
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


IMG_CHANNELS = 3




pathImgSrc = "data\\img\\ori"
pathImgDst = "data\\img\\mask"

dirImgSrc = os.path.join(os.getcwd(), pathImgSrc)
dirImgDst = os.path.join(os.getcwd(), pathImgDst)
labels = os.path.join(os.getcwd(), "data\\label\\annotations.json")
COLOR_MODE = "rgb"

carrie = {
    'r': 255,
    'g': 255,
    'b': 255
}
COLOR_DICT = getPaletteColors()

IMG_HEIGHT = 621
IMG_WIDTH = 815
X_train = np.zeros((len(os.listdir(dirImgSrc)), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(os.listdir(dirImgSrc)), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)


with open(labels) as json_file:
    data = json.load(json_file)

    for id, currentImg in tqdm(enumerate(os.listdir(dirImgSrc)), "Conversion : ", total=len(os.listdir(dirImgSrc))):
        currentImgPath = os.path.join(dirImgSrc, currentImg)
        currentImgPathSave = os.path.join(dirImgDst, currentImg)
        im = Image.open(currentImgPath)
        width, height = im.size
        mask = np.zeros(shape=(height, width, 3), dtype=np.uint8)

        for item in data[currentImg]:
            # points de type polygon ou autre
            if (item['type'] != "meta"):
                res = []
                for a, b in pairwise(item['points']):
                    res.append((b, a))

                tmpR = polygon2mask((height, width, 1), res).astype(int)
                tmpR[tmpR == 1] = COLOR_DICT[item['classId']-1]['colors']['r']
                tmpR[tmpR == 0] = 0
                mask[:, :, 0] = np.maximum(mask[:, :, 0], tmpR[:, :, 0])

                tmpG = polygon2mask((height, width, 1), res).astype(int)
                tmpG[tmpG == 1] = COLOR_DICT[item['classId']-1]['colors']['g']
                tmpG[tmpG == 0] = 0
                mask[:, :, 1] = np.maximum(mask[:, :, 1], tmpG[:, :, 0])

                tmpB = polygon2mask((height, width, 1), res).astype(int)
                tmpB[tmpB == 1] = COLOR_DICT[item['classId']-1]['colors']['b']
                tmpB[tmpB == 0] = 0
                mask[:, :, 2] = np.maximum(mask[:, :, 2], tmpB[:, :, 0])

        loadedImg = load_img(currentImgPath, color_mode=COLOR_MODE)
        loadedImg = img_to_array(loadedImg)
        X_train[id] = loadedImg / 255.
        Y_train[id] = mask / 255.

    np.save(os.path.join("data/numpy", "original.npy"), X_train)
    np.save(os.path.join("data/numpy", "mask.npy"), Y_train)
