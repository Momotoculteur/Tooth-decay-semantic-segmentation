import numpy as np
from glob import glob
from utils import pairwise, getPaletteColors
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from PIL import Image, ImageDraw
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.cm
from skimage.draw import polygon2mask
from scipy.ndimage.morphology import binary_fill_holes as imfill
import cv2
import sys

pathImgSrc = "data\\img\\ori"
pathImgDst = "data\\img\\mask"

dirImgSrc = os.path.join(os.getcwd(), pathImgSrc)
dirImgDst = os.path.join(os.getcwd(), pathImgDst)
labels = os.path.join(os.getcwd(), "data\\label\\annotations.json")

carrie = {
    'r': 255,
    'g': 255,
    'b': 255
}
test = {
    'r': 0,
    'g': 255,
    'b': 0
}

COLOR_DICT = getPaletteColors()

with open(labels) as json_file:
    data = json.load(json_file)

    # Pour chaque radio
    for id, currentImg in tqdm(enumerate(os.listdir(dirImgSrc)), "Conversion : ", total=len(os.listdir(dirImgSrc))):
        currentImgPath = os.path.join(dirImgSrc, currentImg)
        currentImgPathSave = os.path.join(dirImgDst, currentImg)
        im = Image.open(currentImgPath)
        width, height = im.size

        # METHODE CREER MASK V1
        mask = np.zeros(shape=(height, width, 1), dtype=np.uint8)


        # Pour chaque carrie dans une radio
        for item in data[currentImg]:
            # points de type polygon ou autre
            if (item['type'] != "meta"):
                res = []
                for a, b in pairwise(item['points']):
                    res.append((b, a))

                # METHODE CREER MASK V1
                tmpR = polygon2mask((height, width, 1), res).astype(int)
                #tmpR[tmpR == 1] = item['classId']
                tmpR[tmpR == 1] = 255
                tmpR[tmpR == 0] = 0
                mask[:, :, 0] = np.maximum(mask[:, :, 0], tmpR[:, :, 0])



        cv2.imwrite(currentImgPathSave, mask)

        # Afficher un mask
        #plt.imshow(mask)
        #plt.show()