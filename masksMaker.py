import numpy as np
from utils import pairwise
from tqdm import tqdm
import os
import json
from PIL import Image, ImageDraw
from skimage.draw import polygon2mask
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import sys



pathImgSrc = "data\\img\\ori"
pathImgDst = "data\\img\\mask"
pathDf = "data/label/dataset.csv"
MASK_FORMAT = ".png"

dirImgSrc = os.path.join(os.getcwd(), pathImgSrc)
dirImgDst = os.path.join(os.getcwd(), pathImgDst)
labels = os.path.join(os.getcwd(), "data\\label\\annotations.json")

dataDf = []
np.set_printoptions(threshold=sys.maxsize)
with open(labels) as json_file:
    data = json.load(json_file)

    # Pour chaque radio
    for id, currentImg in tqdm(enumerate(os.listdir(dirImgSrc)), "Conversion : ", total=len(os.listdir(dirImgSrc))):
        currentImgPath = os.path.join(dirImgSrc, currentImg)
        currentMask = currentImg.split('.')[0]
        currentMask = currentMask + MASK_FORMAT
        currentImgPathSave = os.path.join(dirImgDst, currentMask)
        im = Image.open(currentImgPath)
        width, height = im.size

        # METHODE CREER MASK V1
        mask = np.zeros(shape=(height, width), dtype=np.uint8)


        # Pour chaque carrie dans une radio
        for item in data[currentImg]:
            # points de type polygon ou autre
            if (item['type'] == "polygon"):
                res = []
                for a, b in pairwise(item['points']):
                    res.append((b, a))

                # METHODE CREER MASK V1
                tmpMask = polygon2mask((height, width), res).astype(int)
                tmpMask[tmpMask == 1] = int(item['classId'])
                mask[:, :] = np.maximum(mask[:, :], tmpMask[:, :])



        #cv2.imwrite(currentImgPathSave, mask, params=cv2.IMREAD_GRAYSCALE)
        toSave = Image.fromarray(mask, 'L')

        toSave.save(currentImgPathSave)

        #GEN NUMPY FILES
        '''
        
        newName = currentImg.split(".")[0] + ".npy"
        toSave = os.path.join(dirImgDst, newName)
        np.save(toSave, mask)
        '''

        dataDf.append([os.path.join(pathImgSrc, currentImg),os.path.join(pathImgDst, currentImg)])

    df = pd.DataFrame(dataDf, columns=['x_path', 'y_path'], dtype=str)
    df.to_csv(pathDf, sep=',')
