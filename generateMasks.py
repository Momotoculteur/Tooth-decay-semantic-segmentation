import numpy as np
from utils import pairwise, getPaletteColors
from tqdm import tqdm
import os
import json
from PIL import Image, ImageDraw
from skimage.draw import polygon2mask
import cv2
import pandas as pd

pathImgSrc = "data\\img\\ori"
pathImgDst = "data\\img\\mask"
pathDf = "data/label/dataset.csv"

dirImgSrc = os.path.join(os.getcwd(), pathImgSrc)
dirImgDst = os.path.join(os.getcwd(), pathImgDst)
labels = os.path.join(os.getcwd(), "data\\label\\annotations.json")

dataDf = []

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
            if (item['type'] == "polygon"):
                res = []
                for a, b in pairwise(item['points']):
                    res.append((b, a))

                # METHODE CREER MASK V1
                tmpMask = polygon2mask((height, width, 1), res).astype(int)
                tmpMask[tmpMask == 1] = item['classId']
                mask[:, :, 0] = np.maximum(mask[:, :, 0], tmpMask[:, :, 0])



        cv2.imwrite(currentImgPathSave, mask)
        '''
        GEN NUMPY FILES
        newName = currentImg.split(".")[0] + ".npy"
        toSave = os.path.join(dirImgDst, newName)
        np.save(toSave, mask)
        '''

        dataDf.append([os.path.join(pathImgSrc, currentImg),os.path.join(pathImgDst, currentImg)])

        # Afficher un mask
        #plt.imshow(mask)
        #plt.show()
    df = pd.DataFrame(dataDf, columns=['x_path', 'y_path'], dtype=str)
    df.to_csv(pathDf, sep=',')
