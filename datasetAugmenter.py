import os
from tqdm import tqdm
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import time
from visualizer import visualize
from PIL import Image
import pandas as pd

DIR_IMG_SRC = "data\\img\\ori"
DIR_MASK_SRC = "data\\img\\mask"

MASK_FORMAT = ".png"
IMG_FORMAT = ".jpg"

N_IMG = len(os.listdir(DIR_IMG_SRC))

N_AUG_PER_IMG = 0

DATASET = pd.read_csv("data\\label\\dataset.csv", sep=',', index_col=0)
pathDfAugmented = "data\\label\\datasetAugmented.csv"

DATASET_AUGMENTED = []

def askInfos():
    global N_AUG_PER_IMG
    os.system('cls')
    print("##################")
    print("# DATA AUGMENTER #")
    print("##################\n")
    print("~~ Nombre d'images : " + str(N_IMG) + "\n")
    print("~~ Nombre de copy par image : ")
    newAugMultiplier = input()

    if(int(newAugMultiplier) == (0 or 1)):
        askInfos()

    print("~~ Nombre total après augmentation : " + str(N_IMG*(int(newAugMultiplier)+1)) + "\n")
    print("~~ Params OK ? o/n : ")
    confirm = input()


    if(confirm == "o"):
        N_AUG_PER_IMG = int(newAugMultiplier)
        launchAugmentation()
    elif(confirm == "n"):
        askInfos()


def launchAugmentation():
    transform = A.Compose([
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.5)
    ])

    for index, row in tqdm(DATASET.iterrows(), total=DATASET.shape[0]):
        rawImgPath = row['x_path'].split('.')[0]
        rawMaskPath = row['y_path'].split('.')[0]
        baseImage = cv2.imread(row['x_path'], cv2.IMREAD_COLOR)
        baseMask = cv2.imread(row['y_path'], cv2.IMREAD_GRAYSCALE)

        for i in range(N_AUG_PER_IMG):
            newImgPath = rawImgPath + "_aug_{:d}".format(i) + IMG_FORMAT
            newMaskPath = rawMaskPath + "_aug_{:d}".format(i) + MASK_FORMAT

            augmented = transform(image=baseImage, mask=baseMask)

            cv2.imwrite(newImgPath, augmented['image'])
            cv2.imwrite(newMaskPath, augmented['mask'])

            DATASET_AUGMENTED.append([newImgPath, newMaskPath])

    #print(DATASET_AUGMENTED)



    df = pd.DataFrame(DATASET_AUGMENTED, columns=['x_path', 'y_path'], dtype=str)
    globalDf = pd.concat([df, DATASET], ignore_index=True, sort=False, keys=['original', 'augmented'])

    globalDf = globalDf.sample(frac=1).reset_index(drop=True)

    # merge dataset et dataset augmented

    globalDf.to_csv(pathDfAugmented, sep=',')

if __name__ == "__main__":
    askInfos()





