import os
from tqdm import tqdm
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import time
from visualizer import visualize
from PIL import Image
import pandas as pd
import json

DIR_IMG_SRC = "data\\img\\ori"
DIR_MASK_SRC = "data\\img\\mask"

MASK_FORMAT = ".png"
IMG_FORMAT = ".jpg"

N_AUG_PER_IMG = 0

DATASET = pd.read_csv("data\\label\\dataset.csv", sep=',', index_col=0)
pathDfAugmented = "data\\label\\datasetAugmented.csv"

DATASET_AUGMENTED = []

AUGMENT_ONLY_CARRY = True
N_IMG = 0



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

def get_transform(image, mask, original_h, original_w):
    transform = A.Compose([
        A.OneOf([
            A.RandomSizedCrop(min_max_height=(100, 600), height=original_h, width=original_w, p=0.3),
            A.PadIfNeeded(min_height=original_h, min_width=original_w, p=0.3)
        ], p=1),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.3)
        ], p=0.3),
        A.CLAHE(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3)
    ])
    return transform

def launchAugmentation():


    for index, row in tqdm(DATASET.iterrows(), total=DATASET.shape[0]):

        rawImgPath = row['x_path'].split('.')[0]
        rawMaskPath = row['y_path'].split('.')[0]
        baseImage = cv2.imread(row['x_path'], cv2.IMREAD_COLOR)
        baseMask = cv2.imread(row['y_path'], cv2.IMREAD_GRAYSCALE)

        height, width, channels = baseImage.shape

        '''
        
        for i in range(N_AUG_PER_IMG):

            newImgPath = rawImgPath + "_aug_{:d}".format(i) + IMG_FORMAT
            newMaskPath = rawMaskPath + "_aug_{:d}".format(i) + MASK_FORMAT

            #augmented = get_transform(image=baseImage, mask=baseMask, original_h=height, original_w=width)
            transform = A.Compose([
                A.OneOf([
                    A.RandomSizedCrop(min_max_height=(50, 200), height=height, width=width, p=0.3),
                    A.PadIfNeeded(min_height=height, min_width=width, p=0.3)
                ], p=1),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
                    A.GridDistortion(p=0.3),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.3)
                ], p=0.3),
                A.CLAHE(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.3)
            ])

            augmented = transform(image=baseImage, mask=baseMask)

            cv2.imwrite(newImgPath, augmented['image'])
            cv2.imwrite(newMaskPath, augmented['mask'])

            DATASET_AUGMENTED.append([newImgPath, newMaskPath])
        '''

    #print(DATASET_AUGMENTED)



    df = pd.DataFrame(DATASET_AUGMENTED, columns=['x_path', 'y_path'], dtype=str)
    globalDf = pd.concat([df, DATASET], ignore_index=True, sort=False, keys=['original', 'augmented'])

    globalDf = globalDf.sample(frac=1).reset_index(drop=True)

    # merge dataset et dataset augmented

    globalDf.to_csv(pathDfAugmented, sep=',')


if __name__ == "__main__":
    if AUGMENT_ONLY_CARRY:
        N_IMG = len(DATASET[DATASET['at_least_one_carry'] == True])
    else:
        N_IMG = len(DATASET)

    #askInfos()
    








