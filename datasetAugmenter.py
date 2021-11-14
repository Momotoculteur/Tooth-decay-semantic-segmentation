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
    print("#                #")
    print("# onlyCarry:{} #".format(AUGMENT_ONLY_CARRY))
    print("##################\n")
    print("~~ Nombre d'images : " + str(N_IMG) + "\n")
    print("~~ Nombre de copy par image : ")
    newAugMultiplier = input()

    if(int(newAugMultiplier) == (0 or 1)):
        askInfos()

    if AUGMENT_ONLY_CARRY:
        print("~~ Nombre total après augmentation : " + str(N_IMG*(int(newAugMultiplier)+1)+N_IMG) + "\n")
    else:
        print("~~ Nombre total après augmentation : " + str(N_IMG*(int(newAugMultiplier)+1)) + "\n")
    print("~~ Params OK ? o/n : ")
    confirm = input()


    if(confirm == "o"):
        N_AUG_PER_IMG = int(newAugMultiplier)
        launchAugmentation()
    elif(confirm == "n"):
        askInfos()

def get_transform(image, mask, original_h, original_w):
    '''


    '''

    transform = A.Compose([
        A.OneOf([
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.HorizontalFlip(p=0.2)
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=15, p=0.2),
            A.Blur(blur_limit=15, p=0.2),
            A.GaussNoise(p=0.2),
        ], p=0.2),
    ])
    return transform

def launchAugmentation():
    transform = A.Compose([
        A.OneOf([
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.HorizontalFlip(p=0.2)
        ], p=1),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=15, p=0.2),
            A.Blur(blur_limit=15, p=0.2),
            A.GaussNoise(p=0.2),
        ], p=1),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.2),
            A.CLAHE(p=0.2),
            A.RandomGamma(p=0.2),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2),
        ], p=1)
    ])


    for index, row in tqdm(DATASET.iterrows(), total=DATASET.shape[0]):

        if (not AUGMENT_ONLY_CARRY) or (AUGMENT_ONLY_CARRY and row['at_least_one_carry'] == True):

            rawImgPath = row['x_path'].split('.')[0]
            rawMaskPath = row['y_path'].split('.')[0]
            baseImage = cv2.imread(row['x_path'], cv2.IMREAD_COLOR)
            baseMask = cv2.imread(row['y_path'], cv2.IMREAD_GRAYSCALE)

            height, width, channels = baseImage.shape



            for i in range(N_AUG_PER_IMG):

                newImgPath = rawImgPath + "_aug_{:d}".format(i) + IMG_FORMAT
                newMaskPath = rawMaskPath + "_aug_{:d}".format(i) + MASK_FORMAT

                #augmented = get_transform(image=baseImage, mask=baseMask, original_h=height, original_w=width)
                '''
                 A.OneOf([
            A.RandomSizedCrop(min_max_height=(50, 101), height=original_h, width=original_w, p=0.2),
            A.PadIfNeeded(min_height=original_h, min_width=original_w, p=0.2)
        ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.CLAHE(p=0.2),

                A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2),
        A.Cutout(num_holes=10, max_h_size=40, max_w_size=40, fill_value=0, p=0.2)
        
                            A.OneOf([
                        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
                        A.GridDistortion(p=0.3),
                        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.3)
                    ], p=0.3),
                    
                                        A.OneOf([
                        A.RandomSizedCrop(min_max_height=(50, 200), height=height, width=width, p=0.2),
                        A.PadIfNeeded(min_height=height, min_width=width, p=0.2)
                    ], p=0.2),
                '''


                augmented = transform(image=baseImage, mask=baseMask)

                cv2.imwrite(newImgPath, augmented['image'])
                cv2.imwrite(newMaskPath, augmented['mask'])

                DATASET_AUGMENTED.append([newImgPath, newMaskPath])



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

    askInfos()
    








