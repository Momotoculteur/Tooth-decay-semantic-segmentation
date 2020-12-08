import os
from tqdm import tqdm
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import time
from visualizer import visualize
from PIL import Image


DIR_IMG_SRC = "data\\img\\ori"
DIR_MASK_SRC = "data\\img\\mask"

MASK_FORMAT = ".png"

N_IMG = len(os.listdir(DIR_IMG_SRC))

N_AUG_PER_IMG = 0

def askInfos():
    global N_AUG_PER_IMG
    os.system('cls')
    print("##################")
    print("# DATA AUGMENTER #")
    print("##################\n")
    print("~~ Nombre d'images : " + str(N_IMG) + "\n")
    print("~~ Nombre de copy par image : ")
    newAugMultiplier = input()
    print("~~ Nombre total après augmentation : " + str(N_IMG*int(newAugMultiplier)) + "\n")
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

    for id, imgName in tqdm(enumerate(os.listdir(DIR_IMG_SRC)), "Conversion : ", total=N_IMG):
        maskName = imgName.split('.')[0]
        maskName = maskName + MASK_FORMAT

        image = cv2.imread(os.path.join(DIR_IMG_SRC,imgName), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(DIR_MASK_SRC,maskName), cv2.IMREAD_GRAYSCALE)

        augmented = transform(image=image, mask=mask)

        visualize(image, mask, augmented['image'], augmented['mask'])




if __name__ == "__main__":
    #askInfos()
    launchAugmentation()





