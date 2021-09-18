from visualizer import visualize
from PIL import Image
import numpy as np
from utils import plot_confusion_matrix
import matplotlib.pyplot as plt


'''
imagePath = '.\\predict\\test1.jpg'
maskPath = '.\\predict\\mask1.png'
prediPath = '.\\predict\\predict1.png'

imageBase = Image.open(imagePath).convert('RGB')
imageBase = imageBase.resize(size=(256, 256))

mask = Image.open(maskPath)
mask = mask.resize(size=(256, 256))

visualize(imageBase, mask)
'''



matrix_data = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])

plot_confusion_matrix(cm=matrix_data, normalize=True, target_names=['Background', 'Carry'],
                      title='Confusion Matrix', cmap=plt.cm.Blues)
