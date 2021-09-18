from visualizer import visualize
from PIL import Image

imagePath = '.\\predict\\test1.jpg'
maskPath = '.\\predict\\mask1.png'
prediPath = '.\\predict\\predict1.png'

imageBase = Image.open(imagePath).convert('RGB')
imageBase = imageBase.resize(size=(256, 256))

mask = Image.open(maskPath)
mask = mask.resize(size=(256, 256))

visualize(imageBase, mask)