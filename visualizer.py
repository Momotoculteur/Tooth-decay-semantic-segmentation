import matplotlib.pyplot as plt
from utils import mask2imgMultipleClasses


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        #ax[1, 0].imshow(original_mask)
        ax[1, 0].imshow(mask2imgMultipleClasses(original_mask))
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        #ax[1, 1].imshow(mask)
        ax[1, 1].imshow(mask2imgMultipleClasses(mask))
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


    plt.show()