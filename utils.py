import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


CLASSES_DEFINICATION_PATH = 'data\\label\\classes.json'

def pairwise(it):
    it = iter(it)
    while True:
        try:
            yield next(it), next(it)
        except StopIteration:
            # no more elements in the iterator
            return

def hexaToRgb(codeHexa):
    r_hex = codeHexa[1:3]
    g_hex = codeHexa[3:5]
    b_hex = codeHexa[5:7]
    return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)

def isMulticlassDataset():
    df = pd.read_json(CLASSES_DEFINICATION_PATH)
    if(len(df)>1):
        return True
    else:
        return False

def getClassesLabelList():
    classes = []
    if (isMulticlassDataset()):
        classes.append('background')
    df = pd.read_json(CLASSES_DEFINICATION_PATH)
    df = df.sort_values(['id'], ascending=[True])
    for index, row in df.iterrows():
        classes.append(row['name'])
    return classes

def getPaletteColors():
    # 0 : background
    palette = [(0,0,0)]

    df = pd.read_json(CLASSES_DEFINICATION_PATH)
    df = df.sort_values(['id'], ascending=[True])

    for index, row in df.iterrows():
        r,g,b = hexaToRgb(row['color'])
        palette.append(
            tuple((r, g, b))
        )
    palette.append((0,255,0))
    print("________________________")
    #print(palette[0])
    return palette

def mask2img(mask):
    #Pour du Integer labeling
    '''
    palette = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (0, 255, 255),
    }
    '''
    palette = getPaletteColors()
    rows = mask.shape[0]
    cols = mask.shape[1]
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for j in range(rows):
        for i in range(cols):
            image[j, i] = palette[np.argmax(mask[j, i])]
    return image


def mask2imgMultipleClasses(mask):
    #pour du one hot encoding
    '''
    palette = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (0, 255, 255),
    }
    '''
    np.set_printoptions(threshold=sys.maxsize)
    palette = getPaletteColors()
    rows = mask.shape[0]
    cols = mask.shape[1]
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for j in range(rows):
        for i in range(cols):
            try:
                image[j, i] = palette[mask[j, i]]
            except IndexError:
                pass
                #print(mask[j, i])
    return image



def plot_roc_curve(y_true, y_pred, title):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)

    fig, ax = plt.subplots(1, 1)
    ax.plot(fpr_keras, tpr_keras, label='ROC curve (area = %0.3f)' % auc_keras)
    ax.plot([0, 1], [0, 1], 'k--')
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.show()

    print('#######')
    print(title)
    print('fpr : {}'.format(fpr_keras))
    print('tpr : {}'.format(tpr_keras))
    print('auc : {}'.format(auc_keras))
    print('#######')


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

if __name__ == '__main__':
    """
    # MAIN
    """
    #getPaletteColors()