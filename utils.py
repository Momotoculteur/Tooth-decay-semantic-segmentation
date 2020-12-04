import pandas as pd
import numpy as np

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
    print(len(df))
    if(len(df)>1):
        return True
    else:
        return False

def getClassesLabelList():
    classes = []
    df = pd.read_json(CLASSES_DEFINICATION_PATH)
    df = df.sort_values(['id'], ascending=[True])
    for index, row in df.iterrows():
        classes.append(row['name'])
    return classes

def getPaletteColors():
    colors = []
    df = pd.read_json(CLASSES_DEFINICATION_PATH)
    df = df.sort_values(['id'], ascending=[True])

    for index, row in df.iterrows():

        r,g,b = hexaToRgb(row['color'])
        colors.append({
            'id': row['id'],
            'colors': {
                'r': r,
                'g': g,
                'b': b
            },
            'name': row['name']
        })

    return np.array(colors)

def mask2img(mask):
    palette = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (0, 255, 255),
    }
    rows = mask.shape[0]
    cols = mask.shape[1]
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for j in range(rows):
        for i in range(cols):
            image[j, i] = palette[np.argmax(mask[j, i])]
    return image


def main():
    #getPaletteColors()
    getClassesLabelList()

if __name__ == '__main__':
    """
    # MAIN
    """
    main()