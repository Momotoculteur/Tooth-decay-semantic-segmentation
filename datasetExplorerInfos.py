import pandas as pd
import json

def getDatasetInfos(annot_file_path):
    with open(annot_file_path) as json_data:
        data = json.load(json_data)


    COMPTEUR = {
        'TOTAL': {
            'TOTAL': 0,
            'PM': 0,
            'CP': 0,
            'IC': 0,
            'I': 0,
            'E': 0,
            'O': 0
        },
        'NO_CARRY': {
            'TOTAL': 0,
            'PM': 0,
            'CP': 0,
            'IC': 0,
            'I': 0,
            'E': 0,
            'O': 0
        },
        'CARRY': {
            'TOTAL': 0,
            'PM': 0,
            'CP': 0,
            'IC': 0,
            'I': 0,
            'E': 0,
            'O': 0
        }
    }

    for key in data:
        carry = 'NO_CARRY'
        tooth_type = ''
        item = data[key]

        for attribute in item:
            if attribute['type'] == "tag":
                tooth_type = attribute['name']

            if attribute['type'] == "polygon":
                carry = 'CARRY'

        if tooth_type != '':
            COMPTEUR[carry][tooth_type] = COMPTEUR[carry][tooth_type] + 1
            COMPTEUR['TOTAL'][tooth_type] = COMPTEUR['TOTAL'][tooth_type] + 1

        COMPTEUR['TOTAL']['TOTAL'] = COMPTEUR['TOTAL']['TOTAL'] + 1

        if carry == 'NO_CARRY':
            COMPTEUR['NO_CARRY']['TOTAL'] = COMPTEUR['NO_CARRY']['TOTAL'] + 1
        if carry == 'CARRY':
            COMPTEUR['CARRY']['TOTAL'] = COMPTEUR['CARRY']['TOTAL'] + 1



    print(json.dumps(COMPTEUR, indent=4))


if __name__ == '__main__':
    """
    # MAIN
    """

    PATH_TO_ANNOTATIONS = './data/label/annotations.json'
    PATH_TO_ANNOTATIONS_TEST = './dataTest/annotations.json'

    getDatasetInfos(PATH_TO_ANNOTATIONS_TEST)
