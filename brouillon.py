X_train = np.zeros((len(os.listdir(dirImg)), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(os.listdir(dirImg)), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

with open(labels) as json_file:
    data = json.load(json_file)

    # Pour chaque radio
    for id, currentImg in tqdm(enumerate(os.listdir(dirImg)), "Conversion : ", total=len(os.listdir(dirImg))):
        print("BONJOU===========")
        print(currentImg)
        print(id)
        currentImgPath = os.path.join(dirImg, currentImg)
        im = Image.open(currentImgPath)
        width, height = im.size
        print(im.size)
        mask = np.zeros((height, width, 1))

        # Pour chaque carrie dans une radio
        for item in data[currentImg]:
            # points de type polygon ou autre

            if (item['type'] != "meta"):
                res = []
                for a, b in pairwise(item['points']):
                    res.append((b, a))

                tmp = polygon2mask((height, width, 1), res)
                plt.imshow(tmp, cmap=matplotlib.cm.Greys_r)
                plt.show()
                '''
                if(item['classId'] == 1):
                    res = []
                    for a, b in pairwise(item['points']):
                        res.append([a,b])

                    tmp = polygon2mask((height, width, 1), res)
                    plt.imshow(tmp, cmap=matplotlib.cm.Greys_r)
                    plt.show()
                if(item['classId'] == 2):
                    polygon = np.array(item['points'])
                    tmp = polygon2mask((height, width, 1), polygon)
                    plt.imshow(tmp, cmap=matplotlib.cm.Greys_r)
                    plt.show()

                '''
                '''
                cv2.fillPoly(mask, *res, 1)
                mask = mask.astype(bool)
                print(mask)
                '''

                '''

                xs, ys = zip(*res)  # create lists of x and y values
                plt.figure()
                plt.plot(xs, ys)
                ax = plt.gca()  # get the axis
                ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
                ax.xaxis.tick_top()  # and move the X-Axis
                #ax.yaxis.set_ticks(np.arange(0, 1, 1))  # set y-ticks
                ax.yaxis.tick_left()  # remove right y-Ticks
                plt.show()
                '''
                # plt.show()  # if you need...

                # mask[res[:, 0], res[:, 1]] = 1
                # out = mask[imfill(mask)].sum()
                # plt.plot(out)
                # plt.show()

                # print(np.array([[e.x, e.y] for e in item['points']]))

                # print(tmp.shape)
                # mask = np.maximum(mask, tmp)

                # te = mark_boundaries(image=mask, label_img=res,
                # color=(1, 0, 0), background_label=255, mode='thick')
                # print(item['points'])

        # plt.imshow(mask, cmap=matplotlib.cm.Greys_r, interpolation='nearest')

        # plt.show()
        # print(mask)

        # Avec Skimage
        # loadedImg = io.imread(currentImgPath, as_gray=False, plugin='matplotlib')
        # loadedImg = resize(loadedImg, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        # print(loadedImg.shape)
        # Avec Pillow
        '''
        loadedImg = Image.open(currentImgPath)
        data = np.asarray(loadedImg, dtype=np.uint8)
        loadedImg.show()
        print(data.shape)
        '''

        # Avec Keras
        # loadedImg = load_img(currentImgPath, color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))
        loadedImg = load_img(currentImgPath, color_mode="grayscale")
        loadedImg = img_to_array(loadedImg)
        # loadedImg = resize(loadedImg, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
        plt.imshow(loadedImg, cmap=matplotlib.cm.Greys_r)
        plt.show()
        # print(loadedImg.shape)

        X_train[id] = loadedImg / 255.0
        # Y_train[index] = mask / 255.0

"""
my format of json

{
"image1.jpg":{
        "filename":"image1.jpg",
        "size":123456,
        "regions":[
            {
                "shape_attributes":{
                    "all_points_x":[
                        675,
                        808,
                        957,
                        967,
                        929,
                        791,
                        678,
                        703
                    ],
                    "all_points_y":[
                        543,
                        518,
                        492,
                        722,
                        760,
                        760,
                        760,
                        647
                    ],
                    "name":"polygon"
                },
                "region_attributes":{
                    "Object":"class1",
                }
            },...all other regions..{}]
    }, ..all other images..., {}
}

"""

import glob, os, cv2
import numpy as np
import multiprocessing as mp
from PIL import Image
from delaunay2D import Delaunay2D


def calculateDelauneyPoints(points):
    points = np.array(points).astype(int)
    points = [list(item) for item in points]
    dt = Delaunay2D()
    for s in points:
        dt.addPoint(s)
    coord, tris = dt.exportDT()
    return np.array(coord)


def maskImage(filename, polygons, o_list):
    root_path = '../data/'
    image = cv.imread(root_path + "all_images/" + filename)
    mask = np.zeros(shape=image.shape, dtype="uint8")
    i = 0
    for points in polygons:
        points = [(int(x), int(y)) for x, y in points]
        try:
            rect = calculateDelauneyPoints(points)
        except:
            continue
        if o_list[i] in ['class1']:
            cv.drawContours(mask, [rect], -1, (1, 1, 1), cv.FILLED)
        elif o_list[i] in ['class2']:
            cv.drawContours(mask, [rect], -1, (2, 2, 2), cv.FILLED)
        elif o_list[i] in ['class3']:
            cv.drawContours(mask, [rect], -1, (3, 3, 3), cv.FILLED)
        i += 1

    cv.imwrite(root_path + "dataset/images/" + os.path.splitext(filename)[0] + ".png", image)
    cv.imwrite(root_path + "dataset/masks/" + os.path.splitext(filename)[0] + ".png", mask)


def getRegionProperties(region):
    shape_attributes = region["shape_attributes"]
    region_attributes = region["region_attributes"]
    objects = region_attributes["Object"]
    regions = ['class1', 'class2', 'class3']
    all_points_x = shape_attributes["all_points_x"]
    all_points_y = shape_attributes["all_points_y"]
    coordinates = []
    for i in range(0, len(all_points_x)):
        coordinates.append((all_points_x[i], all_points_y[i]))

    return (objects, coordinates)


def parallelizePlotting(data, region_mappings, json_data):
    polygon_coordinates = {}
    polygons = []
    img_json_data = json_data[data]

    filename = img_json_data["filename"]

    # Open the original image here
    image_matrix = np.array(Image.open('../data/all_images/' + filename), dtype=np.uint8)
    region_data = img_json_data["regions"]
    objects_list = []

    for region in region_data:
        objects, coordinates = getRegionProperties(region)
        if coordinates is not None:
            polygons.append(coordinates)
            polygon_coordinates[objects] = coordinates
            objects_list.append(objects)

    # Masking the images
    maskImage(filename, polygons, objects_list)
    return (filename, polygon_coordinates)


def main():
    root_path = 'path'

    json_file = open(root_path + 'filename.json')
    json_data = json.load(json_file)

    # for multiprocessing
    pool = mp.Pool(mp.cpu_count() - 1)

    def resultCallback(item):
        return

    for data in json_data:
        pool.apply_async(parallelizePlotting, args=(data, json_data), callback=resultCallback)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()