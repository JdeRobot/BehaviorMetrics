import numpy as np
import cv2
import os
import glob
import json

def create_dataset():
    # We check if dataset exists
    foldername = 'Dataset/Train_balanced_bbdd_w'
    if not os.path.exists(foldername): os.makedirs(foldername)
    folder_images = foldername + '/' + 'Images'
    if not os.path.exists(folder_images): os.makedirs(folder_images)
    foldername = 'Dataset/Train_balanced_bbdd_v'
    if not os.path.exists(foldername): os.makedirs(foldername)
    folder_images = foldername + '/' + 'Images'
    if not os.path.exists(folder_images): os.makedirs(folder_images)
    with open('Dataset/Train_balanced_bbdd_w/train.json', 'a') as file:
        json.dump('', file)
    with open('Dataset/Train_balanced_bbdd_v/train.json', 'a') as file:
        json.dump('', file)


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        array_imgs.append(img)

    return array_imgs


def parse_w_json(data):
    array_class = []
    # We process json
    data_parse = data.split('"class2": ')[1:]
    for d in data_parse:
        classification = d.split(', "class3":')[0]
        array_class.append(classification)

    return array_class


def parse_v_json(data):
    array_class = []
    # We process json
    data_parse = data.split('"class3": ')[1:]
    for d in data_parse:
        classification = d.split(', "w":')[0]
        array_class.append(classification)

    return array_class


def check_empty_directory(directory):
    # We check if the directory is empty
    empty = False
    for dirName, subdirList, fileList in os.walk(directory):
        if len(fileList) == 0:
            empty = True

    return empty


def get_number_image(path):
    list_images = glob.glob(path + '*')
    sort_images = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))
    last_number = sort_images[len(sort_images)-1].split('/')[3].split('.png')[0]
    number = int(last_number) + 1
    return number


def save_image(img, folder_images):
    # We save images
    empty = check_empty_directory(folder_images)
    if empty:
        number = 1
    else:
        number = get_number_image(folder_images)
    name_image = folder_images + str(number) + '.png'
    cv2.imwrite(name_image, img)


def count_classes_w(class_w, array_num_classes):
    # Array with number of data for each class (w)
    # [num_radically_left, num_moderately_left, num_slightly_left, num_slight,
    # num_slightly_right, num_moderately_rigth, num_radically_right]
    if class_w == 'radically_left' or class_w == '"radically_left"':
        array_num_classes[0] = array_num_classes[0] + 1
        index = 0
    elif class_w == 'moderately_left' or class_w == '"moderately_left"':
        array_num_classes[1] = array_num_classes[1] + 1
        index = 1
    elif class_w == 'slightly_left' or class_w == '"slightly_left"':
        array_num_classes[2] = array_num_classes[2] + 1
        index = 2
    elif class_w == 'slight' or class_w == '"slight"':
        array_num_classes[3] = array_num_classes[3] + 1
        index = 3
    elif class_w == 'slightly_right' or class_w == '"slightly_right"':
        array_num_classes[4] = array_num_classes[4] + 1
        index = 4
    elif class_w == 'moderately_right' or class_w == '"moderately_right"':
        array_num_classes[5] = array_num_classes[5] + 1
        index = 5
    elif class_w == 'radically_right' or class_w == '"radically_right"':
        array_num_classes[6] = array_num_classes[6] + 1
        index = 6
    return array_num_classes, index


def count_classes_v(class_v, array_num_classes):
    # Array with number of data for each class (v)
    # [slow, moderate, fast, very_fast]
    if class_v == 'slow' or class_v == '"slow"':
        array_num_classes[0] = array_num_classes[0] + 1
        index = 0
    elif class_v == 'moderate' or class_v == '"moderate"':
        array_num_classes[1] = array_num_classes[1] + 1
        index = 1
    elif class_v == 'fast' or class_v == '"fast"':
        array_num_classes[2] = array_num_classes[2] + 1
        index = 2
    elif class_v == 'very_fast' or class_v == '"very_fast"':
        array_num_classes[3] = array_num_classes[3] + 1
        index = 3
    return array_num_classes, index


def create_balanced_data_w(data, array_class_w, array_images):
    filename = 'Dataset/Train_balanced_bbdd_w/train.json'
    output = ''
    array_num_classes = np.zeros(7)
    max_num = 590

    with open(filename, "r+") as f:
        data_parse = data.split('{')[1:]

        for i in range(0, len(data_parse)):
            array_num_classes, index = count_classes_w(array_class_w[i], array_num_classes)
            if array_num_classes[index] <= max_num:
                output = output + '{' + data_parse[i]
                save_image(array_images[i], 'Dataset/Train_balanced_bbdd_w/Images/')

        f.seek(0)
        f.write(output)
        f.truncate()


def create_balanced_data_v(data, array_class_v, array_images):
    filename = 'Dataset/Train_balanced_bbdd_v/train.json'
    output = ''
    array_num_classes = np.zeros(4)
    max_num = 1162

    with open(filename, "r+") as f:
        data_parse = data.split('{')[1:]

        for i in range(0, len(data_parse)):
            array_num_classes, index = count_classes_v(array_class_v[i], array_num_classes)
            if array_num_classes[index] <= max_num:
                output = output + '{' + data_parse[i]
                save_image(array_images[i], 'Dataset/Train_balanced_bbdd_v/Images/')

        f.seek(0)
        f.write(output)
        f.truncate()


if __name__ == "__main__":
    # Load data
    file = open('Dataset/Train/train.json', 'r')
    data = file.read()
    file.close()

    # Load images
    list_images= glob.glob('Dataset/Train/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))

    # We preprocess images
    array_images = get_images(images)

    # Parse data
    array_class_w = parse_w_json(data)
    array_class_v = parse_v_json(data)

    # We check if datasets exist
    create_dataset()

    # Create balanced bbdd for w
    create_balanced_data_w(data, array_class_w, array_images)
    # Create balanced bbdd for v
    create_balanced_data_v(data, array_class_v, array_images)
