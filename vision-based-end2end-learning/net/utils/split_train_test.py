import os
import glob
import cv2
import json
import yaml

from sklearn.model_selection import train_test_split


def create_folders():
    foldername = 'Dataset/Train'
    if not os.path.exists(foldername): os.makedirs(foldername)
    foldername = 'Dataset/Test'
    if not os.path.exists(foldername): os.makedirs(foldername)
    foldername = 'Dataset/Train/Images'
    if not os.path.exists(foldername): os.makedirs(foldername)
    foldername = 'Dataset/Test/Images'
    if not os.path.exists(foldername): os.makedirs(foldername)


def parse_json(data):
    array = []
    # We process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        data = d + '}'
        array.append(data)

    return array


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        array_imgs.append(img)

    return array_imgs


def save_json(data, file_name):
    # We save the data
    with open(file_name, 'a') as file:
        json.dump(data, file)


def save_data(array, file_name):
    for data in array:
        data_dict = yaml.load(data)
        save_json(data_dict, file_name)


def save_image(list_img, folder_images):
    for i in range(0, len(list_img)):
        # We save images
        name_image = folder_images + str(i) + '.png'
        cv2.imwrite(name_image, list_img[i])


if __name__ == "__main__":
    # Load data
    list_images = glob.glob('Dataset/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[2].split('.png')[0]))

    file = open('Dataset/data.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    array_imgs = get_images(images)
    # We preprocess json
    array_data = parse_json(data)

    # Create train and test folders
    create_folders()

    # Split data into 70% for train and 30% for test
    X_train, X_test, y_train, y_test = train_test_split(array_imgs, array_data, test_size=0.30, random_state=42)

    # Save train and test data
    save_image(X_train, 'Dataset/Train/Images/')
    save_data(y_train, 'Dataset/Train/train.json')
    save_image(X_test, 'Dataset/Test/Images/')
    save_data(y_test, 'Dataset/Test/test.json')
