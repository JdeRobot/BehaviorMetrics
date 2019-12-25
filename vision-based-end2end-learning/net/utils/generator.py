import os
import glob
import json
import cv2

def create_dataset():
	# We check if dataset exists
	foldername = 'Net/Dataset'
	if not os.path.exists(foldername): os.makedirs(foldername)
	folder_images = foldername + '/' + 'Images'
	if not os.path.exists(folder_images): os.makedirs(folder_images)


def get_classification(w):
	if w == '-':
		classification = 'right'
	else:
		classification = 'left'
	return classification


def get_classification_w(w):
	if w < 0 and abs(w) >= 1.0:
		classification = 'radically_right'
	elif w < 0 and abs(w) >= 0.5:
		classification = 'moderately_right'
	elif w < 0 and abs(w) >= 0.1:
		classification = 'slightly_right'
	elif abs(w) >= 1.0:
		classification = 'radically_left'
	elif abs(w) >= 0.5:
		classification = 'moderately_left'
	elif abs(w) >= 0.1:
		classification = 'slightly_left'
	else:
		classification = 'slight'
	return classification


def get_classification_v(v):
	if v > 11:
		classification = 'very_fast'
	elif v > 9:
		classification = 'fast'
	elif v > 7:
		classification = 'moderate'
	else:
		classification = 'slow'
	return classification


def save_json(v, w):
	# We save the speed data

	file_name = 'Net/Dataset/data.json'

	classification = get_classification(w)
	classification_w = get_classification_w(w)
	classification_v = get_classification_v(v)

	data = {
    	'v': v,
    	'w': w,
		'classification': classification,
		'class2': classification_w,
		'class3': classification_v
	}

	with open(file_name, 'a') as file:
		json.dump(data, file)


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


def save_image(img):
	# We save images
	folder_images = 'Net/Dataset/Images/'
	empty = check_empty_directory(folder_images)
	if empty:
		number = 1
	else:
		number = get_number_image(folder_images)
	name_image = folder_images + str(number) + '.png'
	cv2.imwrite(name_image, img)

