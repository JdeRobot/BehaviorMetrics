import glob
import os
import cv2

DIR_carla_dataset_name_images = '../../carla_dataset_28_09/'
carla_dataset_images = glob.glob(DIR_carla_dataset_name_images + '*')
array_imgs = []
#print(carla_dataset_images)

previous_image = 0

for iterator, filename in enumerate(carla_dataset_images):
    try:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #print((previous_image==img).all())
        if (previous_image==img).all() == True:
            print(iterator)
        previous_image = img
        array_imgs.append(img)
    except:
        print('error')

