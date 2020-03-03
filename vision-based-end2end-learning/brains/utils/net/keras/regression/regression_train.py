import glob
import numpy as np
import cv2
import os
import matplotlib

matplotlib.use('Agg')

from time import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from net.keras.regression.regression_model import *


def load_data(folder):
    name_folder = '../' + folder + '/Images/'
    list_images = glob.glob(name_folder + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))
    name_file = '../' + folder + '/data.json'
    file = open(name_file, 'r')
    data = file.read()
    file.close()
    return images, data


def parse_json(data, array_v, array_w):
    # We process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        v = d.split('"v": ')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split(('"w": '))[1]
        array_v.append(float(v))
        array_w.append(float(w))

    return array_v, array_w


def get_images(list_images, type_image, array_imgs):
    # We read the images
    for name in list_images:
        img = cv2.imread(name)
        if type_image == 'cropped':
            img = img[220:480, 0:640]
        img = cv2.resize(img, (img.shape[1] / 4, img.shape[0] / 4))
        array_imgs.append(img)

    return array_imgs


def add_extreme_data(array_w, imgs_w, array_v, imgs_v):
    for i in range(0, len(array_w)):
        if abs(array_w[i]) >= 1:
            if abs(array_w[i]) >= 2:
                num_iter = 10
            else:
                num_iter = 5
            for j in range(0, num_iter):
                array_w.append(array_w[i])
                imgs_w.append(imgs_w[i])
        if float(array_v[i]) <= 2:
            for j in range(0, 1):
                array_v.append(array_v[i])
                imgs_v.append(imgs_v[i])
    return array_w, imgs_w, array_v, imgs_v


def add_extreme_data_temporal(array_w, imgs_w, array_v, imgs_v):
    timestep = 5
    for i in range(0, len(array_w)):
        if abs(array_w[i]) >= 2:
            for j in range(0, 2):
                for k in range(timestep, 0, -1):
                    array_w.append(array_w[i - k])
                    imgs_w.append(imgs_w[i - k])
        if float(array_v[i]) <= 2:
            for j in range(0, 5):
                for k in range(timestep, 0, -1):
                    array_v.append(array_v[i - k])
                    imgs_v.append(imgs_v[i - k])
    return array_w, imgs_w, array_v, imgs_v


def preprocess_data(array_w, array_v, imgs):
    # We take the image and just flip it and negate the measurement
    flip_imgs = []
    array_flip_w = []
    for i in range(len(array_w)):
        flip_imgs.append(cv2.flip(imgs[i], 1))
        array_flip_w.append(-array_w[i])
    new_array_w = array_w + array_flip_w
    new_array_v = array_v + array_v
    new_array_imgs = imgs + flip_imgs
    return new_array_w, new_array_v, new_array_imgs


def preprocess_images_lstm(imgs):
    new_array_imgs = []
    timestep = 5
    for i in range(0, len(imgs)):
        if i < timestep:
            array = []
            for j in range(0, (timestep - i)):
                array.append(imgs[0])
            for j in range(0, i):
                array.append(imgs[j])
            new_array_imgs.append(array)
        else:
            new_array_imgs.append(imgs[i - timestep:i])
    return new_array_imgs


def normalize_image(array):
    rng = np.amax(array) - np.amin(array)
    if rng == 0:
        rng = 1
    amin = np.amin(array)
    return (array - amin) * 255.0 / rng


def stack_frames(imgs, type_net):
    new_imgs = []
    margin = 10
    for i in range(0, len(imgs)):
        # if i - 2*(margin+1) < 0:
        #     index1 = 0
        # else:
        #     index1 = i - 2*(margin+1)
        if i - (margin + 1) < 0:
            index2 = 0
        else:
            index2 = i - (margin + 1)
        # im1 =  np.concatenate([imgs[index1], imgs[index2]], axis=2)
        # im2 = np.concatenate([im1, imgs[i]], axis=2)
        if type_net == 'stacked_dif':
            # im = imgs[i] - imgs[index2]
            i1 = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
            i2 = cv2.cvtColor(imgs[index2], cv2.COLOR_BGR2GRAY)
            i1 = cv2.GaussianBlur(i1, (5, 5), 0)
            i2 = cv2.GaussianBlur(i2, (5, 5), 0)
            difference = np.zeros((i1.shape[0], i1.shape[1], 1))
            difference[:, :, 0] = cv2.subtract(np.float64(i1), np.float64(i2))
            mask1 = cv2.inRange(difference[:, :, 0], 15, 255)
            mask2 = cv2.inRange(difference[:, :, 0], -255, -15)
            mask = mask1 + mask2
            difference[:, :, 0][np.where(mask == 0)] = 0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            difference[:, :, 0] = cv2.morphologyEx(difference[:, :, 0], cv2.MORPH_CLOSE, kernel)
            im = difference
            if np.ptp(im) != 0:
                im = 256 * (im - np.min(im)) / np.ptp(im) - 128
            else:
                im = 256 * (im - np.min(im)) / 1 - 128
            im2 = np.concatenate([im, imgs[i]], axis=2)

        elif type_net == 'stacked':
            im2 = np.concatenate([imgs[index2], imgs[i]], axis=2)
        elif type_net == 'temporal':
            # i1 = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2HSV)
            # i2 = cv2.cvtColor(imgs[index2], cv2.COLOR_BGR2HSV)
            # dif = np.zeros((i1.shape[0], i1.shape[1], 2))
            # dif[:,:,0] = cv2.absdiff(i1[:, :, 0], i2[:, :, 0])
            # dif[:,:,1] = cv2.absdiff(i1[:, :, 1], i2[:, :, 1])
            # i1 = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
            # i2 = cv2.cvtColor(imgs[index2], cv2.COLOR_BGR2GRAY)
            # dif = np.zeros((i1.shape[0], i1.shape[1], 1))
            # dif[:,:,0] = cv2.subtract(i1, i2)
            # im2 = np.add(imgs[i], imgs[index2])
            # im2 = normalize_image(dif)


            # i1 = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
            # i2 = cv2.cvtColor(imgs[index2], cv2.COLOR_BGR2GRAY)
            # i1 = cv2.GaussianBlur(i1, (5, 5), 0)
            # i2 = cv2.GaussianBlur(i2, (5, 5), 0)
            # difference = np.zeros((i1.shape[0], i1.shape[1], 1))
            # difference[:, :, 0] = cv2.absdiff(i1, i2)
            # _, difference[:, :, 0] = cv2.threshold(difference[:, :, 0], 15, 255, cv2.THRESH_BINARY)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # difference[:, :, 0] = cv2.morphologyEx(difference[:, :, 0], cv2.MORPH_CLOSE, kernel)
            # im2 = difference


            i1 = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
            i2 = cv2.cvtColor(imgs[index2], cv2.COLOR_BGR2GRAY)
            i1 = cv2.GaussianBlur(i1, (5, 5), 0)
            i2 = cv2.GaussianBlur(i2, (5, 5), 0)
            difference = np.zeros((i1.shape[0], i1.shape[1], 1))
            difference[:, :, 0] = cv2.subtract(np.float64(i1), np.float64(i2))
            mask1 = cv2.inRange(difference[:, :, 0], 15, 255)
            mask2 = cv2.inRange(difference[:, :, 0], -255, -15)
            mask = mask1 + mask2
            difference[:, :, 0][np.where(mask == 0)] = 0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            difference[:, :, 0] = cv2.morphologyEx(difference[:, :, 0], cv2.MORPH_CLOSE, kernel)
            im2 = difference
            if np.ptp(im2) != 0:
                im2 = 256 * (im2 - np.min(im2)) / np.ptp(im2) - 128
            else:
                im2 = 256 * (im2 - np.min(im2)) / 1 - 128

        new_imgs.append(im2)
    return new_imgs


def choose_model(type_net, img_shape, type_image):
    model_png = 'models/model_' + type_net + '.png'
    if type_image == 'cropped':
        model_file_v = 'models/model_' + type_net + '_' + type_image + '_v.h5'
        model_file_w = 'models/model_' + type_net + '_' + type_image + '_w.h5'
    else:
        model_file_v = 'models/model_' + type_net + '_v.h5'
        model_file_w = 'models/model_' + type_net + '_w.h5'
    if type_net == 'pilotnet':
        model_v = pilotnet_model(img_shape)
        model_w = pilotnet_model(img_shape)
        batch_size_v = 64  # 16
        batch_size_w = 64
        nb_epoch_v = 300  # 223
        nb_epoch_w = 300  # 212
    elif type_net == 'tinypilotnet':
        model_v = tinypilotnet_model(img_shape)
        model_w = tinypilotnet_model(img_shape)
        batch_size_v = 64  # 16
        batch_size_w = 64
        nb_epoch_v = 250  # 223
        nb_epoch_w = 1000  # 212
    elif type_net == 'stacked':
        model_v = pilotnet_model(img_shape)
        model_w = pilotnet_model(img_shape)
        batch_size_v = 64
        batch_size_w = 64
        nb_epoch_v = 150
        nb_epoch_w = 150
    elif type_net == 'stacked_dif':
        model_v = temporal_model(img_shape)
        model_w = temporal_model(img_shape)
        batch_size_v = 64
        batch_size_w = 64
        nb_epoch_v = 100
        nb_epoch_w = 100
    elif type_net == 'temporal':
        model_v = temporal_model(img_shape)
        model_w = temporal_model(img_shape)
        batch_size_v = 64
        batch_size_w = 64
        nb_epoch_v = 300
        nb_epoch_w = 300
    elif type_net == 'lstm_tinypilotnet':
        model_v = lstm_tinypilotnet_model(img_shape, type_image)
        model_w = lstm_tinypilotnet_model(img_shape, type_image)
        batch_size_v = 12  # 8
        batch_size_w = 12  # 8
        nb_epoch_v = 200  # 223
        nb_epoch_w = 200  # 212
    elif type_net == 'deepestlstm_tinypilotnet':
        model_v = deepestlstm_tinypilotnet_model(img_shape, type_image)
        model_w = deepestlstm_tinypilotnet_model(img_shape, type_image)
        batch_size_v = 12  # 8
        batch_size_w = 12  # 8
        nb_epoch_v = 150  # 223
        nb_epoch_w = 150  # 212
    elif type_net == 'lstm':
        model_v = lstm_model(img_shape)
        model_w = lstm_model(img_shape)
        batch_size_v = 128  # 8
        batch_size_w = 128  # 8
        nb_epoch_v = 100  # 223
        nb_epoch_w = 100  # 212
    elif type_net == 'controlnet':
        model_v = controlnet_model(img_shape)
        model_w = controlnet_model(img_shape)
        batch_size_v = 128  # 24 #64
        batch_size_w = 128  # 24 #64
        nb_epoch_v = 200  # 300
        nb_epoch_w = 200  # 300
    return model_v, model_w, model_file_v, model_file_w, model_png, batch_size_v, nb_epoch_v, batch_size_w, nb_epoch_w

def train(params):
    
    type_image = params[0]
    type_net = params[1]

    # Load data
    images, data = load_data('Dataset')
    images_curve, data_curve = load_data('Dataset_Curves')

    # We preprocess images
    array_imgs = []
    x = get_images(images, type_image, array_imgs)
    x = get_images(images_curve, type_image, x)
    # We preprocess json
    array_v = []
    array_w = []
    y_v, y_w = parse_json(data, array_v, array_w)
    y_v, y_w = parse_json(data_curve, y_v, y_w)

    # Split data into 80% for train and 20% for validation
    if type_net == 'pilotnet' or type_net == 'tinypilotnet':
        # We adapt the data
        y_w, y_v, x = preprocess_data(y_w, y_v, x)
        x_w = x[:]
        x_v = x[:]
        y_w, x_w, y_v, x_v = add_extreme_data(y_w, x_w, y_v, x_v)
        X_train_v, X_validation_v, y_train_v, y_validation_v = train_test_split(x_v, y_v, test_size=0.20,
                                                                                random_state=42)
        X_train_w, X_validation_w, y_train_w, y_validation_w = train_test_split(x_w, y_w, test_size=0.20,
                                                                                random_state=42)
        # X_train_v, X_validation_v, y_train_v, y_validation_v = train_test_split(x_v,y_v,test_size=0.20,random_state=42)
        # X_train_w, X_validation_w, y_train_w, y_validation_w = train_test_split(x_w,y_w,test_size=0.20,random_state=42)
    elif type_net == 'stacked' or type_net == 'stacked_dif' or type_net == 'temporal':
        # We stack frames
        y_w, y_v, x = preprocess_data(y_w, y_v, x)
        x = stack_frames(x, type_net)
        x_w = x[:]
        x_v = x[:]
        y_w, x_w, y_v, x_v = add_extreme_data(y_w, x_w, y_v, x_v)
        X_train_v, X_validation_v, y_train_v, y_validation_v = train_test_split(x_v, y_v, test_size=0.20,
                                                                                random_state=42)
        X_train_w, X_validation_w, y_train_w, y_validation_w = train_test_split(x_w, y_w, test_size=0.20,
                                                                                random_state=42)
    elif type_net == 'lstm_tinypilotnet' or type_net == 'lstm' or type_net == 'deepestlstm_tinypilotnet' or \
                    type_net == 'controlnet':
        y_w, y_v, x = preprocess_data(y_w, y_v, x)
        if type_net == 'controlnet' or type_net == 'lstm':
            x = preprocess_images_lstm(x[:])
        x_w = x[:]
        x_v = x[:]
        y_w, x_w, y_v, x_v = add_extreme_data(y_w, x_w, y_v, x_v)  # add_extreme_data_temporal(y_w, x_w, y_v, x_v)
        X_train_v = x_v
        X_train_w = x_w
        y_train_v = y_v
        y_train_w = y_w
        X_t_v, X_validation_v, y_t_v, y_validation_v = train_test_split(x_v, y_v, test_size=0.20, random_state=42)
        X_t_w, X_validation_w, y_t_w, y_validation_w = train_test_split(x_w, y_w, test_size=0.20, random_state=42)

    # Variables
    if type_net == 'stacked':
        if type_image == 'cropped':
            # img_shape = (65, 160, 9)
            img_shape = (65, 160, 6)
        else:
            # img_shape = (120, 160, 9)
            img_shape = (120, 160, 6)
    if type_net == 'stacked_dif':
        if type_image == 'cropped':
            img_shape = (65, 160, 4)
        else:
            img_shape = (120, 160, 4)
    elif type_net == 'temporal':
        if type_image == 'cropped':
            # img_shape = (65, 160, 2)
            img_shape = (65, 160, 1)
        else:
            # img_shape = (120, 160, 2)
            img_shape = (120, 160, 1)
    elif type_net == 'controlnet' or type_net == 'lstm':
        # img_shape = (120, 160, 3)
        # img_shape = (250, 5, 57600)
        img_shape = (5, 120, 160, 3)
    else:
        if type_image == 'cropped':
            img_shape = (65, 160, 3)
        else:
            img_shape = (120, 160, 3)

    # We adapt the data
    X_train_v = np.stack(X_train_v, axis=0)
    y_train_v = np.stack(y_train_v, axis=0)
    X_validation_v = np.stack(X_validation_v, axis=0)
    y_validation_v = np.stack(y_validation_v, axis=0)

    X_train_w = np.stack(X_train_w, axis=0)
    y_train_w = np.stack(y_train_w, axis=0)
    X_validation_w = np.stack(X_validation_w, axis=0)
    y_validation_w = np.stack(y_validation_w, axis=0)

    # Get model
    model_v, model_w, model_file_v, model_file_w, model_png, batch_size_v, nb_epoch_v, batch_size_w, \
    nb_epoch_w = choose_model(type_net, img_shape, type_image)

    # Print layers
    print(model_v.summary())
    # Plot layers of model
    plot_model(model_v, to_file=model_png)

    #  We train
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    if not os.path.exists('csv'): os.makedirs('csv')
    filename = 'csv/' + type_net + '_' + type_image + '_v.csv'
    csv_logger = CSVLogger(filename=filename, separator=',', append=True)

    model_checkpoint = ModelCheckpoint(model_file_v,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       monitor='val_loss',
                                       verbose=1)

    model_history_v = model_v.fit(X_train_v, y_train_v, epochs=nb_epoch_v, batch_size=batch_size_v, verbose=2,
                                  validation_data=(X_validation_v, y_validation_v),
                                  callbacks=[tensorboard, model_checkpoint, csv_logger])

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    filename = 'csv/' + type_net + '_' + type_image + '_w.csv'
    csv_logger = CSVLogger(filename=filename, separator=',', append=True)

    model_checkpoint = ModelCheckpoint(model_file_w,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       monitor='val_loss',
                                       verbose=1)

    model_history_w = model_w.fit(X_train_w, y_train_w, epochs=nb_epoch_w, batch_size=batch_size_w, verbose=2,
                                  validation_data=(X_validation_w, y_validation_w),
                                  callbacks=[tensorboard, model_checkpoint, csv_logger])

    # We evaluate the model
    score = model_v.evaluate(X_validation_v, y_validation_v, verbose=0)
    print('Evaluating v')
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    print('Test mean squared error: ', score[2])
    print('Test mean absolute error: ', score[3])

    score = model_w.evaluate(X_validation_w, y_validation_w, verbose=0)
    print('Evaluating w')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test mean squared error: ', score[2])
    print('Test mean absolute error: ', score[3])

    # We save the model
    # model_v.save(model_file_v)
    # model_w.save(model_file_w)

    # Plot the training and validation loss for each epoch
    # plt.plot(model_history.history['loss'])
    # plt.plot(model_history.history['val_loss'])
    # plt.title('mse')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.ylim([0, 0.1])
    # plt.show()
    #
    # # Accuracy Curves
    # plt.figure(figsize=[8, 6])
    # plt.plot(model_history.history['acc'], 'r', linewidth=3.0)
    # plt.plot(model_history.history['val_acc'], 'b', linewidth=3.0)
    # plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    # plt.xlabel('Epochs ', fontsize=16)
    # plt.ylabel('Accuracy', fontsize=16)
    # plt.title('Accuracy Curves', fontsize=16)
    # plt.show()

if __name__ == "__main__":
    # Choose options
    type_image = raw_input('Choose the type of image you want: normal or cropped: ')
    type_net = raw_input('Choose the type of network you want: pilotnet, tinypilotnet, lstm_tinypilotnet, lstm, '
                         'deepestlstm_tinypilotnet, controlnet, stacked, stacked_dif or temporal: ')
    print('Your choice: ' + type_net + ', ' + type_image)

    train(params=[type_image, type_net])
