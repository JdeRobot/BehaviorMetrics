#  Based on: https://arxiv.org/pdf/1604.07316.pdf
#
#  Authors :
#       Vanessa Fernandez Martinez <vanessa_1895@msn.com>

from keras.models import Sequential
from keras.layers import Flatten,Dense,Conv2D,BatchNormalization,Dropout,ConvLSTM2D,Reshape,Activation,MaxPooling2D
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam


def pilotnet_model(img_shape):
    '''
    Model of End to End Learning for Self-Driving Cars (NVIDIA)
    '''
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


def tinypilotnet_model(img_shape):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


def lstm_tinypilotnet_model(img_shape, type_image):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=(2, 2), input_shape=img_shape, activation="relu"))
    model.add(Conv2D(16, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), activation="relu"))
    if type_image == 'cropped':
        model.add(Reshape((1, 7, 19, 32)))
    else:
        model.add(Reshape((1, 14, 19, 32)))
    model.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3, border_mode='same', return_sequences=True))
    if type_image == 'cropped':
        model.add(Reshape((7, 19, 40)))
    else:
        model.add(Reshape((14, 19, 40)))
    model.add(Conv2D(1, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Flatten())
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


def deepestlstm_tinypilotnet_model(img_shape, type_image):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=(2, 2), input_shape=img_shape, activation="relu"))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.2))
    if type_image == 'cropped':
        model.add(Reshape((1, 7, 19, 8)))
    else:
        model.add(Reshape((1, 14, 19, 8)))
    model.add(ConvLSTM2D(nb_filter=16, nb_row=3, nb_col=3, border_mode='same', return_sequences=True))
    model.add(ConvLSTM2D(nb_filter=16, nb_row=3, nb_col=3, border_mode='same', return_sequences=True))
    model.add(ConvLSTM2D(nb_filter=12, nb_row=3, nb_col=3, border_mode='same', return_sequences=True))
    if type_image == 'cropped':
        model.add(Reshape((7, 19, 12)))
    else:
        model.add(Reshape((14, 19, 12)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


def temporal_model(img_shape):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    model.add(Conv2D(12, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(24, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(150, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


def lstm_model(img_shape):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Reshape((9600, 1)))
    model.add(LSTM(12))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    adam = Adam(lr=0.00001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


def controlnet_model(img_shape):
    model = Sequential()
    #model.add(Conv2D(16, (5, 5), input_shape=img_shape, activation="relu"))
    #model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    #model.add(Conv2D(16, (5, 5), activation="relu"))
    #model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    #model.add(Conv2D(16, (3, 3), activation="relu"))
    #model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    #model.add(Conv2D(16, (3, 3), activation="relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(16, (3, 3), activation="relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dense(50, activation="relu"))
    #model.add(Dense(50, activation="relu"))
    #model.add(Flatten())
    #model.add(Reshape((100, 1)))
    #model.add(LSTM(5))
    #model.add(Activation('softmax'))
    #model.add(Dense(1))

    model.add(TimeDistributed(Conv2D(16, (5, 5), activation="relu"), input_shape=img_shape))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2))))
    model.add(TimeDistributed(Conv2D(16, (5, 5), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2))))
    model.add(TimeDistributed(Conv2D(16, (3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2))))
    model.add(TimeDistributed(Conv2D(16, (3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(strides=(2, 2))))
    model.add(TimeDistributed(Conv2D(16, (3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(strides=(2, 2))))
    model.add(TimeDistributed(Dense(50, activation="relu")))
    model.add(TimeDistributed(Dense(50, activation="relu")))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100, return_sequences=False))
    model.add(Activation('softmax'))
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model