import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.models import load_model


def parse_json(data):
    array_v = []
    array_w = []
    # We process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        v = d.split('"v": ')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split(('"w": '))[1]
        array_v.append(float(v))
        array_w.append(float(w))

    return array_v, array_w


def get_images(list_images, type_image, type_net):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        if type_image == 'cropped':
            img = img[220:480, 0:640]
        if type_net == 'lstm':
            img = cv2.resize(img, (img.shape[1] / 10, img.shape[0] / 10))
        else:
            img = cv2.resize(img, (img.shape[1] / 4, img.shape[0] / 4))
        array_imgs.append(img)

    return array_imgs


def stack_frames(imgs):
    new_imgs = []
    margin = 2
    for i in range(0, len(imgs)):
        if i - 2*(margin+1) < 0:
            index1 = 0
        else:
            index1 = i - 2*(margin+1)
        if i - (margin + 1) < 0:
            index2 = 0
        else:
            index2 = i - (margin + 1)
        im1 =  np.concatenate([imgs[index1], imgs[index2]], axis=2)
        im2 = np.concatenate([im1, imgs[i]], axis=2)
        new_imgs.append(im2)
    return new_imgs


def choose_model(type_net, type_image):
    if type_image == 'cropped':
        model_file_v = 'models/model_' + type_net + '_' + type_image + '_v.h5'
        model_file_w = 'models/model_' + type_net + '_' + type_image + '_w.h5'
    else:
        model_file_v = 'models/model_' + type_net + '_v.h5'
        model_file_w = 'models/model_' + type_net + '_w.h5'
    return model_file_v, model_file_w


def make_predictions(data, model):
    """
    Function to make the predictions over a data set
    :param data: np.array - Images to predict
    :return: np.array - Labels of predictions
    """
    predictions = model.predict(data)
    predicted = [float(prediction[0]) for prediction in predictions]

    return np.array(predicted)


def plot_confusion_matrix(cm, cmap=plt.cm.Blues):
    """
    Function to plot the confusion matrix
    :param cm: np.array - Confusion matrix to plot
    :param cmap: plt.cm - Color map
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

def test(params):

    type_image = params[0]
    type_net = params[1]

    # Load data
    list_images = glob.glob('../Dataset/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))

    file = open('../Dataset/data.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    x_test = get_images(images, type_image, type_net)
    # We preprocess json
    y_test_v, y_test_w = parse_json(data)

    # We adapt stacked frames
    if type_net == 'stacked':
        x_test = stack_frames(x_test)

    # We adapt the data
    X_test = np.stack(x_test, axis=0)
    y_test_v = np.stack(y_test_v, axis=0)
    y_test_w = np.stack(y_test_w, axis=0)

    # Get model
    model_file_v, model_file_w = choose_model(type_net, type_image)

    # Load model
    print('Loading model...')
    model_v = load_model(model_file_v)
    model_w = load_model(model_file_w)

    # Make predictions
    print('Making predictions...')
    y_predict_v = make_predictions(X_test, model_v)
    y_predict_w = make_predictions(X_test, model_w)

    # We create the figure and subplots
    fig, ax = plt.subplots()
    ax.set_title('Difference between ground truth and prediction')
    ax.axis([-3, 3, -1, 1.5])
    ax.set_xlabel('Angles of ground truth')
    ax.set_ylabel('Difference between ground truth and prediction')
    fig, ax1 = plt.subplots()
    ax1.set_title('Difference between ground truth and prediction')
    ax1.axis([-2, 14, -5, 5])
    ax1.set_xlabel('Speed of ground truth')
    ax1.set_ylabel('Difference between ground truth and prediction')
    for i in range(0, len(y_predict_w)):
        print('w: ', y_test_w[i], y_predict_w[i])
        dif_w = y_test_w[i] - y_predict_w[i]
        dif_v = y_test_v[i] - y_predict_v[i]
        print('v: ', y_test_v[i], y_predict_v[i])
        ax.plot(y_test_w[i], dif_w, 'ro')
        ax1.plot(y_test_v[i], dif_v, 'bo')
    plt.show()

    # Evaluation
    print('Making evaluation...')
    score_v = model_v.evaluate(X_test, y_test_v)
    score_w = model_w.evaluate(X_test, y_test_w)

    # Test loss, accuracy, mse and mae
    print('Evaluation v:')
    print('Test loss:', score_v[0])
    print('Test accuracy:', score_v[1])
    print('Test mean squared error: ', score_v[2])
    print('Test mean absolute error: ', score_v[3])

    print('Evaluation w:')
    print('Test loss:', score_w[0])
    print('Test accuracy:', score_w[1])
    print('Test mean squared error: ', score_w[2])
    print('Test mean absolute error: ', score_w[3])


if __name__ == "__main__":
    # Choose options
    type_image = raw_input('Choose the type of image you want: normal or cropped: ')
    type_net = raw_input('Choose the type of network you want: pilotnet, tinypilotnet, lstm_tinypilotnet, lstm, '
                         'deepestlstm_tinypilotnet or stacked: ')
    print('Your choice: ' + type_net + ', ' +type_image)

    test(params=[type_image, type_net])