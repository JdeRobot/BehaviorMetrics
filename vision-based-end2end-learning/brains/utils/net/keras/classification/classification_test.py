import glob
import numpy as np
import cv2
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

from keras.models import load_model
from keras.utils import np_utils
from sklearn import metrics


def parse_json_2_classes_w(data):
    array_class = []
    # We process json
    data_parse = data.split('"classification": ')[1:]
    for d in data_parse:
        classification = d.split(', "class2":')[0]
        array_class.append(classification)

    return array_class


def parse_json_7_classes_w(data):
    array_class = []
    # We process json
    data_parse = data.split('"class2": ')[1:]
    for d in data_parse:
        classification = d.split(', "class3":')[0]
        array_class.append(classification)

    return array_class


def parse_json_9_classes_w(data):
    array_class = []
    # We process json
    data_parse = data.split('"class_w_9": ')[1:]
    for d in data_parse:
        classification = d.split(', "classification":')[0]
        array_class.append(classification)

    return array_class


def parse_json_other_classes_v(data):
    array_class = []
    # We process json
    data_parse = data.split('"class3": ')[1:]
    for d in data_parse:
        classification = d.split(', "w":')[0]
        array_class.append(classification)

    return array_class


def parse_json(data, num_classes, name_variable):
    if num_classes == 2 and name_variable == 'w':
        array_class = parse_json_2_classes_w(data)
    elif num_classes == 7 and name_variable == 'w':
        array_class = parse_json_7_classes_w(data)
    elif num_classes == 9 and name_variable == 'w':
        array_class = parse_json_9_classes_w(data)
    elif name_variable == 'v':
        array_class = parse_json_other_classes_v(data)
    return array_class



def get_images(list_images, type_image):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        if type_image == 'cropped':
            img = img[240:480, 0:640]
        img = cv2.resize(img, (img.shape[1] / 4, img.shape[0] / 4))
        array_imgs.append(img)

    return array_imgs


def adapt_label_7_w(label):
    if label == '"radically_left"' or label == 'radically_left':
        label = 0
    elif label == '"moderately_left"' or label == 'moderately_left':
        label = 1
    elif label == '"slightly_left"' or label == 'slightly_left':
        label = 2
    elif label == '"slight"' or label == 'slight':
        label = 3
    elif label == '"slightly_right"' or label == 'slightly_right':
        label = 4
    elif label == '"moderately_right"' or label == 'moderately_right':
        label = 5
    elif label == '"radically_right"' or label == 'radically_right':
        label = 6
    return label


def adapt_label_9_w(label):
    if label == '"radically_left"' or label == 'radically_left':
        label = 0
    elif label == '"strongly_left"' or label == 'strongly_left':
        label = 1
    elif label == '"moderately_left"' or label == 'moderately_left':
        label = 2
    elif label == '"slightly_left"' or label == 'slightly_left':
        label = 3
    elif label == '"slight"' or label == 'slight':
        label = 4
    elif label == '"slightly_right"' or label == 'slightly_right':
        label = 5
    elif label == '"moderately_right"' or label == 'moderately_right':
        label = 6
    elif label == '"strongly_right"' or label == 'strongly_right':
        label = 7
    elif label == '"radically_right"' or label == 'radically_right':
        label = 8
    return label


def adapt_label_4_v(label):
    if label == '"slow"' or label == 'slow':
        label = 0
    elif label == '"moderate"' or label == 'moderate':
        label = 1
    elif label == '"fast"' or label == 'fast':
        label = 2
    elif label == '"very_fast"' or label == 'very_fast':
        label = 3
    return label


def adapt_labels(array_labels, num_classes, name_variable):
    for i in range(0, len(array_labels)):
        if name_variable == 'w':
            if num_classes == 2:
                if array_labels[i] == '"left"':
                    array_labels[i] = 0
                else:
                    array_labels[i] = 1
            elif num_classes == 7:
                array_labels[i] = adapt_label_7_w(array_labels[i])
            elif num_classes == 9:
                array_labels[i] = adapt_label_9_w(array_labels[i])

        elif name_variable == 'v':
            array_labels[i] = adapt_label_4_v(array_labels[i])

    return array_labels


def choose_model(name, num_classes, name_var, type_net, type_image):
    if name == "lenet":
        model_file = 'models/model_lenet5_7classes.h5'
    elif name == "smaller_vgg":
        if type_image == 'cropped':
            model_file = 'models/model_smaller_vgg_' + str(num_classes) + 'classes_' + type_net + '_' + \
                         type_image + '_' + name_var + '.h5'
        else:
            model_file = 'models/model_smaller_vgg_' + str(num_classes) + 'classes_' + type_net + '_'+ name_var + '.h5'
    elif name == "other":
        model_file = 'models/model_binary_classification.h5'
    return model_file



def make_predictions(data):
    """
    Function to make the predictions over a data set
    :param data: np.array - Images to predict
    :return: np.array - Labels of predictions
    """
    predictions = model.predict(data)
    predicted = [np.argmax(prediction) for prediction in predictions]

    return np.array(predicted)


def top_k_accuracy(labels, y_predict, k):
    top_k = 0
    for i in range(0, len(labels)):
        if (labels[i] -(k-1)) <= y_predict[i] and y_predict[i] <= (labels[i] + (k-1)):
            top_k += 1
    top_k = top_k * 100 / len(labels)
    return top_k


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

    name_variable = params[0]
    num_classes = params[1]
    name_model = params[2]
    type_net = params[3]
    type_image = params[4]

    # Load data
    list_images = glob.glob('../Dataset/Test/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    file = open('../Dataset/Test/test.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    x_test = get_images(images, type_image)
    # We preprocess json
    y_test = parse_json(data, num_classes, name_variable)

    # We adapt string labels to int labels
    y_test = adapt_labels(y_test, num_classes, name_variable)
    labels = y_test

    # Convert the labels from integers to vectors
    y_test = np_utils.to_categorical(y_test, num_classes)

    # We adapt the data
    X_test = np.stack(x_test, axis=0)
    y_test = np.stack(y_test, axis=0)

    # Get model
    model_file = choose_model(name_model, num_classes, name_variable, type_net, type_image)

    # Load model
    print('Loading model...')
    model = load_model(model_file)

    # Make predictions
    print('Making predictions...')
    y_predict = make_predictions(X_test)

    # Evaluation
    print('Making evaluation...')
    score = model.evaluate(X_test, y_test)

    evaluation = metrics.classification_report(labels, y_predict)

    # Test loss and accuracy
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # We calculate top 2 accuracy
    top_2_accuracy = top_k_accuracy(labels, y_predict, 2)
    print('Top 2 accuracy: ' + str(top_2_accuracy) +'%')

    # Precision, recall, F1 score for each class
    print("Evaluation's metrics: ")
    print(evaluation)

    # Confusion matrix
    conf_matrix = metrics.confusion_matrix(labels, y_predict)
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)

if __name__ == "__main__":

    # Choose options
    num_classes = int(input('Choose the number of classes: '))
    name_variable = raw_input('Choose the variable you want to train: v or w: ')
    type_image = raw_input('Choose the type of image you want: normal or cropped: ')
    type_net = raw_input('Choose the type of network you want: normal, biased or balanced: ')
    name_model = raw_input('Choose the model you want to use: mobilenet, lenet, smaller_vgg or other: ')
    print('Your choice: ' + str(num_classes) + ', ' + name_variable + ', ' + type_net + ' and ' + name_model)

    test(params=[num_classes, name_variable, type_image, type_net, name_model])
    