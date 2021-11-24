from gradcam.gradcam import GradCAM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="path to the input image")
parser.add_argument("-m", "--model", required=True, type=str, help="model to be used")
parser.add_argument('-is', '--image_shape', nargs='+', help='<Required> Set flag', required=True)


args = vars(parser.parse_args())
image_shape = (int(args["image_shape"][0]), int(args["image_shape"][1]))
model = load_model(args["model"])

# load the original image from disk (in OpenCV format) and then
# resize the image to its target dimensions
orig = cv2.imread(args["image"])
orig = orig[240:480, 0:640]
orig = cv2.resize(orig, image_shape)
# load the input image from disk (in Keras/TensorFlow format) and
# preprocess it
image = cv2.imread(args["image"])
image = image[240:480, 0:640]
img = cv2.resize(image, image_shape)
image = np.expand_dims(img, axis=0)

# use the network to make predictions on the input image and find
# the class label index with the largest corresponding probability
preds = model.predict(image)

i = np.argmax(preds[0])

# initialize our gradient class activation map and build the heatmap
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)
# resize the resulting heatmap to the original input image dimensions
# and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

# save the original image and resulting heatmap and output image
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
result = cv2.imwrite(r'image.png', output)

if result:
    print("File saved successfully")
else:
    print("Error in saving file")

