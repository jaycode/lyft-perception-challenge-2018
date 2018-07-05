"""
The first submission
FC DenseNet 56
1.111 FPS
Car F score: 0.686 | Car Precision: 0.712 | Car Recall: 0.680 | Road F score: 0.977 | 
Road Precision: 0.977 | Road Recall: 0.979 | Averaged F score: 0.832

"""
import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

import tensorflow as tf
import os.path
import cv2
import scipy.misc
import numpy as np
from helper2 import reshape_to_ori
import matplotlib.pyplot as plt
import datetime
import errno

MODEL_DIR = "./2018-05-17-2057"
IMG_SIZE = (256, 256)

DEFAULT_VIDEO = "../Example/test_video.mp4"

NET_INPUT_NAME = "net_input:0"
LOGITS_NAME = "logits:0"

file = sys.argv[-1]

if file == os.path.basename(__file__):
  file = DEFAULT_VIDEO

# Define encoder function
def encode(array):
    # print("array to encode:",array.shape)
    pil_img = Image.fromarray(array)
    if pil_img.mode != 'I':
        pil_img = pil_img.convert('I')
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

sess = tf.Session()
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

meta_graph_def = tf.saved_model.loader.load(sess,
                                            [tf.saved_model.tag_constants.SERVING], MODEL_DIR)

image_input = sess.graph.get_tensor_by_name(NET_INPUT_NAME)
logits = sess.graph.get_tensor_by_name(LOGITS_NAME)

for rgb_frame in video:

    image = cv2.resize(rgb_frame, (IMG_SIZE[1], IMG_SIZE[0]))

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {image_input: [image]})

    # Cars
    im_softmax_vehicle = im_softmax[0][:, 2].reshape(IMG_SIZE[0], IMG_SIZE[1])
    binary_car_result = \
      (im_softmax_vehicle >= 0.50)
    # print("rgb_frame:", rgb_frame.shape)
    binary_car_result = reshape_to_ori(binary_car_result, rgb_frame.shape)

    # Road
    im_softmax_road = im_softmax[0][:, 1].reshape(IMG_SIZE[0], IMG_SIZE[1])
    binary_road_result = \
      (im_softmax_road >= 0.50)
    binary_road_result = reshape_to_ori(binary_road_result, rgb_frame.shape)

    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]

    # Increment frame
    frame+=1

print (json.dumps(answer_key))
