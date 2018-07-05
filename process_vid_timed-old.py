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
import time

NET_INPUT_NAME = "net_input:0"
LOGITS_NAME = "logits:0"

# FC-DenseNet-56
# AVG FPS below 1.0
# --------------------------------------
# 256*256
# 1.071 FPS
# MODEL_DIR = "./saved_models/2018-05-17-2057"

# 128*128
# MODEL_DIR = "./saved_models/2018-05-18-0215"

# 64*64
# MODEL_DIR = "./saved_models/2018-05-18-1306"

# --------------------------------------

# MobileUNet-Skip
LOGITS_NAME = "logits_1:0"
# --------------------------------------
# 256*256
# 1.304 FPS
MODEL_DIR = "./saved_models/2018-05-19-1614"
# 32*32
# 1.153 FPS
# MODEL_DIR = "./saved_models/2018-05-18-1401"
# --------------------------------------

# ENet
# --------------------------------------
# 600*800
# FPS under 1
# MODEL_DIR = "./saved_models/2018-05-19-1412"
# 256*256
# FPS under 1
# MODEL_DIR = "./saved_models/2018-05-19-1403"
# 300*400
# skip_connections = False
# FPS under 1
# MODEL_DIR = "./saved_models/2018-05-19-1501"

# --------------------------------------

RESULTS_DIR = "saved_results"
IMG_SIZE = (256, 256)

DEFAULT_VIDEO = "../Example/test_video.mp4"

file = sys.argv[-1]

if file == os.path.basename(__file__):
  print("Load default video", DEFAULT_VIDEO)
  print("To load input video run `python demo.py video_path.mp4`")
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

start = time.time()

sess = tf.Session()
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

meta_graph_def = tf.saved_model.loader.load(sess,
                                            [tf.saved_model.tag_constants.SERVING], MODEL_DIR)

image_input = sess.graph.get_tensor_by_name(NET_INPUT_NAME)
logits = sess.graph.get_tensor_by_name(LOGITS_NAME)

end = time.time()
total_time_start = end - start

total_time_resize_down = 0
total_time_run = 0
total_time_resize_up = 0
total_time_encode = 0

for rgb_frame in video:

    start = time.time()
    image = cv2.resize(rgb_frame, (IMG_SIZE[1], IMG_SIZE[0]))
    end = time.time()
    total_time_resize_down += (end-start)
    
    start = time.time()
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {image_input: [image]})
    end = time.time()
    total_time_run += (end-start)

    start = time.time()

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

    end = time.time()
    total_time_resize_up += (end-start)

    start = time.time()
    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    end = time.time()
    total_time_encode += (end-start)

    # Increment frame
    frame+=1

# print (json.dumps(answer_key))

total_time = total_time_start + total_time_resize_down + total_time_run + \
             total_time_resize_up + total_time_encode

print("total_time_start: {:.3f} ({:.1%})".format(
    total_time_start, (total_time_start/total_time)))

print("total_time_resize_down: {:.3f} ({:.1%})".format(
    total_time_resize_down, (total_time_resize_down/total_time)))

print("total_time_run: {:.3f} ({:.1%})".format(
    total_time_run, (total_time_run/total_time)))

print("total_time_resize_up: {:.3f} ({:.1%})".format(
    total_time_resize_up, (total_time_resize_up/total_time)))

print("total_time_encode: {:.3f} ({:.1%})".format(
    total_time_encode, (total_time_encode/total_time)))

print("FPS:", frame / total_time)
