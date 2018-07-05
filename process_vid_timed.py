import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

import tensorflow as tf
import os.path
import cv2
import scipy.misc
import numpy as np
from helper2 import reshape_to_ori, softmax, reverse_one_hot
import matplotlib.pyplot as plt
import datetime
import errno
import time
import pandas as pd

NET_INPUT_NAME = "net_input:0"
LOGITS_NAME = "logits:0"

# FC-DenseNet-56
# AVG FPS below 1.0
# --------------------------------------
# 256*256
# 1.111 FPS
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
# 1.304 FPS / 1.875 FPS without softmax
MODEL_DIR = "./saved_models/2018-05-21-1306"
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

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=config)

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

    output_image = sess.run(
        logits, feed_dict={image_input: [image]})

    # im_softmax = sess.run(
    #     tf.nn.softmax(logits),
    #     {image_input: [image]})

    # logits_softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), 0)
    # im_softmax = sess.run(
    #     logits_softmax, feed_dict={image_input: [image]})

    end = time.time()
    total_time_run += (end-start)


    # im_softmax = np.array(im_softmax[0,:,:,:])
    # im_softmax = reverse_one_hot(im_softmax)

    # print("shape:")
    # print(im_softmax.shape)

    # data = pd.DataFrame(im_softmax)
    # print("summary:")
    # print(data.describe())

    # print("test softmax correctness:")
    # print(np.sum(im_softmax, axis=0))

    start = time.time()
    output_image = np.array(output_image[0,:,:,:])
    
    print("output_image shape:", output_image.shape)

    im_softmax_road = output_image[:, :, 1]
    im_softmax_vehicle = output_image[:, :, 2]

    seg_road = \
      ((im_softmax_road >= 0.50) & (im_softmax_road >= im_softmax_vehicle))
        
    seg_vehicle = \
      ((im_softmax_vehicle >= 0.50) & (im_softmax_vehicle >= im_softmax_road))

    # kernel = np.ones((3,3),np.uint8)
    # seg_vehicle =  np.array(seg_vehicle, dtype=np.uint8)
    # seg_vehicle = cv2.morphologyEx(seg_vehicle, cv2.MORPH_OPEN, kernel)
    # seg_vehicle = cv2.morphologyEx(seg_vehicle, cv2.MORPH_CLOSE, kernel)
    # seg_road = (seg_road) & (seg_road != seg_vehicle)

    seg_road = reshape_to_ori(seg_road, rgb_frame.shape)
    seg_vehicle = reshape_to_ori(seg_vehicle, rgb_frame.shape)

    start = time.time()
    answer_key[frame] = [encode(seg_vehicle), encode(seg_road)]
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
