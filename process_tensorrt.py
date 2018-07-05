import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

import tensorflow as tf
import os.path
import cv2
import scipy.misc
import numpy as np
from helper4 import reshape_to_ori
import matplotlib.pyplot as plt
import datetime
import errno

MODEL_DIR = "./saved_models/2018-05-26-1309"
IMG_SIZE = (256, 256)

NET_INPUT_NAME = "net_input:0"
LOGITS_NAME = "logits_1:0"

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=config)

signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

meta_graph_def = tf.saved_model.loader.load(sess,
                                            [tf.saved_model.tag_constants.SERVING], MODEL_DIR)

image_input = sess.graph.get_tensor_by_name(NET_INPUT_NAME)
logits = sess.graph.get_tensor_by_name(LOGITS_NAME)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.67)

trt_graph = trt.create_inference_graph(
                input_graph_def=frozen_graph_def,
                outputs=output_node_name,
                max_batch_size=batch_size,
                max_workspace_size_bytes=workspace_size,
                precision_mode=precision)