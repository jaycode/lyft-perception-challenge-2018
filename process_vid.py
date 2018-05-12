import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

import tensorflow as tf
import os.path
import cv2
import scipy.misc
import numpy as np
from helper import cut_hood, reshape_to_ori
import matplotlib.pyplot as plt
import datetime
import errno

MODEL_DIR = "./saved_models/2018-05-11-1539"
RESULTS_DIR = "saved_results"
IMG_SIZE = (256, 256)

# Area of Interest percentage (from the top)
AOI_PERC = 0.83

DEFAULT_VIDEO = "../Example/test_video.mp4"

file = sys.argv[-1]

if file == os.path.basename(__file__):
  print("Load default video", DEFAULT_VIDEO)
  print("To load input video run `python demo.py video_path.mp4`")
  file = DEFAULT_VIDEO

# Define encoder function
def encode(array):
    print("array to encode:",array.shape)
    pil_img = Image.fromarray(array)
    if pil_img.mode != 'I':
        pil_img = pil_img.convert('I')
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def decode(packet):
    img = base64.b64decode(packet)
    filename = './image.png'
    with open(filename, 'wb') as f:
        f.write(img)
    result = scipy.misc.imread(filename)
    return result

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

sess = tf.Session()
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

meta_graph_def = tf.saved_model.loader.load(sess,
                                            [tf.saved_model.tag_constants.SERVING], MODEL_DIR)

image_input = sess.graph.get_tensor_by_name('image_input:0')
keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
logits = sess.graph.get_tensor_by_name('logits:0')

for rgb_frame in video:

    image = cv2.resize(cut_hood(rgb_frame, perc_retain=AOI_PERC),
                       (IMG_SIZE[1], IMG_SIZE[0]))

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_input: [image]})

    # Cars
    im_softmax_vehicle = im_softmax[0][:, 2].reshape(IMG_SIZE[0], IMG_SIZE[1])
    binary_car_result = \
      (im_softmax_vehicle >= 0.50)
    # print("rgb_frame:", rgb_frame.shape)
    binary_car_result = reshape_to_ori(binary_car_result, rgb_frame.shape, AOI_PERC)

    # Road
    im_softmax_road = im_softmax[0][:, 1].reshape(IMG_SIZE[0], IMG_SIZE[1])
    binary_road_result = \
      (im_softmax_road >= 0.50)
    binary_road_result = reshape_to_ori(binary_road_result, rgb_frame.shape, AOI_PERC)

    print("shape:", binary_car_result.shape)
    print("car pixels:", np.sum(binary_car_result == 1))
    print("non-car pixels:", np.sum(binary_car_result == 0))
    print("road pixels:", np.sum(binary_road_result == 1))
    print("non-road pixels:", np.sum(binary_road_result == 0))

    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]

    print("DECODE")
    print("image shape:", decode(answer_key[frame][0]).shape)
    print("car pixels:", np.sum(decode(answer_key[frame][0]) == 1))
    print("non-car pixels:", np.sum(decode(answer_key[frame][0]) == 0))
    print("road pixels:", np.sum(decode(answer_key[frame][1]) == 1))
    print("non-road pixels:", np.sum(decode(answer_key[frame][1]) == 0))

    print("\n")

    # Increment frame
    frame+=1

# Store output in a JSON file
today = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

try:
    os.makedirs(RESULTS_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
save_file = os.path.join(RESULTS_DIR, "{}.json".format(today))
with open(save_file, 'w') as outfile:
    json.dump(answer_key, outfile)

# Print output in proper json format
print ("Results saved at", save_file)
