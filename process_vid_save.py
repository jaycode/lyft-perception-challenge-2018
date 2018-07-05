import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

import tensorflow as tf
import os.path
import cv2
import scipy.misc
import numpy as np
from helper2 import reshape_to_ori, softmax
import matplotlib.pyplot as plt
import datetime
import errno
from moviepy.editor import VideoFileClip

MODEL_DIR = "./saved_models/2018-05-26-1309"
RESULTS_DIR = "saved_results"
IMG_SIZE = (256, 256)

DEFAULT_VIDEO = "../Example/test_video.mp4"

NET_INPUT_NAME = "net_input:0"
LOGITS_NAME = "logits_1:0"
# LOGITS_NAME = "logits/BiasAdd:0"

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

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=config)

signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

meta_graph_def = tf.saved_model.loader.load(sess,
                                            [tf.saved_model.tag_constants.SERVING], MODEL_DIR)

# print("all names:")
# print(tf.all_variables())
image_input = sess.graph.get_tensor_by_name(NET_INPUT_NAME)
logits = sess.graph.get_tensor_by_name(LOGITS_NAME)

for rgb_frame in video:

    image = cv2.resize(rgb_frame, (IMG_SIZE[1], IMG_SIZE[0]))

    output_image = sess.run(
        logits, feed_dict={image_input: [image]})

    output_image = np.array(output_image[0,:,:,:])
    im_softmax_road = output_image[:, :, 1]
    im_softmax_vehicle = output_image[:, :, 2]

    seg_road = \
      ((im_softmax_road >= 0.50) & (im_softmax_road >= im_softmax_vehicle))
        
    seg_vehicle = \
      ((im_softmax_vehicle >= 0.50) & (im_softmax_vehicle >= im_softmax_road))

    kernel = np.ones((11,11),np.uint8)
    seg_vehicle =  np.array(seg_vehicle, dtype=np.uint8)
    seg_vehicle = cv2.morphologyEx(seg_vehicle, cv2.MORPH_CLOSE, kernel)
    seg_road = (seg_road) & (seg_road != seg_vehicle)

    seg_road = reshape_to_ori(seg_road, rgb_frame.shape)
    seg_vehicle = reshape_to_ori(seg_vehicle, rgb_frame.shape)


    # print("shape:", seg_vehicle.shape)
    # print("car pixels:", np.sum(seg_vehicle == 1))
    # print("non-car pixels:", np.sum(seg_vehicle == 0))
    # print("road pixels:", np.sum(seg_road == 1))
    # print("non-road pixels:", np.sum(seg_road == 0))

    answer_key[frame] = [encode(seg_vehicle), encode(seg_road)]

    # print("DECODE")
    # print("image shape:", decode(answer_key[frame][0]).shape)
    # print("car pixels:", np.sum(decode(answer_key[frame][0]) == 1))
    # print("non-car pixels:", np.sum(decode(answer_key[frame][0]) == 0))
    # print("road pixels:", np.sum(decode(answer_key[frame][1]) == 1))
    # print("non-road pixels:", np.sum(decode(answer_key[frame][1]) == 0))

    # print("\n")

    # Increment frame
    frame+=1

# Store output in a JSON file
now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

try:
    os.makedirs(RESULTS_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
save_json = os.path.join(RESULTS_DIR, "{}.json".format(now))
with open(save_json, 'w') as outfile:
    json.dump(answer_key, outfile)


class Answer():
    def __init__(self):
        self.frame = 0

    def draw_answer(self, rgb_frame):
        street_im = scipy.misc.toimage(rgb_frame)
        print("frame", self.frame)
        if self.frame in answer_key.keys():
            result = answer_key[self.frame]
            print("result exists")
            seg_car = decode(result[0])
            seg_road = decode(result[1])

            seg_car = seg_car.reshape(rgb_frame.shape[0], rgb_frame.shape[1], 1)
            seg_road = seg_road.reshape(rgb_frame.shape[0], rgb_frame.shape[1], 1)
            
            mask = np.dot(seg_road, np.array([[0, 255, 0, 127]])) + \
                   np.dot(seg_car, np.array([[255, 0, 0, 127]]))

            mask = scipy.misc.toimage(mask, mode="RGBA")

            street_im.paste(mask, box=None, mask=mask)
        self.frame += 1
        street_im = scipy.misc.fromimage(street_im)
        return street_im

a = Answer()
save_mp4 = os.path.join(RESULTS_DIR, "{}.mp4".format(now))
clip2 = VideoFileClip(file)
clip = clip2.fl_image(a.draw_answer)
clip.write_videofile(save_mp4, audio=False)

# Print output in proper json format
print ("Results saved at", save_json)
# print (json.dumps(answer_key))
