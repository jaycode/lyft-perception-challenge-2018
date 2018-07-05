import sys, glob, json, base64
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
from moviepy.editor import ImageClip, concatenate_videoclips

MODEL_DIR = "./saved_models/2018-06-03-1517"
RESULTS_DIR = "saved_results"
IMG_SIZE = (256, 256)

DEFAULT_IMAGES_DIR = "../lyft_training_data/Test/new_data/*/CameraRGB"

NET_INPUT_NAME = "net_input:0"
LOGITS_NAME = "logits_1:0"
# LOGITS_NAME = "logits/BiasAdd:0"

fps = 24

images_dir = sys.argv[-1]

if images_dir == os.path.basename(__file__):
    print("Load default images in directory", DEFAULT_IMAGES_DIR)
    print("To load input images run `python demo.py images_dir_path")
    images_dir = DEFAULT_IMAGES_DIR
else:
    print("Load images in", images_dir)

# Define encoder function
def encode(array):
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")

def decode(packet):
    img = base64.b64decode(packet)
    filename = './image.png'
    with open(filename, 'wb') as f:
        f.write(img)
    result = scipy.misc.imread(filename)
    return result

def draw_answer(rgb_frame, seg_car, seg_road):
    street_im = scipy.misc.toimage(rgb_frame)
    seg_car = seg_car.reshape(rgb_frame.shape[0], rgb_frame.shape[1], 1)
    seg_road = seg_road.reshape(rgb_frame.shape[0], rgb_frame.shape[1], 1)
    
    mask = np.dot(seg_road, np.array([[0, 255, 0, 127]])) + \
           np.dot(seg_car, np.array([[255, 0, 0, 127]]))

    mask = scipy.misc.toimage(mask, mode="RGBA")

    street_im.paste(mask, box=None, mask=mask)
    street_im = scipy.misc.fromimage(street_im)
    return street_im

image_paths = sorted(glob.glob(os.path.join(images_dir, '*.png')))

print("Path:", os.path.join(images_dir, '*.png'))
print("Num images:", len(image_paths))

answer_key = {}

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=config)

signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

meta_graph_def = tf.saved_model.loader.load(sess,
                                            [tf.saved_model.tag_constants.SERVING], MODEL_DIR)

image_input = sess.graph.get_tensor_by_name(NET_INPUT_NAME)
logits = sess.graph.get_tensor_by_name(LOGITS_NAME)

frame = 1

# Store output in a JSON file
now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

save_mp4 = os.path.join(RESULTS_DIR, "{}.mp4".format(now))
clips = []

for i, img_path in enumerate(image_paths):

    print ("{}/{}".format(i, len(image_paths)))
    rgb_frame = cv2.imread(img_path)
    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    image = cv2.resize(rgb_frame, (IMG_SIZE[1], IMG_SIZE[0]))

    output_image = sess.run(
        logits, feed_dict={image_input: [image]})

    output_image = np.array(output_image[0,:,:,:])
    im_softmax_road = output_image[:, :, 1]
    im_softmax_vehicle = output_image[:, :, 2]

    seg_road = \
      ((im_softmax_road >= 0.50) & (im_softmax_road >= im_softmax_vehicle))

    seg_road[:128, :] = False
        
    seg_vehicle = \
      ((im_softmax_vehicle >= 0.50) & (im_softmax_vehicle >= im_softmax_road))

    # Unused
    # kernel = np.ones((11,11),np.uint8)
    # seg_vehicle =  np.array(seg_vehicle, dtype=np.uint8)
    # seg_vehicle = cv2.morphologyEx(seg_vehicle, cv2.MORPH_CLOSE, kernel)
    # seg_road = (seg_road) & (seg_road != seg_vehicle)

    # Unused
    # kernel = np.ones((3,3), np.uint8)
    # seg_vehicle = np.array(seg_vehicle, dtype=np.uint8)
    # seg_vehicle = cv2.dilate(seg_vehicle, kernel, iterations=1)

    # Unused
    # seg_road = cv2.morphologyEx(np.array(seg_road, dtype=np.uint8), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    # seg_road = cv2.morphologyEx(np.array(seg_road, dtype=np.uint8), cv2.MORPH_OPEN, kernel)

    # seg_road = (seg_road) & (seg_road != seg_vehicle)


    seg_road = reshape_to_ori(seg_road, rgb_frame.shape)
    seg_vehicle = reshape_to_ori(seg_vehicle, rgb_frame.shape)

    answer_key[frame] = [encode(seg_vehicle), encode(seg_road)]

    clips.append(ImageClip(draw_answer(rgb_frame, seg_vehicle, seg_road)).set_duration(0.06))

    frame += 1


try:
    os.makedirs(RESULTS_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
save_json = os.path.join(RESULTS_DIR, "{}.json".format(now))
with open(save_json, 'w') as outfile:
    json.dump(answer_key, outfile)

print(len(clips))
concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile(save_mp4, audio=False, fps=fps)

# Print output in proper json format
print ("Results saved at", save_json)
# print (json.dumps(answer_key))
