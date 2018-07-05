import sys, json, base64
import numpy as np

import tensorflow as tf
import cv2
import numpy as np

GRAPH_PATH = "./frozen_model.pb"
IMG_SIZE = (256, 256)
ORI_IMG_SIZE = (600, 800)

NET_INPUT_NAME = "net_input:0"
LOGITS_NAME = "logits_1:0"
# LOGITS_NAME = "logits/BiasAdd:0"

file = sys.argv[-1]

def reshape_to_ori(mask):
    return cv2.resize(np.array(mask, dtype=np.uint8), (ORI_IMG_SIZE[1], ORI_IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)

# Define encoder function
def encode(array):
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")


answer_key = {}

# Frame numbering starts at 1
frame = 1

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

with tf.gfile.GFile(GRAPH_PATH, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map=None,
                        return_elements=None,
                        name="")
        
## NOW the complete graph with values has been restored
logits = graph.get_tensor_by_name(LOGITS_NAME)
## Let's feed the images to the input placeholders
image_input = graph.get_tensor_by_name(NET_INPUT_NAME)
sess= tf.Session(graph=graph, config=config)

kernel = np.ones((3,3), np.uint8)

video = cv2.VideoCapture(file)
while video.isOpened():
    try:
        image = cv2.resize(cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2RGB), (IMG_SIZE[1], IMG_SIZE[0]))

        output_image = sess.run(
            logits, feed_dict={image_input: [image]})

        output_image = np.array(output_image[0,:,:,:])

        seg_road = \
          ((output_image[:, :, 1] >= 0.50) & (output_image[:, :, 1] >= output_image[:, :, 2]))

        seg_vehicle = \
          ((output_image[:, :, 2] >= 0.50) & (output_image[:, :, 2] >= output_image[:, :, 1]))

        seg_vehicle = cv2.dilate(np.array(seg_vehicle, dtype=np.uint8), kernel, iterations=1)
        seg_road = (seg_road) & (seg_road != seg_vehicle)

        seg_road = reshape_to_ori(seg_road)
        seg_vehicle = reshape_to_ori(seg_vehicle)

        answer_key[frame] = [encode(seg_vehicle), encode(seg_road)]

        # Increment frame
        frame+=1
    except Exception:
        print (json.dumps(answer_key))
        video.release()
