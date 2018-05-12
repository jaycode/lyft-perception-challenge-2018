import tensorflow as tf
from glob import glob
import os.path

TEST_DIR = '../lyft_training_data/Test'
MODEL_DIR = 'saved_models/2018-05-10-2219'
IMG_SIZE = (224, 224)

sess = tf.Session()
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

meta_graph_def = tf.saved_model.loader.load(sess,
                                            [tf.saved_model.tag_constants.SERVING], MODEL_DIR)

signature = meta_graph_def.signature_def

# These code doesn't work somehow???
# image_input_name = signature[signature_key].inputs['image_input'].name
# keep_prob_input_name = signature[signature_key].inputs['keep_prob'].name
# logits_output_name = signature[signature_key].outputs['logits'].name
# print("image_input_name:", image_input_name)

# So we manually write the names
image_input = sess.graph.get_tensor_by_name('image_input:0')
keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
logits = sess.graph.get_tensor_by_name('logits:0')

# y_out = sess.run(y, {x: 3.0})