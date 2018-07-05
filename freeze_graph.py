import tensorflow as tf

saver = tf.train.import_meta_graph('./2018-06-02-2151-ckpt-hsv-e5/model.ckpt.meta', clear_devices=True)
# saver = tf.train.import_meta_graph('./dogs-cats-model/dogs-cats-model.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./2018-06-02-2151-ckpt-hsv-e5/model.ckpt")
# saver.restore(sess, "./dogs-cats-model/dogs-cats-model")

# output_node_names="logits_1"
output_node_names="logits_1/BiasAdd"
# output_node_names="y_pred"
output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, # The session
    input_graph_def, # input_graph_def is useful for retrieving the nodes 
    output_node_names.split(",")  
)

output_graph="./frozen_model.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
 
sess.close()
