"""
Trying out ENet (does not work yet)
"""
import sys
import os.path
import tensorflow as tf
import helper2 as helper
import warnings
from distutils.version import LooseVersion
import time
import datetime

sys.path.append("models")
from ENet import ENet

TRAINING_DIR = '../lyft_training_data/Train/'
TEST_DIR = '../lyft_training_data/Test/'
RGB_DIR = 'CameraRGB'
SEG_DIR = 'CameraSeg'

SAVE_MODEL_DIR = './saved_models/'

# height, width

# IMG_SIZE = (600, 800)
IMG_SIZE = (300, 400)
# IMG_SIZE = (224, 224)
# IMG_SIZE = (256, 256)
# IMG_SIZE = (128, 128)
# IMG_SIZE = (64, 64)
# IMG_SIZE = (32, 32)

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    for epoch in range(epochs):
        print("epoch: ", epoch)
        batch = 0
        for images, labels in get_batches_fn(batch_size):
            # Training
            start = time.time()
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image:images,
                                          correct_label:labels})
            end = time.time()
            print('batch = ', batch, ', loss = ', loss, ', time = ', end-start)
            batch += 1
    pass

def custom_loss(network, labels):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=labels)
    loss = tf.reduce_mean(losses)
    return loss

def run():
    num_classes = 3
    image_shape = IMG_SIZE
    data_dir = TRAINING_DIR
    runs_dir = './runs'
    epochs = 2
    batch_size = 1
    learning_rate=1e-5

    net_input = tf.placeholder(
        tf.float32,shape=[None,image_shape[0], image_shape[1],3],
        name="net_input")
    net_output = tf.placeholder(
        tf.float32,shape=[None,image_shape[0], image_shape[1],num_classes],
        name="net_output") 
    
    logits, probabilities = ENet(net_input,
                                 num_classes,
                                 batch_size=batch_size,
                                 is_training=True,
                                 reuse=None,
                                 num_initial_blocks=1,
                                 stage_two_repeat=2,
                                 skip_connections=False)

    network = tf.reshape(probabilities, (-1, num_classes), name='logits')
    loss = custom_loss(network, net_output)

    # annotations = tf.reshape(annotations, shape=[batch_size, image_shape[0], image_shape[1]])
    # annotations_ohe = tf.one_hot(annotations, num_classes, axis=-1)

    opt = tf.train.AdamOptimizer(1e-4).minimize(loss,
        var_list=[var for var in tf.trainable_variables()])

    with tf.Session() as sess:

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir), RGB_DIR, SEG_DIR, image_shape)

        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver()

        # Runs training
        sess.run(init_op)
        train_nn(sess, epochs, batch_size, get_batches_fn, opt, loss, net_input,
                 net_output, learning_rate)

        # Save the trained model
        today = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
        save_dir = os.path.join(SAVE_MODEL_DIR, today)
        helper.save_model(sess, net_input, network, save_dir)

        print("SavedModel saved at {}".format(save_dir))

        test_dir = TEST_DIR
        helper.save_inference_samples(runs_dir, test_dir, sess, image_shape,
                                      network, net_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()