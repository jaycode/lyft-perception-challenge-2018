"""
More training data
"""
import sys
import os.path
import tensorflow as tf
import helper4 as helper
import warnings
from distutils.version import LooseVersion
import time
import datetime
import argparse

sys.path.append("models")
from MobileUNet import build_mobile_unet

TRAINING_DIRS = ['../lyft_training_data/Train/', '../training_data_1/*/']
TEST_DIR = '../lyft_training_data/Test/CameraRGB'
RGB_DIR = 'CameraRGB'
SEG_DIR = 'CameraSeg'

SAVE_MODEL_DIR = './saved_models/'

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--load_model', type=str, default=None, help='Path to the model to load')
parser.add_argument('--load_logits_name', type=str, default='logits_1:0', help='Loaded logits name')
parser.add_argument('--load_net_input_name', type=str, default='net_input:0', help='Loaded net_input name')
parser.add_argument('--load_net_output_name', type=str, default='net_output:0', help='Loaded net_output name')
parser.add_argument('--load_optimizer_name', type=str, default='optimizer', help='Loaded optimizer name')
parser.add_argument('--load_loss_name', type=str, default='loss:0', help='Loaded loss name')
parser.add_argument('--img_height', type=int, default=256, help='Height of final input image to network')
parser.add_argument('--img_width', type=int, default=256, help='Width of final input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=10, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--brightness', type=float, default=0.5, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change.')
args = parser.parse_args()

IMG_SIZE = (args.img_height, args.img_width)

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
    loss = tf.reduce_mean(losses, name="loss")
    return loss

def run():
    num_classes = 3
    image_shape = IMG_SIZE
    runs_dir = './runs'
    epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate=1e-5

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as sess:

        if args.load_model is not None:
            meta_graph_def = tf.saved_model.loader.load(sess,
                                            [tf.saved_model.tag_constants.SERVING],
                                            args.load_model)
            graph = tf.get_default_graph()
            net_input = graph.get_tensor_by_name(args.load_net_input_name)
            net_output = graph.get_tensor_by_name(args.load_net_output_name)
            network = graph.get_tensor_by_name(args.load_logits_name)
            loss = graph.get_tensor_by_name(args.load_loss_name)
            opt = graph.get_operation_by_name(args.load_optimizer_name)
        else:
            net_input = tf.placeholder(
                tf.float32,shape=[None,image_shape[0], image_shape[1],3],
                name="net_input")

            
            network = build_mobile_unet(net_input, preset_model = 'MobileUNet-Skip', num_classes=num_classes)

            network = tf.identity(network, name='logits')

            net_output = tf.placeholder(
                tf.float32,shape=[None,image_shape[0], image_shape[1], num_classes],
                name="net_output") 

            loss = custom_loss(network, net_output)
            opt = tf.train.AdamOptimizer(1e-4).minimize(
                loss,
                var_list=[var for var in tf.trainable_variables()],
                name='optimizer')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(TRAINING_DIRS, RGB_DIR, SEG_DIR, args)

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