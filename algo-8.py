"""
- Correct brightness augmentation
- Save on big loss
"""
import sys
import os.path, errno
import tensorflow as tf
import helper6 as helper
import warnings
from distutils.version import LooseVersion
import time
import datetime
import argparse
import shutil
import cv2

sys.path.append("models")
from MobileUNet import build_mobile_unet

# TRAINING_DIRS = ['../lyft_training_data/Train/', '../training_data_1/*/']
# TRAINING_DIRS = ['../lyft_training_data/Train_small/']
TRAINING_DIRS = ['../lyft_training_data/Train/', '../training_data_1/*/',
                 '../lyft_training_data/Train/']

TEST_DIR = '../lyft_training_data/Test/CameraRGB'
RGB_DIR = 'CameraRGB'
SEG_DIR = 'CameraSeg'

SAVE_MODEL_DIR = './saved_models/'
SAVE_BIG_LOSS_DIR = './big_loss/'

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
parser.add_argument('--training_dir', type=str, default=None, help='Use this training dir instead. Needs to contain both {} and {} directories'.format(RGB_DIR, SEG_DIR))
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for')
parser.add_argument('--load_model', type=str, default=None, help='Path to the model to load')
parser.add_argument('--load_logits_name', type=str, default='logits_1:0', help='Loaded logits name')
parser.add_argument('--load_net_input_name', type=str, default='net_input:0', help='Loaded net_input name')
parser.add_argument('--load_net_output_name', type=str, default='net_output:0', help='Loaded net_output name')
parser.add_argument('--load_optimizer_name', type=str, default='optimizer', help='Loaded optimizer name')
parser.add_argument('--load_loss_name', type=str, default='loss:0', help='Loaded loss name')
parser.add_argument('--img_height', type=int, default=256, help='Height of final input image to network')
parser.add_argument('--img_width', type=int, default=256, help='Width of final input image to network')
parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=10, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--brightness', type=str2bool, default=True, help='Whether to randomly change the image brightness for data augmentation. Boolean.')
parser.add_argument('--rotation', type=float, default=5.0, help='Whether to randomly rotate the cars for data augmentation. Specifies the max rotation angle.')
parser.add_argument('--save_big_loss', type=float, default=None, help='Whether to store results with big losses. Images with loss larger than this setting will be saved.')
parser.add_argument('--save_big_loss_epochs', type=int, default=50, help='Start saving images on big loss scores after this many epochs.')
args = parser.parse_args()

if args.training_dir is not None:
    TRAINING_DIRS = [args.training_dir]

IMG_SIZE = (args.img_height, args.img_width)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, learning_rate, saver, checkpoint_path, network, save_dir,
             save_big_loss=None, save_big_loss_epochs=50):
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
    :param saver: tf.Saver object
    :param checkpoint_path: Path to checkpoint file
    :param network: TF model
    :param save_dir: Dir to save model
    """
    # TODO: Implement function

    
    for epoch in range(epochs):
        print("epoch: ", epoch)
        batch = 0
        for images, labels, aug_params in get_batches_fn(batch_size):
            # Training
            start = time.time()
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image:images,
                                          correct_label:labels})
            end = time.time()
            print(('batch: {} loss: {} flip: {} ' + \
                  'gamma: {:.4f} rotation: {:.4f} ' + \
                  'time: {:.4f}').format(
                  batch, loss, aug_params[0], aug_params[1], aug_params[2], end-start))

            # Save bad results
            if save_big_loss is not None and epoch >= save_big_loss_epochs and \
               loss > save_big_loss:
                for i, img in enumerate(images):
                    filename = 'e{}-f{}-g{:.4f}-r{:.4f}-l{:.4f}.png'.format(epoch,
                                                            aug_params[0],
                                                            aug_params[1],
                                                            aug_params[2], loss)
                    path = os.path.join(SAVE_BIG_LOSS_DIR, RGB_DIR, filename)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

                    seg = helper.labels2seg(labels[i])
                    path = os.path.join(SAVE_BIG_LOSS_DIR, SEG_DIR, filename)
                    cv2.imwrite(path, seg, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            batch += 1

        # Save the trained model
        if os.path.exists(save_dir):
            save_dir1 = '{}-1'.format(save_dir)
            print("creating SavedModel at {}".format(save_dir1))
            helper.save_model(sess, input_image, network, save_dir1)
            print("replacing SavedModel {} with {}".format(save_dir, save_dir1))
            shutil.rmtree(save_dir, ignore_errors=True)
            os.rename(save_dir1, save_dir)
            shutil.rmtree(save_dir1, ignore_errors=True)
        else:
            helper.save_model(sess, input_image, network, save_dir)
        
        print("SavedModel saved at {}".format(save_dir))

        saver.save(sess, checkpoint_path, write_meta_graph=True)
        print("Saved checkpoint to", checkpoint_path)

    pass

def custom_loss(network, labels):
    # https://gist.github.com/Mistobaan/337222ac3acbfc00bdac

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

        # Prepares saver and loads checkpoint if any found.
        saver = tf.train.Saver(max_to_keep=1)
        today = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
        save_dir = os.path.join(SAVE_MODEL_DIR, today)
        checkpoint_path = os.path.join(SAVE_MODEL_DIR, '{}-ckpt'.format(today), 'model.ckpt')

        # Runs training
        sess.run(init_op)

        if args.load_model is not None:
            load_checkpoint_path = os.path.join('{}-ckpt'.format(args.load_model), 'model.ckpt')
            if os.path.exists('{}-ckpt'.format(args.load_model)):
                print("Loads checkpoint", load_checkpoint_path)
                saver.restore(sess, load_checkpoint_path)
            else:
                print("Checkpoint", load_checkpoint_path, "not found. Restart training instead.")

        try:
            os.makedirs(os.path.join(SAVE_BIG_LOSS_DIR, RGB_DIR))
            os.makedirs(os.path.join(SAVE_BIG_LOSS_DIR, SEG_DIR))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        train_nn(sess, epochs, batch_size, get_batches_fn, opt, loss, net_input,
                 net_output, learning_rate, saver, checkpoint_path, network, save_dir,
                 args.save_big_loss, args.save_big_loss_epochs)


if __name__ == '__main__':
    run()