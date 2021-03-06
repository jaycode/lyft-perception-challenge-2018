"""
Multiple training directories
"""

import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2

# 0: nothing, 1: road + roadlines, 2: vehicles
SEG_MAP = {
    6: 1, 7: 1,
    10: 2
}

# Area of Interest percentage (from the top)
AOI_PERC = 0.83


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def remove_hood(seg_img, perc_retain=0.9, car_id=10):
    """
    Remove the hood part of the car from the
    segmentation image
    :param seg_img: The segmentation image
    """
    height = seg_img.shape[0]
    seg_img[int(height*perc_retain):, :][
      np.where(seg_img[int(height*perc_retain):, :] == car_id)] = 0
    return seg_img

def remap_seg(seg, seg_map):
    """
    Re-map segmentation image
    :param seg: Segmentation image, 2-dimensional [width:height]
    :param seg_map: dict object, mapping of old_id to new_id,
                    unmapped ids are converted to 0.
    """
    keys = list(seg_map.keys())
    
    # Convert unmapped cells
    seg[~np.isin(seg, keys)] = 0
    
    # Convert cells with relevant keys
    for key in keys:
        seg[seg == key] = seg_map[key]
    return seg

def seg2labels(seg, num_classes=3):
    """
    Convert segmentation image into multi channels,
    that may be used as labels to use in the neural network.
    :return: Multi-channel image
    """
    labels = np.zeros((seg.shape[0], seg.shape[1], num_classes))
    for c in range(num_classes):
        layer = np.zeros(seg.shape)
        layer[seg == c] = True
        labels[:,:, c] = layer
    return labels

def data_augmentation(input_image, output_image, args):

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.brightness:
        factor = random.uniform(-1*args.brightness, args.brightness)
        table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)
    return input_image, output_image

def gen_batch_function(data_folders, rgb_dir, seg_dir, args):
    """
    Generate function to create batches of training data
    """
    image_shape = (args.img_height, args.img_width)
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        rgb_paths = []
        seg_paths = []

        for data_folder in data_folders:
            rgb_paths += glob(os.path.join(data_folder, rgb_dir, '*.png'))
            seg_paths += glob(os.path.join(data_folder, seg_dir, '*.png'))

        random_ids = np.random.permutation(len(rgb_paths))
        for batch_i in range(0, len(rgb_paths), batch_size):
            rgbs = []
            labels = []
            for i, rgb_file in enumerate(np.take(rgb_paths, random_ids[batch_i:batch_i+batch_size])):
                seg_file = np.take(seg_paths, random_ids[batch_i+i])
                rgb = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
                seg = remove_hood(cv2.imread(seg_file, cv2.IMREAD_COLOR)[:, :, 2],
                                  perc_retain=AOI_PERC)
                rgb, seg = data_augmentation(rgb, seg, args)
                rgb = cv2.resize(rgb,
                                 (image_shape[1], image_shape[0]))
                
                seg = cv2.resize(seg,
                                 (image_shape[1], image_shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
                seg = remap_seg(seg, SEG_MAP)
                
                rgbs.append(rgb)
                labels.append(seg2labels(seg))

            yield np.array(rgbs), np.array(labels)
    return get_batches_fn


def reshape_to_ori(mask, ori_img_shape):
    scale_height = int(ori_img_shape[0])
    scale_width = int(ori_img_shape[1])
    # print("mask shape before:", mask.shape)
    mask = np.array(mask, dtype=np.uint8)
    # print("mask shape mid:", mask.shape)
    mask = cv2.resize(mask, (scale_width, scale_height), interpolation=cv2.INTER_NEAREST)
    # print("mask shape after:", mask.shape)
    return mask

def gen_test_output(sess, logits, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, '*.png')):
        ori_img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        image = cv2.resize(ori_img, (image_shape[1], image_shape[0]))

        output_image = sess.run(
            logits, feed_dict={image_pl: [image]})

        output_image = np.array(output_image[0,:,:,:])
        im_softmax_road = output_image[:, :, 1]
        im_softmax_vehicle = output_image[:, :, 2]

        seg_road = \
          ((im_softmax_road >= 0.50) & (im_softmax_road >= im_softmax_vehicle)). \
            reshape(image_shape[0], image_shape[1], 1)
            
        seg_vehicle = \
          ((im_softmax_vehicle >= 0.50) & (im_softmax_vehicle >= im_softmax_road)) . \
            reshape(image_shape[0], image_shape[1], 1)

        mask = np.dot(seg_road, np.array([[0, 255, 0, 127]])) + \
               np.dot(seg_vehicle, np.array([[255, 0, 0, 127]]))
        
        mask = reshape_to_ori(mask, ori_img.shape)

        mask = scipy.misc.toimage(mask, mode="RGBA")
        
        street_im = scipy.misc.toimage(ori_img)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, input_image, data_dir, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def save_model(sess, input_image, logits, save_dir):
    builder = tf.saved_model.builder.SavedModelBuilder(save_dir)

    tensor_info_input_image = tf.saved_model.utils.build_tensor_info(input_image)
    tensor_info_logits = tf.saved_model.utils.build_tensor_info(logits)

    print("tensor_info_input_image:", tensor_info_input_image)
    print("tensor_info_logits:", tensor_info_logits)

    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'net_input': tensor_info_input_image},
          outputs={'logits': tensor_info_logits},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              prediction_signature 
      },
      )
    builder.save()
    return save_dir

def softmax1(x):
    """Compute softmax values for each sets of scores in x."""
    max_val = np.max(x, axis=0)
    print("max_val shape:", max_val.shape)
    x[:, 0] = x[:, 0] - max_val[0]
    x[:, 1] = x[:, 1] - max_val[1]
    x[:, 2] = x[:, 2] - max_val[2]
    e_x = np.exp(x)
    print("e_x shape:", e_x.shape)
    sum_e_x = e_x.sum(axis=0)
    e_x[:, 0] = e_x[:, 0] - sum_e_x[0]
    e_x[:, 1] = e_x[:, 1] - sum_e_x[1]
    e_x[:, 2] = e_x[:, 2] - sum_e_x[2]
    return e_x

def softmax(x):
    print("np.exp(x):", np.exp(x).shape)
    print("np.sum(np.exp(x), axis=0):", np.sum(np.exp(x), axis=0).shape)
    sm = np.exp(x) / np.sum(np.exp(x), axis=0)
    # return sm / np.max(sm, axis=0)
    return sm

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis = -1)
    return x
