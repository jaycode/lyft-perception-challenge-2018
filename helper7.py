"""
helper6 with Grayscale
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

def labels2seg(labels):
    """
    Convert labels into segmentation image.
    Useful to save labels that the system had difficulty with.
    Todo: Generalize with SEG_MAP
    """
    seg = np.zeros((labels.shape[0], labels.shape[1]))
    for c in range(labels.shape[2]):
        x = labels[:,:,c]
        x[x==1] = 6
        x[x==2] = 10
        seg += x
    return seg

def random_gamma():
    # min_brightness = 0.2
    # max_brightness = 9
    min_brightness = 0.4
    max_brightness = 3.0
    factor = random.uniform(-1, 1)
    if factor > 0:
        # brighten
        gamma = 1 + ((max_brightness - 1) * factor)
    else:
        # darken
        gamma = min_brightness + ((1 - min_brightness) * -factor)
    return gamma

def adj_brightness(input_image, gamma):
    if gamma == 0:
        gamma = 0.01
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(input_image, table)

def data_augmentation(input_image, output_image, args):
    gamma = 1
    angle = 0
    flip = 0
    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
        flip = 1
    if args.brightness and random.randint(0,1):
        gamma = random_gamma()
        input_image = adj_brightness(input_image, gamma)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)
    return input_image, output_image, (flip, gamma, angle)

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
            images = []
            labels = []
            for i, rgb_file in enumerate(np.take(rgb_paths, random_ids[batch_i:batch_i+batch_size])):
                seg_file = np.take(seg_paths, random_ids[batch_i+i])
                img = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2GRAY)
                seg = remove_hood(cv2.imread(seg_file, cv2.IMREAD_COLOR)[:, :, 2],
                                  perc_retain=AOI_PERC)
                img = cv2.resize(img,
                                 (image_shape[1], image_shape[0]))
                
                seg = cv2.resize(seg,
                                 (image_shape[1], image_shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
                seg = remap_seg(seg, SEG_MAP)
                img, seg, aug_params = data_augmentation(img, seg, args)
                
                img = img[..., None]
                images.append(img)
                labels.append(seg2labels(seg))

            yield np.array(images), np.array(labels), aug_params
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
