from main import framework

import sys
import os
from copy import deepcopy
from glob import glob
from unittest import mock

import numpy as np

if framework == "keras":
    import tensorflow as tf


def test_safe(func):
    """
    Isolate tests
    """
    def func_wrapper(*args):
        with tf.Graph().as_default():
            result = func(*args)
        print('Tests Passed for ' + str(func))
        return result

    return func_wrapper


def _prevent_print(function, params):
    sys.stdout = open(os.devnull, "w")
    function(**params)
    sys.stdout = sys.__stdout__


def _assert_tensor_shape(tensor, shape, display_name):
    assert tf.compat.v1.assert_rank(tensor, len(shape), message='{} has wrong rank'.format(display_name))

    tensor_shape = tensor.get_shape().as_list() if len(shape) else []

    wrong_dimension = [ten_dim for ten_dim, cor_dim in zip(tensor_shape, shape)
                       if cor_dim is not None and ten_dim != cor_dim]
    assert not wrong_dimension, \
        '{} has wrong shape.  Found {}'.format(display_name, tensor_shape)


class TmpMock(object):
    """
    Mock a attribute.  Restore attribute when exiting scope.
    """
    def __init__(self, module, attrib_name):
        self.original_attrib = deepcopy(getattr(module, attrib_name))
        setattr(module, attrib_name, mock.MagicMock())
        self.module = module
        self.attrib_name = attrib_name

    def __enter__(self):
        return getattr(self.module, self.attrib_name)

    def __exit__(self, type, value, traceback):
        setattr(self.module, self.attrib_name, self.original_attrib)


@test_safe
def test_load_vgg(load_vgg, tf_module):
    with TmpMock(tf_module.compat.v1.saved_model.loader, 'load') as mock_load_model:
        vgg_path = ''
        sess = tf.compat.v1.Session()
        test_input_image = tf.compat.v1.placeholder(tf.float32, name='image_input')
        test_keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        test_vgg_layer3_out = tf.compat.v1.placeholder(tf.float32, name='layer3_out')
        test_vgg_layer4_out = tf.compat.v1.placeholder(tf.float32, name='layer4_out')
        test_vgg_layer7_out = tf.compat.v1.placeholder(tf.float32, name='layer7_out')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        assert mock_load_model.called, \
            'tf.saved_model.loader.load() not called'
        assert mock_load_model.call_args == mock.call(sess, ['vgg16'], vgg_path), \
            'tf.saved_model.loader.load() called with wrong arguments.'

        assert input_image == test_input_image, 'input_image is the wrong object'
        assert keep_prob == test_keep_prob, 'keep_prob is the wrong object'
        assert vgg_layer3_out == test_vgg_layer3_out, 'layer3_out is the wrong object'
        assert vgg_layer4_out == test_vgg_layer4_out, 'layer4_out is the wrong object'
        assert vgg_layer7_out == test_vgg_layer7_out, 'layer7_out is the wrong object'


@test_safe
def test_layers(layers):
    num_classes = 2
    vgg_layer3_out = tf.compat.v1.placeholder(tf.float32, [None, None, None, 256])
    vgg_layer4_out = tf.compat.v1.placeholder(tf.float32, [None, None, None, 512])
    vgg_layer7_out = tf.compat.v1.placeholder(tf.float32, [None, None, None, 4096])
    layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

    _assert_tensor_shape(layers_output, [None, None, None, num_classes], 'Layers Output')



@test_safe
def test_for_kitti_dataset(data_dir):
    # Note: should do auto download of this dataset if not present, like maybe_download_pretrained_vgg().
    # http://www.cvlibs.net/datasets/kitti/eval_road.php
    # --> https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip
    # Downloaded using wget in WSL shell
    # Extracted to CarND-Semantic-Segmentation\data\data_road

    kitti_dataset_path = os.path.join(data_dir, 'data_road')
    training_labels_count = len(glob(os.path.join(kitti_dataset_path, 'training/gt_image_2/*_road_*.png')))
    training_images_count = len(glob(os.path.join(kitti_dataset_path, 'training/image_2/*.png')))
    testing_images_count = len(glob(os.path.join(kitti_dataset_path, 'testing/image_2/*.png')))

    assert not (training_images_count == training_labels_count == testing_images_count == 0),\
        'Kitti dataset not found. Extract Kitti dataset in {}'.format(kitti_dataset_path)
    assert training_images_count == 289, 'Expected 289 training images, found {} images.'.format(training_images_count)
    assert training_labels_count == 289, 'Expected 289 training labels, found {} labels.'.format(training_labels_count)
    assert testing_images_count == 290, 'Expected 290 testing images, found {} images.'.format(testing_images_count)
