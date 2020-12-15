__author__ = 'moonkey'

#from keras import models, layers
import logging
import numpy as np
# from src.data_util.synth_prepare import SynthGen

#import keras.backend as K
import tensorflow as tf


def var_random(name, shape, regularizable=False):
    '''
    Initialize a random variable using xavier initialization.
    Add regularization if regularizable=True
    :param name:
    :param shape:
    :param regularizable:
    :return:
    '''
    v = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    if regularizable:
        with tf.name_scope(name + '/Regularizer/'):
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(v))
    return v

def MaxPooling2D(incoming, k_size, strides, name):
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize=(1, *k_size, 1), strides=(1, *strides, 1), padding='VALID')

def batch_norm(incoming, is_training):
    return tf.contrib.layers.batch_norm(incoming, is_training=is_training, scale=True, decay=0.99)

def Conv2D(incoming, num_filters, filter_size, strides=(1, 1), padding_type='SAME'):
    num_filters_from = incoming.get_shape().as_list()[3]

    conv_W = var_random('W', tuple(filter_size) + (num_filters_from, num_filters), regularizable=True)

    return tf.nn.conv2d(incoming, conv_W, strides=(1,*strides,1), padding=padding_type)

def BNRelu(incoming, is_training):
    after_bn = batch_norm(incoming, is_training)

    return tf.nn.relu(after_bn)

def ConvBNRelu(incoming, num_filters, filter_size, name, is_training, padding_type = 'SAME'):
    with tf.variable_scope(name):
        after_conv = Conv2D(incoming, num_filters, filter_size, padding_type=padding_type)

        return BNRelu(after_conv, is_training)

def BNReluConv(incoming, num_filters, filter_size, name, is_training, strides=(1, 1), padding_type = 'SAME'):
    with tf.variable_scope(name):
        after_bnrelu = BNRelu(incoming, is_training)

        return Conv2D(after_bnrelu, num_filters, filter_size, strides, padding_type)

# Resnet Layers

def ResidualBlock(incoming, n_filters, blk_id, no_width_subsampling, is_training):
    # pre-activation version
    if blk_id == 1:
        # already got pre-activations
        x = Conv2D(incoming, n_filters, (3, 3))
        x_id = incoming
    else:
        # apply pre-activations
        strides = (1, 2) if no_width_subsampling else (2, 2)
        x = BNReluConv(incoming, n_filters, (3, 3), f'res_block_{blk_id}', is_training, strides) # subsample with stride
        x_id = Conv2D(incoming, n_filters, (1, 1), strides, 'VALID') # subsample id with 1x1 conv
    x = BNReluConv(incoming, n_filters, (3, 3), f'res_block_{blk_id}', is_training)
    return tf.add(x, x_id)

class CNN(object):
    """
    Usage for tf tensor output:
    o = CNN(x).tf_output()

    """

    def __init__(self, input_tensor, is_training):
        self._build_network(input_tensor, is_training)

    def _build_network(self, input_tensor, is_training):
        """
        https://github.com/bgshih/crnn/blob/master/model/crnn_demo/config.lua
        :return:
        """
        print('input_tensor dim: {}'.format(input_tensor.get_shape()))
        net = tf.transpose(input_tensor, perm=[0, 2, 3, 1])
        net = tf.add(net, (-128.0))
        net = tf.multiply(net, (1/128.0))

        # image size is w=135, h=55

        # :: first block: output h=28
        # zero pad height
        z_pad = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
        net = tf.pad(net, z_pad, 'CONSTANT')
        # now h=57
        net = ConvBNRelu(net, 64, (7, 7), 'conv1_conv')
        net = MaxPooling2D(net, (3, 3), (2, 2), 'conv1_maxpool')

        # :: res block 1: output h=14
        net = ResidualBlock(net, 64, 1, no_width_subsampling=False, is_training=is_training)

        # :: res block 2: output h=8
        net = ResidualBlock(net, 128, 2, no_width_subsampling=True, is_training=is_training)
        # zero pad height
        z_pad = tf.constant([[0, 0], [1, 0], [0, 0], [0, 0]])
        net = tf.pad(net, z_pad, 'CONSTANT')

        # :: res block 3: output h=4
        net = ResidualBlock(net, 256, 3, no_width_subsampling=True, is_training=is_training)

        # :: res block 4: output h=2

        net = ResidualBlock(net, 512, 4, no_width_subsampling=True, is_training=is_training)

        # :: res block 5: output h=1

        net = ResidualBlock(net, 512, 5, no_width_subsampling=True, is_training=is_training)

        print('CNN outdim: {}'.format(net.get_shape()))
        self.model = net

    def tf_output(self):
        # if self.input_tensor is not None:
        return self.model
    '''
    def __call__(self, input_tensor):
        return self.model(input_tensor)
    '''
    def save(self):
        pass