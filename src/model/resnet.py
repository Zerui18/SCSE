from collections import namedtuple

import tensorflow as tf
import numpy as np

# from py

import numpy as np
import tensorflow as tf

## TensorFlow helper functions

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

def _relu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')


def _conv(x, filter_size, out_channel, strides, pad='SAME', name='conv'):
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        # Main operation: conv2d
        kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                        tf.float32, initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)
    return conv


def _fc(x, out_dim, name='fc'):
    with tf.variable_scope(name):
        # Main operation: fc
        w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                        tf.float32, initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(1.0/out_dim)))
        b = tf.get_variable('biases', [out_dim], tf.float32,
                            initializer=tf.constant_initializer(0.0))
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (w.name, str(w.get_shape().as_list())))
        fc = tf.nn.bias_add(tf.matmul(x, w), b)

    return fc


def _bn(x, is_train, name='bn'):
    moving_average_decay = 0.9
    with tf.variable_scope(name):
        decay = moving_average_decay
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                        initializer=tf.zeros_initializer(), trainable=False)
        sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                        initializer=tf.ones_initializer(), trainable=False)
        beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                        initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                        initializer=tf.ones_initializer())
        # BN when training
        update = 1.0 - decay
        # with tf.control_dependencies([tf.Print(decay, [decay])]):
            # update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_sigma = sigma.assign_sub(update*(sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        if is_train:
            mean, var = batch_mean, batch_var
        else:
            mean, var = mu, sigma

        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return bn

# back to resnet.py

class ResNet18(object):
    """
    Usage for tf tensor output:
    o = CNN(x).tf_output()

    """

    def __init__(self, input_tensor, is_training):
        self.is_train = is_training
        self._build_network(input_tensor)

    def _build_network(self, input_tensor):
        print('Building model')
        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(input_tensor, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block(x, name='conv2_1')
        x = self._residual_block(x, name='conv2_2')

        # conv3_x
        x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
        x = self._residual_block(x, name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
        x = self._residual_block(x, name='conv4_2')

        # conv5_x
        x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
        x = self._residual_block(x, name='conv5_2')

        self.model = x

    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


    def _residual_block(self, x, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = _conv(x, filter_size, out_channel, stride, pad, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        return x

    def _fc(self, x, out_dim, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = _fc(x, out_dim, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        return x

    def _bn(self, x, name="bn"):
        x = _bn(x, self.is_train, self._global_step, name)
        return x

    def _relu(self, x, name="relu"):
        x = _relu(x, 0.0, name)
        return x

    def tf_output(self):
        return self.model

    def save(self):
        pass