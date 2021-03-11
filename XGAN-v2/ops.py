import math
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
# from tensorflow.python.compiler.tensorrt import trt_convert as trt

from utils import *


def batch_norm(x, name="batch_norm"):
    return tf.nn.batch_normalization(x, variance=0.9, scale=True, variance_epsilon=1e-5, name=name)
    # return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def instance_norm(input, name="instance_norm"):
    with tf.compat.v1.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.compat.v1.get_variable("scale", [depth],
                                          initializer=tf.compat.v1.random_normal_initializer(1.0, 0.02,
                                                                                             dtype=tf.float32))
        offset = tf.compat.v1.get_variable("offset", [depth], initializer=tf.compat.v1.constant_initializer(0.0))
        mean, variance = tf.nn.moments(x=input, axes=[1, 2], keepdims=True)
        epsilon = 1e-5
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    return tf.keras.layers.Conv2D(filters=output_dim, kernel_size=ks, strides=s, padding=padding.lower(), activation=None,
                           bias_initializer=None,
                           kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev), name=name)(input_)
    '''
    with tf.compat.v1.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                           weights_initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)
    '''


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    return tf.keras.layers.Conv2DTranspose(filters=output_dim, kernel_size=ks, strides=s, activation=None,
                           kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev), padding="same", name=name)(input_)
    '''
    with tf.compat.v1.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                     weights_initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)
    '''


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    with tf.compat.v1.variable_scope(scope or "Linear"):
        matrix = tf.compat.v1.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                           tf.compat.v1.random_normal_initializer(stddev=stddev))
        bias = tf.compat.v1.get_variable("bias", [output_size],
                                         initializer=tf.compat.v1.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
