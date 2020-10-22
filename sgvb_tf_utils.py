
import math
from collections import defaultdict
import numpy as np
import tensorflow as tf
import cv2


def gaussian_pyramid(pixels, levels=None, no_channels=False):

    # pixels is indexed by *, y, x, channel; dimensions * and channel are assumed to have static shape
    # If levels is None, then it is set automatically such that the smallest scale is 1x1
    # If no_channels is True, then neither x nor the result includes the trailing channel dimension

    if no_channels:
        pixels = pixels[..., np.newaxis]

    if levels is None:
        size = max(int(pixels.get_shape()[-2]), int(pixels.get_shape()[-3]))
        levels = int(math.ceil(math.log(size) / math.log(2))) + 1

    assert levels > 0  # includes the original scale

    kernel_sigma = 1.
    kernel_size = 3
    kernel_1d = cv2.getGaussianKernel(kernel_size, kernel_sigma)
    kernel = tf.constant(np.tile((kernel_1d * kernel_1d.T)[:, :, np.newaxis, np.newaxis], [1, 1, int(pixels.get_shape()[-1]), 1]), dtype=tf.float32)

    pyramid = [tf.reshape(pixels, [-1] + pixels.get_shape()[-3:].as_list())]
    for level in range(levels - 1):
        downsampled = tf.nn.depthwise_conv2d(pyramid[-1], kernel, [1, 2, 2, 1], 'SAME')
        pyramid.append(downsampled)

    # original_size = tf.cast(tf.size(pixels), tf.float32)
    # return [level * original_size / tf.cast(tf.size(level), tf.float32) for level in pyramid]
    result_with_channels = [tf.reshape(level, pixels.get_shape()[:-3].concatenate(level.get_shape()[-3:])) for level in pyramid]

    if no_channels:
        return [result_level[..., 0] for result_level in result_with_channels]
    else:
        return result_with_channels


