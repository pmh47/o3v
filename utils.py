
import numpy as np
import tensorflow as tf
import base64
import os
import cv2
import scipy.cluster.vq

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.

    Reference: https://arxiv.org/abs/1803.08494.

    "Group Normalization", Yuxin Wu, Kaiming He

    Args:
    inputs: A Tensor with at least 2 dimensions one which is channels. All
     shape dimensions must be fully defined.
    groups: Integer. Divide the channels into this number of groups over which
      normalization statistics are computed. This number must be commensurate
      with the number of channels in `inputs`.
    channels_axis: An integer. Specifies index of channels axis which will be
      broken into `groups`, each of which whose statistics will be computed
      across. Must be mutually exclusive with `reduction_axes`. Preferred usage
      is to specify negative integers to be agnostic as to whether a batch
      dimension is included.
    reduction_axes: Tuple of integers. Specifies dimensions over which
       statistics will be accumulated. Must be mutually exclusive with
       `channels_axis`. Statistics will not be accumulated across axes not
       specified in `reduction_axes` nor `channel_axis`. Preferred usage is to
       specify negative integers to be agnostic to whether a batch dimension is
       included.

      Some sample usage cases:
        NHWC format: channels_axis=-1, reduction_axes=[-3, -2]
        NCHW format: channels_axis=-3, reduction_axes=[-2, -1]

    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    *_initializer: Optional initializers for beta and gamma
    mean_close_to_zero: The mean of `input` before ReLU will be close to zero
      when batch size >= 4k for Resnet-50 on TPU. If `True`, use
      `nn.sufficient_statistics` and `nn.normalize_moments` to calculate the
      variance. This is the same behavior as `fused` equals `True` in batch
      normalization. If `False`, use `nn.moments` to calculate the variance.
      When `mean` is close to zero, like 1e-4, use `mean` to calculate the
      variance may have poor result due to repeated roundoff error and
      denormalization in `mean`.  When `mean` is large, like 1e2,
      sum(`input`^2) is so large that only the high-order digits of the elements
      are being accumulated. Thus, use sum(`input` - `mean`)^2/n to calculate
      the variance has better accuracy compared to (sum(`input`^2)/n - `mean`^2)
      when `mean` is large.

    Returns:
    A `Tensor` representing the output of the operation.

    Raises:
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
    ValueError: If number of groups is not commensurate with number of channels.
    ValueError: If reduction_axes or channels_axis are out of bounds.
    ValueError: If reduction_axes are not mutually exclusive with channels_axis.
    """

    def __init__(
        self,
        groups=32,
        channels_axis=-1,
        reduction_axes=(-3, -2),
        center=True,
        scale=True,
        epsilon=1e-6,
        activation_fn=None,
        beta_initializer='zeros',
        gamma_initializer='ones',
        mean_close_to_zero=False,
        trainable=True,
        name=None,
        **kwargs
    ):

        super(GroupNormalization, self).__init__(name=name, trainable=trainable, **kwargs)

        self.groups = groups
        self.channels_axis = channels_axis
        self.reduction_axes = reduction_axes
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.activation_fn = activation_fn
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.mean_close_to_zero = mean_close_to_zero

    def build(self, input_shape):

        input_shape = tensor_shape.TensorShape(input_shape)

        if input_shape.ndims is None:
            raise ValueError('Input has undefined rank.')
        if self.channels_axis > (input_shape.ndims - 1):
            raise ValueError('Axis is out of bounds.')

        # Standardize the channels_axis to be positive and identify # of channels.
        if self.channels_axis < 0:
            self.channels_axis = input_shape.ndims + self.channels_axis
        self.channels = input_shape[self.channels_axis].value

        if self.channels is None:
            raise ValueError('Input has undefined channel dimension.')

        # Standardize the reduction_axes to be positive.
        reduction_axes = list(self.reduction_axes)
        for i in range(len(reduction_axes)):
            if reduction_axes[i] < 0:
                reduction_axes[i] += input_shape.ndims

        for a in reduction_axes:
            if a > input_shape.ndims:
                raise ValueError('Axis is out of bounds.')
            if input_shape[a].value is None:
                raise ValueError('Input has undefined dimensions %d.' % a)
            if self.channels_axis == a:
                raise ValueError('reduction_axis must be mutually exclusive with channels_axis')
            if self.groups > self.channels:
                raise ValueError('Invalid groups %d for %d channels.' % (self.groups, self.channels))
            if self.channels % self.groups != 0:
                raise ValueError('%d channels is not commensurate with %d groups.' % (self.channels, self.groups))

        # Determine axes before channels. Some examples of common image formats:
        #  'NCHW': before = [N], after = [HW]
        #  'NHWC': before = [NHW], after = []
        self.axes_before_channels = input_shape.as_list()[:self.channels_axis]
        self.axes_after_channels = input_shape.as_list()[self.channels_axis + 1:]

        # Manually broadcast the parameters to conform to the number of groups.
        self.params_shape_broadcast = ([1] * len(self.axes_before_channels) + [self.groups, self.channels // self.groups] + [1] * len(self.axes_after_channels))

        # Determine the dimensions across which moments are calculated.
        self.moments_axes = [self.channels_axis + 1]
        for a in reduction_axes:
            if a > self.channels_axis:
                self.moments_axes.append(a + 1)
            else:
                self.moments_axes.append(a)

        params_shape = [self.channels]
        if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
            params_dtype = dtypes.float32
        else:
            params_dtype = self.dtype or dtypes.float32

        self.beta = self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=params_shape,
                dtype=params_dtype,
                initializer=self.beta_initializer,
                trainable=True
            )
        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=params_shape,
                dtype=params_dtype,
                initializer=self.gamma_initializer,
                trainable=True
            )

    def call(self, inputs, **kwargs):

        inputs = ops.convert_to_tensor(inputs)
        original_shape = inputs.get_shape()

        # Reshape the input by the group within the channel dimension.
        inputs_shape = (self.axes_before_channels + [self.groups, self.channels // self.groups] + self.axes_after_channels)
        inputs = array_ops.reshape(inputs, inputs_shape)

        # Calculate the moments.
        if self.mean_close_to_zero:
            # One pass algorithm returns better result when mean is close to zero.
            counts, means_ss, variance_ss, _ = tf.nn.sufficient_statistics(inputs, self.moments_axes, keep_dims=True)
            mean, variance = tf.nn.normalize_moments(counts, means_ss, variance_ss, shift=None)
        else:
            mean, variance = tf.nn.moments(inputs, self.moments_axes, keep_dims=True)

        # Compute normalization.
        gain = math_ops.rsqrt(variance + self.epsilon)
        offset = -mean * gain
        if self.gamma is not None:
            gamma = array_ops.reshape(self.gamma, self.params_shape_broadcast)
            gain *= gamma
            offset *= gamma
        if self.beta is not None:
            beta = array_ops.reshape(self.beta, self.params_shape_broadcast)
            offset += beta
        outputs = inputs * gain + offset

        # Collapse the groups into the channel dimension.
        outputs = array_ops.reshape(outputs, original_shape)

        return outputs


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = tf.keras.backend.epsilon() * tf.keras.backend.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = tf.keras.backend.mean(inputs, axis=-1, keepdims=True)
        variance = tf.keras.backend.mean(tf.keras.backend.square(inputs - mean), axis=-1, keepdims=True)
        std = tf.keras.backend.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class ResDense(tf.keras.layers.Layer):

    def __init__(self, activation):
        super(ResDense, self).__init__()
        self._activation = activation

    def build(self, input_shape):
        self._dense = tf.keras.layers.Dense(input_shape[-1], activation=self._activation)

    def call(self, input, training=None, mask=None):
        return input + self._dense(input)


class Residual(tf.keras.layers.Layer):

    def __init__(self, inner_layer):

        super(Residual, self).__init__()
        self._inner_layer = inner_layer

    def call(self, inputs, training=None, mask=None):

        return inputs + self._inner_layer(inputs)


def reduce_alpha(colour, alpha, background, axis):

    # colour is indexed by A0..n, layer, B0..m, channel (where A* and B* are optional sequences of axes)
    # alpha is indexed by A0..n, layer, B0..m (matching the leading axes of colour)
    # background is indexed by A0..n, B0..m, channel
    # axis names the 'layer' axis of colour/alpha, i.e. layer = n
    # Index zero (along axis) represents the most distant layer

    # ** we could presumably avoid the unstack by fancy indexing
    colour_layers = tf.unstack(colour, axis=axis)
    alpha_layers = tf.unstack(alpha[..., None], axis=axis)

    result = background
    for colour_layer, alpha_layer in zip(colour_layers, alpha_layers):
        result = result * (1 - alpha_layer) + colour_layer * alpha_layer

    return result


def make_chequerboard(width, height, batch_dims=[], spacing=16, first_colour=[0.35, 0.5, 0.35], second_colour=[1., 0.7, 1.]):

    chequerboard = tf.stack(tf.meshgrid(tf.range(width), tf.range(height)), axis=-1) // spacing
    chequerboard = tf.cast((chequerboard[..., :1] + chequerboard[..., 1:]) % 2, tf.float32)
    chequerboard = chequerboard * first_colour + (1. - chequerboard) * second_colour
    return tf.tile(chequerboard[(None,) * len(batch_dims)], batch_dims +  [1, 1, 1])


def to_homogeneous(x):
    return tf.concat([x, tf.ones_like(x[..., :1])], axis=-1)


def dot(a, b):
    assert a.shape[-1] == b.shape[-1]
    return tf.reduce_sum(a * b, axis=-1)

