
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Distribution, Bernoulli, Normal, Laplace, Uniform, FULLY_REPARAMETERIZED
from tensorflow_probability.python.internal.distribution_util import gen_new_seed
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops

from sgvb_tf_utils import gaussian_pyramid


rv_all = [
    'NormalPyramid',
]


class NormalPyramid(Distribution):

    # means is indexed by *, y, x, channel
    # base_sigma is a scalar, and is the std used for the 'raw' pixels; subsequent levels use smaller stds

    def __init__(self, means, base_sigma, levels=None, validate_args=False, allow_nan_stats=True, name='NormalPyramid'):
        with ops.name_scope(name, values=[means, base_sigma]) as ns:
            self._means = array_ops.identity(means, name='means')
            self._base_sigma = array_ops.identity(base_sigma, name='base_sigma')
            self._base_dist = Normal(loc=self._means, scale=self._base_sigma)
            self._standard_normal = Normal(loc=0., scale=1.)
            self._levels = levels
            super(NormalPyramid, self).__init__(
                dtype=tf.float32,
                parameters={'means': means, 'base_sigma': base_sigma},
                reparameterization_type=FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=ns
            )

    def _log_prob(self, x):
        # The resulting density here will be indexed by *, i.e. we sum over x, y, channel, and pyramid-levels
        z = (x - self._means) / self._base_sigma
        z_shape = list(map(int, z.get_shape()))
        z_pyramid = gaussian_pyramid(z, self._levels)
        return sum(
            tf.reduce_mean(self._standard_normal.log_prob(z_level), axis=[-3, -2, -1])  # ** check the rescaling here!
            for level_index, z_level in enumerate(z_pyramid)
        ) / len(z_pyramid)

    def _sample_n(self, n, seed=None):
        return self._base_dist._sample_n(n, seed)

    def _mean(self):
        return self._means

    def _mode(self):
        return self._means


