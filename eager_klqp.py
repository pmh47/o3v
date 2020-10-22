
from enum import Enum
import itertools
import tensorflow as tf
import tensorflow_probability as tfp
import distributions as local_distributions


class _RvBuilder:

    import tensorflow_probability.python.experimental.edward2.generated_random_variables as tfpegrv
    tfp_distribution_names = set(tfpegrv.rv_dict.keys())
    local_distribution_names = set(local_distributions.rv_all)

    def __init__(self, rv_to_value):
        self._rv_to_value = rv_to_value

    def __getattr__(self, dist_name):
        if dist_name in _RvBuilder.tfp_distribution_names:
            dist = getattr(tfp.distributions, dist_name)
        elif dist_name in _RvBuilder.local_distribution_names:
            dist = getattr(local_distributions, dist_name)
        else:
            raise AttributeError
        return lambda rv_name, *args, **kwargs: \
            self._rv_to_value(dist(name=rv_name, *args, **kwargs), rv_name)


def _get_mode_or_mean(rv, name):
    try:
        return rv.mode()
    except NotImplementedError:
        print('warning: using mean instead of mode for {}'.format(name))
        try:
            return rv.mean()  # fall back to mean, e.g. for uniform random variables
        except NotImplementedError:
            print('warning: even mean not implemented; using sample')
            return rv.sample()


class GenerativeMode(Enum):

    UNCONDITIONED = 1  # i.e. sampling the learnt prior
    CONDITIONED = 2  # i.e. sampling the posterior, with variational samples substituted
    RECONSTRUCTION = 3  # i.e. mode of the posterior


class IntegratedEagerKlqp:

    def __init__(self, generative, variational, integrated_name_to_values, beta=1., verbose=False, conditioning_names=[]):

        assert tf.executing_eagerly()  # as otherwise, handling of randomness in train is incorrect

        self.generative = generative
        self.variational = variational
        self.integrated_name_to_values = integrated_name_to_values
        self.beta = beta
        self.verbose = verbose
        self.conditioning_names = conditioning_names

    def _generate_integrated_values(self):
        names = list(self.integrated_name_to_values.keys())
        for values in itertools.product(*self.integrated_name_to_values.values()):
            yield dict(zip(names, values))

    def _split_observations_and_conditioning(self, name_to_observation_or_conditioning):
        name_to_observation = {
            name: value
            for name, value in name_to_observation_or_conditioning.items()
            if name not in self.conditioning_names
        }
        name_to_conditioning = {
            name: name_to_observation_or_conditioning[name]
            for name in self.conditioning_names
        }
        assert len(name_to_observation) + len(name_to_conditioning) == len(name_to_observation_or_conditioning)
        return name_to_observation, name_to_conditioning

    def sample_prior(self, **name_to_conditioning):

        assert all(name in self.conditioning_names for name in name_to_conditioning), 'variable passed to sample_prior is not in conditioning_names'
        def add_loss(_):
            assert False, 'adding a loss in the unconditioned generative is not supported'  # in theory we could make this do the obvious thing (just add the loss), but the behavior is strange given that all other calls to add_loss are integrated-over
        with tf.name_scope('generative/unconditioned'):
            return self.generative(_RvBuilder(lambda rv, name: rv.sample()), GenerativeMode.UNCONDITIONED, add_loss, **name_to_conditioning)

    def train(self, check_prior, **name_to_observation_or_conditioning):

        name_to_observation, name_to_conditioning = self._split_observations_and_conditioning(name_to_observation_or_conditioning)

        if check_prior:

            # Build the 'prior', i.e. the generative without variational substitutions
            name_to_unconditioned_generative_rv = {}
            def rv_to_value(rv, name):
                assert name not in name_to_unconditioned_generative_rv, 'duplicate variable {} in unconditioned generative'.format(name)
                name_to_unconditioned_generative_rv[name] = rv
                return rv.sample()
            def add_loss(_):
                assert False, 'adding a loss in the unconditioned generative is not supported'  # in theory we could make this do the obvious thing (just add the loss), but the behaviour is strange given that all other calls to add_loss are integrated-over
            with tf.name_scope('generative/unconditioned'):
                unconditioned_generative = self.generative(_RvBuilder(rv_to_value), GenerativeMode.UNCONDITIONED, add_loss, **name_to_conditioning)

            for name in name_to_observation:
                assert name in name_to_unconditioned_generative_rv, 'observed variable {} not present in generative'.format(name)

            for name in self.integrated_name_to_values:
                assert name in name_to_unconditioned_generative_rv, 'integrated variable {} not present in generative'.format(name)
                assert name not in name_to_observation, 'integrated variable {} may not also be observed'.format(name)

        if tf.executing_eagerly():
            random_seed = int(tf.random.uniform([], 0, tf.int32.max, dtype=tf.int32))
        elif len(self.integrated_name_to_values) > 0:
            print('warning: IntegratedEagerKlqp does not reuse randomness correctly in graph mode!')

        total_weighted_log_Px = 0.
        total_weighted_log_Pz = 0.
        total_weighted_log_Qz = 0.
        total_weighted_additional_loss = 0.
        for integrated_name_to_value in self._generate_integrated_values():

            # Ensure that we use the same values for all random variables (up to conditioning on the integrated variables) for each
            # value of the integrated-over variables in the summation
            if tf.executing_eagerly():
                tf.random.set_random_seed(random_seed)  # note that this relies on us actually being in eager mode!

            additional_losses = []
            def add_loss(loss):
                additional_losses.append(loss)

            # Build the variational, also using variational samples for ancestral substitutions
            name_to_substituted_value = dict(name_to_observation)
            name_to_variational_rv = {}
            def rv_to_value(rv, name):
                if check_prior:
                    assert name in  name_to_unconditioned_generative_rv, 'variational rv {} not present in generative'.format(name)
                assert name not in name_to_variational_rv, '{} already has variational binding'.format(name)
                assert name not in name_to_observation, '{} may not be given by variational, as it is observed'.format(name)
                name_to_variational_rv[name] = rv
                if name in integrated_name_to_value:
                    substituted_value = integrated_name_to_value[name]
                else:
                    substituted_value = rv.sample()
                name_to_substituted_value[name] = substituted_value
                return substituted_value
            with tf.name_scope('variational/conditioned'):
                self.variational(_RvBuilder(rv_to_value), add_loss, **name_to_observation, **name_to_conditioning)

            # Build the 'conditioned generative', with values substituted from the variational and observations
            name_to_conditioned_generative_rv = {}
            def rv_to_value(rv, name):
                assert name not in name_to_conditioned_generative_rv, 'duplicate variable {} in conditioned generative'.format(name)
                if name not in name_to_substituted_value:
                    assert name not in integrated_name_to_value, 'variable {} is integrated over, but has no variational distribution; this case is not supported'.format(name)
                    print('warning: variable {} has neither variational distribution nor observed value, hence will be marginalised by sampling'.format(name))
                    name_to_substituted_value[name] = rv.sample()
                name_to_conditioned_generative_rv[name] = rv
                return name_to_substituted_value[name]
            with tf.name_scope('generative/conditioned'):
                self.generative(_RvBuilder(rv_to_value), GenerativeMode.CONDITIONED, add_loss, **name_to_conditioning)

            def mean_over_nonbatch_axes(x):
                shape = x.shape
                if len(shape) < 2:  # should never be zero; ideally it is always one, if batch-vs-event indexing of RVs is correct
                    return x
                else:
                    return tf.reduce_mean(x, axis=tuple(range(1, len(shape))))

            log_Px = sum(
                mean_over_nonbatch_axes(name_to_conditioned_generative_rv[name].log_prob(name_to_substituted_value[name]))
                for name in name_to_observation
            )
            log_Pz = sum(
                mean_over_nonbatch_axes(name_to_conditioned_generative_rv[name].log_prob(name_to_substituted_value[name]))
                for name in name_to_variational_rv  # variational not generative so we only include things with variational (not prior) substitutions
                if name not in name_to_observation  # ...as it's in P(x) instead
            )
            log_Qz = sum(
                mean_over_nonbatch_axes(name_to_variational_rv[name].log_prob(name_to_substituted_value[name]))
                for name in name_to_variational_rv
            )

            Q_integrated_values = tf.exp(sum([
                mean_over_nonbatch_axes(name_to_variational_rv[name].log_prob(value))
                for name, value in integrated_name_to_value.items()
            ], 0.))  # :: iib

            additional_loss = sum(additional_losses)

            total_weighted_log_Px += tf.reduce_mean(Q_integrated_values * log_Px)
            total_weighted_log_Pz += tf.reduce_mean(Q_integrated_values * log_Pz)
            total_weighted_log_Qz += tf.reduce_mean(Q_integrated_values * log_Qz)
            total_weighted_additional_loss += tf.reduce_mean(Q_integrated_values * additional_loss)

        beta = self.beta() if callable(self.beta) else self.beta
        loss = -(total_weighted_log_Px + beta * (total_weighted_log_Pz - total_weighted_log_Qz)) + total_weighted_additional_loss  # :: iib

        if self.verbose:
            if tf.executing_eagerly():  # if we can, print with nice formatting (i.e. two decimal places)
                print('log P(x) = {:.2f}, beta * KL= {:.2f} (log P(z) = {:.2f}, log Q(z) = {:.2f}), L* = {:.2f}, total loss = {:.2f}'.format(
                    total_weighted_log_Px,
                    beta * (total_weighted_log_Pz - total_weighted_log_Qz),
                    total_weighted_log_Pz,
                    total_weighted_log_Qz,
                    total_weighted_additional_loss, loss
                ))
            else:
                tf.print(
                    'log P(x) = ', total_weighted_log_Px,
                    ', beta * KL= ', beta * (total_weighted_log_Pz - total_weighted_log_Qz),
                    ' (log P(z) = ', total_weighted_log_Pz,
                    ', log Q(z) = ', total_weighted_log_Qz,
                    '), L* = ', total_weighted_additional_loss,
                    ', total loss = ', loss,
                    sep=''
                )

        if check_prior:
            return loss, unconditioned_generative
        else:
            return loss

    def reconstruct(self, **name_to_observation_or_conditioning):

        name_to_observation, name_to_conditioning = self._split_observations_and_conditioning(name_to_observation_or_conditioning)

        # Build a copy of the variational, with the (variational) mode of each variable substituted, in order to do
        # a full 'ancestrally modal' reconstruction in the non-MF case
        name_to_variational_mode = {}
        def rv_to_value(rv, name):
            assert name not in name_to_variational_mode, 'duplicate variable {} in modal variational'.format(name)
            assert name not in name_to_observation, '{} may not be given by variational, as it is observed'.format(name)
            mode = _get_mode_or_mean(rv, name)
            name_to_variational_mode[name] = mode
            return mode
        with tf.name_scope('variational/reconstruction'):
            self.variational(_RvBuilder(rv_to_value), lambda _: None, **name_to_observation, **name_to_conditioning)

        # This copy of the generative substitutes variational modes for ancestral latents, but does not substitute observations
        def rv_to_value(rv, name):
            if name in name_to_variational_mode:
                return name_to_variational_mode[name]
            else:
                return _get_mode_or_mean(rv, name)
        with tf.name_scope('generative/reconstruction'):
            reconstruction_modes = self.generative(_RvBuilder(rv_to_value), GenerativeMode.RECONSTRUCTION, lambda _: None, **name_to_conditioning)

        return reconstruction_modes

