
import os
import sys
import shutil
import tempfile
import numpy as np
from datetime import datetime
from collections import namedtuple, defaultdict
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import matplotlib.cm
import cv2
import dirt
import dirt.matrices
import dirt.lighting
import dirt.projection

import utils
import sgvb_tf_utils
from eager_klqp import IntegratedEagerKlqp, GenerativeMode


output_base_path = './output'


class Hyperparams:

    _task_id = os.getenv('SLURM_ARRAY_TASK_ID')
    rng = np.random.default_rng(seed=int(_task_id)) if _task_id is not None else None

    def __init__(self):

        self._hyperparameter_to_value = {}
        for arg in sys.argv[1:]:
            name, value = arg.split('=')
            if ',' in value:
                values = value.split(',')
                if self.rng is None:
                    raise RuntimeError('hyperparameter {} has {} values specified, but we are not running as a job array'.format(name, len(values)))
                value = self.rng.choice(values)
                print('hyper: randomly selected value {} for {}'.format(value, name))
            self._hyperparameter_to_value[name] = value
        self._requested_hyperparameters = set()

    def __call__(self, default, name, converter=float):

        # Retrieve the value of the given parameter specified on the command-line, or return default
        if name in self._hyperparameter_to_value:
            value = converter(self._hyperparameter_to_value[name])
            use_default = False
        else:
            value = default
            use_default = True
        if name not in self._requested_hyperparameters:
            print('hyper: {} = {}{}'.format(name, value, ' (default)' if use_default else ''))
            self._requested_hyperparameters.add(name)
        return value

    def verify_args(self):

        # Check that no hyperparameters were given on the command-line, that were not also requested through hyper(...)
        missing = []
        for name in self._hyperparameter_to_value:
            if name not in self._requested_hyperparameters:
                missing.append(name)
        if len(missing) > 0:
            raise RuntimeError('the following unrecognised hyperparameters were specified: ' + ', '.join(missing))


hyper = Hyperparams()


episodes_per_batch = hyper(4, 'eppb', int)
frames_per_episode = 6
frame_width, frame_height = 96, 72
random_seed = hyper(0, 'seed', int)
subbatch_count = hyper(3, 'subbatches', int)
final_eval = hyper(0, 'final-eval', int)


def get_unit_cube(vertices_only=False):

    vertices = tf.constant([[x, y, z] for z in [-1, 1] for y in [-1, 1] for x in [-1, 1]], dtype=tf.float32)

    if vertices_only:
        return vertices

    quads = [
        [0, 1, 3, 2], [4, 5, 7, 6],  # back, front
        [1, 5, 4, 0], [2, 6, 7, 3],  # bottom, top
        [4, 6, 2, 0], [3, 7, 5, 1],  # left, right
    ]
    triangles = tf.constant(sum([[[a, b, c], [c, d, a]] for [a, b, c, d] in quads], []), dtype=tf.int32)

    return vertices, triangles


def sample_texture(texture, uvs):

    # texture is indexed by eib, v, u, channel
    # uvs is indexed by eib, fie, y, x, u/v

    uv_indices = uvs * [texture.get_shape()[2].value - 1, texture.get_shape()[1].value - 1]

    vu_indices = uv_indices[..., ::-1]
    vu_indices = tf.maximum(vu_indices, 0.)
    vu_indices = tf.minimum(vu_indices, tf.cast(tf.shape(texture)[1:3], tf.float32) - 1.00001)
    floor_vu_indices = tf.floor(vu_indices)
    frac_vu_indices = vu_indices - floor_vu_indices
    floor_vu_indices = tf.cast(floor_vu_indices, tf.int32)
    floor_evu_indices = tf.concat([
        tf.tile(
            tf.range(episodes_per_batch)[:, None, None, None, None],
            [1, floor_vu_indices.shape[1], frame_height, frame_width, 1]
        ),
        floor_vu_indices
    ], axis=-1)  # :: eib, fie, y, x, eib/y/x

    neighbours = tf.gather_nd(
        texture,
        tf.stack([
            floor_evu_indices,
            floor_evu_indices + [0, 0, 1],
            floor_evu_indices + [0, 1, 0],
            floor_evu_indices + [0, 1, 1]
        ]),
    )  # :: neighbour. eib, fie, y, x, r/g/b
    top_left, top_right, bottom_left, bottom_right = tf.unstack(neighbours)

    texture_samples = \
        top_left * (1. - frac_vu_indices[..., 1:]) * (1. - frac_vu_indices[..., :1]) + \
        top_right * frac_vu_indices[..., 1:] * (1. - frac_vu_indices[..., :1]) + \
        bottom_left * (1. - frac_vu_indices[..., 1:]) * frac_vu_indices[..., :1] + \
        bottom_right * frac_vu_indices[..., 1:] * frac_vu_indices[..., :1]

    return texture_samples


def sample_voxels(
    episode_and_object_indices,  # :: fg-pixel, eib/obj
    locations,  # :: fg-pixel, sample, x/y/z
    voxels  # :: eib, obj, z, y, x, channel
):
    # result :: fg-pixel, sample, channel

    if voxels.shape[2] == 1:
        voxels = tf.tile(voxels, [1, 1, 2, 1, 1, 1])
    if voxels.shape[3] == 1:
        voxels = tf.tile(voxels, [1, 1, 1, 2, 1, 1])
    if voxels.shape[4] == 1:
        voxels = tf.tile(voxels, [1, 1, 1, 1, 2, 1])

    dhw = tf.convert_to_tensor(voxels.shape[2:5], dtype=tf.int64)
    dhw_f = tf.cast(dhw, tf.float32)
    zyx_indices = (locations[:, :, ::-1] + 1.) / 2. * dhw_f

    zyx_indices = tf.maximum(zyx_indices, 0.)
    zyx_indices = tf.minimum(zyx_indices, dhw_f - 1.00001)
    floor_zyx_indices = tf.floor(zyx_indices)  # :: fg-pixel, sample, z/y/x
    frac_zyx_indices = zyx_indices - floor_zyx_indices
    floor_zyx_indices = tf.cast(floor_zyx_indices, tf.int64)
    floor_zyx_indices = tf.concat([
        tf.tile(episode_and_object_indices[:, None], [1, locations.get_shape()[1].value, 1]),
        floor_zyx_indices
    ], axis=-1)  # :: fg-pixel, sample, eib/obj/z/y/x

    neighbours = tf.gather_nd(
        voxels,
        tf.stack([
            floor_zyx_indices + [0, 0, 0, 0, 0], floor_zyx_indices + [0, 0, 0, 0, 1],
            floor_zyx_indices + [0, 0, 0, 1, 0], floor_zyx_indices + [0, 0, 0, 1, 1],
            floor_zyx_indices + [0, 0, 1, 0, 0], floor_zyx_indices + [0, 0, 1, 0, 1],
            floor_zyx_indices + [0, 0, 1, 1, 0], floor_zyx_indices + [0, 0, 1, 1, 1]
        ]),
    )  # :: z-floor/-ceil * y-floor/-ceil * x-floor/-ceil, fg-pixel, sample, channel
    neighbours = tf.reshape(neighbours, tf.concat([[2, 2, 2], tf.shape(neighbours)[1:4]], axis=0))


    one_minus_frac_zyx_indices = 1. - frac_zyx_indices  # :: fg-pixel, sample, z/y/x
    neighbours *= \
        tf.stack([one_minus_frac_zyx_indices[..., 0], frac_zyx_indices[..., 0]], axis=0)[:, None, None, :, :, None] * \
        tf.stack([one_minus_frac_zyx_indices[..., 1], frac_zyx_indices[..., 1]], axis=0)[None, :, None, :, :, None] * \
        tf.stack([one_minus_frac_zyx_indices[..., 2], frac_zyx_indices[..., 2]], axis=0)[None, None, :, :, :, None]

    return tf.reduce_sum(neighbours, axis=[0, 1, 2])


def get_3d_object_depthmaps(object_positions_world, object_azimuths, object_sizes, world_to_view_matrices, projection_matrix, far_depth_value):

    frame_count = object_positions_world.get_shape()[1].value
    object_count = object_positions_world.get_shape()[2].value

    cube_vertices_object, cube_faces = get_unit_cube()
    cube_vertices_object = cube_vertices_object[None, None, :, :] * object_sizes[:, :, None, :] / 2.  # :: eib, obj, vertex, x/y/z

    object_rotation_matrices = dirt.matrices.rodrigues(object_azimuths[..., None] * [0., 1., 0.], three_by_three=True)  # :: eib, fie, obj, x/y/z (in), x/y/z (out)

    cube_vertices_world = tf.matmul(cube_vertices_object[:, None], object_rotation_matrices)  # :: eib, fie, obj, vertex, x/y/z
    cube_vertices_world = cube_vertices_world + object_positions_world[:, :, :, None, :]  # ditto
    cube_vertices_world = tf.concat([cube_vertices_world, tf.ones_like(cube_vertices_world[..., :1])], axis=-1)  # :: eib, fie, obj, vertex, x/y/z/w

    # ** would be nice to share this with clip-to-voxel stuff in render_3d_objects_over_background
    cube_vertices_view = tf.einsum('efovi,efij->efovj', cube_vertices_world, world_to_view_matrices)
    cube_vertices_clip = tf.einsum('efovj,jk->efovk', cube_vertices_view, projection_matrix)  # ditto
    cube_vertices_clip_flat = tf.reshape(cube_vertices_clip, [-1] + cube_vertices_clip.get_shape()[3:].as_list())

    object_depths = dirt.rasterise_batch(
        background=far_depth_value * tf.ones([episodes_per_batch * frame_count * object_count, frame_height, frame_width, 1]),
        vertices=cube_vertices_clip_flat,
        vertex_colors=cube_vertices_clip_flat[..., 3:],
        faces=tf.tile(cube_faces[None], [episodes_per_batch * frame_count * object_count, 1, 1])
    )  # :: eib * fie * obj, y, x, singleton
    object_depths = tf.reshape(object_depths, [episodes_per_batch, frame_count, object_count, frame_height, frame_width])

    return object_depths, object_rotation_matrices


def render_3d_objects_over_background(
    object_voxel_colours,  # :: eib, obj, z, y, x, r/g/b
    object_voxel_alphas,  # :: eib, obj, z, y, x
    object_positions_world,  # :: eib, fie, obj, x/y/z
    object_azimuths,  # :: eib, fie, obj
    object_sizes,  # :: eib, obj, x/y/z
    object_presences,  # :: eib, obj
    background_pixels,  # :: eib, fie, y, x, r/g/b
    background_depth,  # :: eib, fie, y, x
    world_to_view_matrices,  # :: eib, fie, x/y/z/w (in), x/y/z/w (out)
    projection_matrix,  # :: x/y/z/w (in), x/y/z/w (out)
    composed_output=True,  # if set to false, we render each object into its own image
    object_colour_transforms=None  # :: eib, fie, obj, x/y/z/w (in), x/y/z/w (out)
):
    # return background_pixels + 0. * tf.reduce_sum(object_voxel_colours) + tf.reduce_sum(object_positions_world) + tf.reduce_sum(object_azimuths)

    # print(' before (MB):', tf.contrib.memory_stats.BytesInUse().numpy() // 1024 ** 2)

    # print('  background_pixels', background_pixels.shape.num_elements() // 1024 ** 2, tf.contrib.memory_stats.BytesInUse().numpy() // 1024 ** 2)

    object_count = object_positions_world.get_shape()[2].value

    object_initial_distances = -tf.matmul(utils.to_homogeneous(object_positions_world[:, 0]), world_to_view_matrices[:, 0])[..., 2]  # :: eib, obj
    object_indices_farthest_to_nearest = tf.argsort(object_initial_distances, axis=-1, direction='DESCENDING')  # :: eib, obj-by-depth
    object_voxel_colours = tf.gather(object_voxel_colours, object_indices_farthest_to_nearest, batch_dims=1)
    object_voxel_alphas = tf.gather(object_voxel_alphas, object_indices_farthest_to_nearest, batch_dims=1)
    object_positions_world = tf.gather(object_positions_world, object_indices_farthest_to_nearest, axis=2, batch_dims=1)
    object_azimuths = tf.gather(object_azimuths, object_indices_farthest_to_nearest, axis=2, batch_dims=1)
    object_sizes = tf.gather(object_sizes, object_indices_farthest_to_nearest, batch_dims=1)
    object_presences = tf.gather(object_presences, object_indices_farthest_to_nearest, batch_dims=1)
    if object_colour_transforms is not None:
        object_colour_transforms = tf.gather(object_colour_transforms, object_indices_farthest_to_nearest, axis=2, batch_dims=1)

    far_depth_value = 2.e1
    object_depths, object_rotation_matrices = get_3d_object_depthmaps(
        object_positions_world, object_azimuths, object_sizes,
        world_to_view_matrices, projection_matrix,
        far_depth_value
    )

    object_depths = tf.where(tf.greater(object_depths, tf.tile(background_depth[:, :, None, :, :], [1, 1, object_count, 1, 1])), far_depth_value * tf.ones_like(object_depths), object_depths)
    object_masks = tf.cast(tf.not_equal(object_depths, far_depth_value), tf.float32)  # :: eib, fie, obj, y, x
    object_masks = tf.reshape(
        tf.nn.max_pool(tf.reshape(object_masks, [-1, frame_height, frame_width, 1]), [1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME'),
        [episodes_per_batch, frames_per_episode, object_count, frame_height, frame_width]
    )

    if final_eval:
        downsample_factor_x, downsample_factor_y = 1, 1
    else:
        downsample_factor_x, downsample_factor_y = 2, 2
        object_masks = object_masks[:, :, :, ::downsample_factor_y, ::downsample_factor_x]

    fg_pixel_indices = tf.where(object_masks)  # :: fg-pixel, eib/fie/obj/y/x

    fg_pixel_indices = tf.gather(fg_pixel_indices, tf.argsort(fg_pixel_indices[:, 2]))  # thus, now 'grouped' by object-index
    fg_first_pixel_index_by_object = [tf.reduce_min(tf.where(tf.greater_equal(fg_pixel_indices[:, 2], object_index))) for object_index in range(object_count)]

    clip_to_voxel_matrices = tf.linalg.inv(dirt.matrices.compose(
        dirt.matrices.scale(object_sizes[:, None, :] * [1., -1., 1.] / 2.),
        dirt.matrices.pad_3x3_to_4x4(object_rotation_matrices),
        dirt.matrices.translation(object_positions_world),
        world_to_view_matrices[:, :, None],
        projection_matrix[None, None, None]
    ))
    clip_to_voxel_matrices_by_fg_pixel = tf.gather_nd(clip_to_voxel_matrices, fg_pixel_indices[:, :3])

    # print('  clip_to_voxel_matrices_by_fg_pixel', clip_to_voxel_matrices_by_fg_pixel.shape.num_elements() // 1024 ** 2, tf.contrib.memory_stats.BytesInUse().numpy() // 1024 ** 2)

    with tf.device('/cpu:0'):  # ** necessary under CUDA 10.0 due to a bug in cublasGemmBatchedEx
        ray_starts, ray_directions = dirt.projection.unproject_pixels_to_rays(
            tf.cast(fg_pixel_indices[:, 3:][:, ::-1] * [downsample_factor_x, downsample_factor_y], tf.float32) + [(downsample_factor_x - 1) / 2, (downsample_factor_y - 1) / 2] + 0.5,  # the + 0.5 accounts for the fact that OpenGL has pixel centres at half-integer coordinates
            clip_to_voxel_matrices_by_fg_pixel,
            [[frame_width, frame_height]]
        )  # each :: fg-pixel, x/y/z

    # print('  ray_starts', ray_starts.shape.num_elements() // 1024 ** 2, tf.contrib.memory_stats.BytesInUse().numpy() // 1024 ** 2)

    # See http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/; we have the sphere centered
    # at the origin, with radius sqrt(3) so it circumscribes the voxel cube
    quadratic_as = tf.reduce_sum(tf.square(ray_directions), axis=-1)
    quadratic_bs = tf.reduce_sum(ray_directions * ray_starts, axis=-1) * 2.
    quadratic_cs = tf.reduce_sum(tf.square(ray_starts), axis=-1) - 3.  # the 3. here is the square of the circumscribing radius
    discriminants = tf.square(quadratic_bs) - 4. * quadratic_as * quadratic_cs
    discriminants = tf.nn.relu(discriminants) + 1.e-12  # in theory, the relu is an identity, as we should always intersect the sphere
    near_ts = (-quadratic_bs - tf.sqrt(discriminants)) / (2. * quadratic_as)  # :: fg-pixel
    far_ts = (-quadratic_bs + tf.sqrt(discriminants)) / (2. * quadratic_as)

    # sample_count = max(object_voxel_colours.get_shape()[2:5].as_list()) * 5 // 4  # number of samples to take along each ray
    sample_count = 32

    # Note that we store samples in order farthest-to-nearest, which matches reduce_alpha
    most_distant_sample_locations = ray_starts + far_ts[:, None] * ray_directions  # :: fg-pixel, x/y/z
    sample_spacings = (near_ts - far_ts)[:, None] * ray_directions / (sample_count - 1)  # ditto
    sample_locations = most_distant_sample_locations[:, None, :] + sample_spacings[:, None, :] * tf.range(sample_count, dtype=tf.float32)[None, :, None]  # :: fg-pixel, sample, x/y/z

    # print('  post-sample_locations (MB)', tf.contrib.memory_stats.BytesInUse().numpy() // 1024 ** 2)

    # The padding (and corresponding adjustment to sample_locations) adds a one-voxel 'clamping' border to colour,
    # and a zero border to alpha; this ensures correct values are sampled just outside the voxel cuboid, given that
    # sample_voxels clamps the texel coordinates
    original_voxel_dhw = object_voxel_colours.shape[2:5].as_list()
    padded_voxel_dhw = [dim + 2 for dim in original_voxel_dhw]
    padded_voxel_colours = tf.reshape(
        tf.pad(tf.reshape(object_voxel_colours, [-1] + original_voxel_dhw + [3]), [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC'),
        [episodes_per_batch, object_count] + padded_voxel_dhw + [3]
    )
    padded_voxel_alphas = tf.pad(
        object_voxel_alphas,
        [[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]]
    )
    padded_sample_locations = sample_locations * (tf.constant(original_voxel_dhw, dtype=tf.float32) / padded_voxel_dhw[::-1])[::-1]

    sample_colours_and_alphas = sample_voxels(
        fg_pixel_indices[:, 0:3:2],  # :: fg-pixel, eib/obj
        padded_sample_locations,
        tf.concat([
            padded_voxel_colours,
            padded_voxel_alphas[..., None] * object_presences[..., None, None, None, None]
        ], axis=-1)
    )
    sample_colours = sample_colours_and_alphas[..., :3]  # :: fg-pixel, sample, r/g/b
    sample_alphas = sample_colours_and_alphas[..., 3]  # :: fg-pixel, sample

    if object_colour_transforms is not None:
        # This is used when rendering depth-maps; then the 'colour' of each voxel should be its view-space
        # coordinate, but this varies from frame to frame, whereas object_voxel_colours is constant. So, we
        # allow a per-frame transform matrix to be applied after the sampling stage, which is equivalent but
        # keeps memory usage tractable
        # fg_pixels :: fg-pixel, eib/fie/obj/y/x; it is ordered by object
        # object_colour_transforms :: eib, fie, obj, x/y/z/w (in), x/y/z/w (out)
        colour_transforms_by_sample = tf.gather_nd(object_colour_transforms, fg_pixel_indices[:, :3], )  # :: fg-pixel, x/y/z/w (in), x/y/z/w (out)
        sample_colours = tf.einsum('fsi,fij->fsj', utils.to_homogeneous(sample_colours), colour_transforms_by_sample)[:, :, :3]

    # sample_precomposed_colours is the 'full' alpha composition for a given pixel except for the component involving the
    # background, which may itself have been overlaid by a more-distant object; sample_bg_weights is the weighting that
    # the background should receive
    sample_precomposed_colours = utils.reduce_alpha(sample_colours[:, 1:], sample_alphas[:, 1:], sample_colours[:, 0] * sample_alphas[:, 0, None], axis=1)  # :: fg-pixel, r/g/b
    sample_bg_weights = tf.reduce_prod(1. - sample_alphas, axis=1)  # :: fg-pixel

    if composed_output:
        final_pixels = background_pixels
        final_bg_mask = tf.ones([episodes_per_batch, frames_per_episode, frame_height, frame_width])
    else:  # this is used for evaluating segmentation, when we require one mask per object
        final_pixels_by_object = [background_pixels] * object_count

    # ** the composition model here is slightly wrong -- while we iterate objects in farthest-to-nearest order, there
    # ** is no handling of depthwise overlap (should 'interleave' samples from each object)
    for object_index in range(object_count):

        first_fg_pixel_index_for_object = fg_first_pixel_index_by_object[object_index]
        last_fg_pixel_index_for_object = fg_first_pixel_index_by_object[object_index + 1] if object_index < object_count - 1 else fg_pixel_indices.shape[0]
        fg_pixel_indices_for_object = fg_pixel_indices[first_fg_pixel_index_for_object : last_fg_pixel_index_for_object]  # :: fg-pixel-in-obj, eib/fie/obj/y/x
        sample_precomposed_colours_for_object = sample_precomposed_colours[first_fg_pixel_index_for_object : last_fg_pixel_index_for_object]  # :: fg-pixel-in-obj, r/g/b
        sample_bg_weights_for_object = sample_bg_weights[first_fg_pixel_index_for_object : last_fg_pixel_index_for_object]  # :: fg-pixel-in-obj

        if downsample_factor_y == 1 and downsample_factor_x == 1:
            fg_pixel_indices_for_object = tf.concat([fg_pixel_indices_for_object[:, :2], fg_pixel_indices_for_object[:, 3:]], axis=-1)  # :: fg-pixel-in-obj, eib/fie/y/x
        elif downsample_factor_y == 2 and downsample_factor_x == 2:
            fg_pixel_indices_for_object = tf.concat([
                tf.tile(fg_pixel_indices_for_object[:, :2], [downsample_factor_y * downsample_factor_x, 1]),
                tf.concat([
                    fg_pixel_indices_for_object[:, 3:] * [downsample_factor_y, downsample_factor_x],
                    fg_pixel_indices_for_object[:, 3:] * [downsample_factor_y, downsample_factor_x] + [0, 1],
                    fg_pixel_indices_for_object[:, 3:] * [downsample_factor_y, downsample_factor_x] + [1, 0],
                    fg_pixel_indices_for_object[:, 3:] * [downsample_factor_y, downsample_factor_x] + [1, 1]
                ], axis=0)
            ], axis=-1)  # :: fg-pixel-in-obj, eib/fie/y/x
        else:
            assert False

        sample_precomposed_colours_for_object = tf.tile(sample_precomposed_colours_for_object, [downsample_factor_y * downsample_factor_x, 1])
        sample_bg_weights_for_object = tf.tile(sample_bg_weights_for_object, [downsample_factor_y * downsample_factor_x])

        if not composed_output:
            final_pixels = final_pixels_by_object[object_index]

        pixel_backgrounds = tf.gather_nd(final_pixels, fg_pixel_indices_for_object)  # :: fg-pixel-in-obj, r/g/b
        composed_colours = pixel_backgrounds * sample_bg_weights_for_object[:, None] + sample_precomposed_colours_for_object  # :: fg-pixel-in-obj, r/g/b
        final_pixels = tf.tensor_scatter_nd_update(final_pixels, fg_pixel_indices_for_object, composed_colours)

        if not composed_output:
            final_pixels_by_object[object_index] = final_pixels
        else:
            previous_bg_mask = tf.gather_nd(final_bg_mask, fg_pixel_indices_for_object)  # :: fg-pixel-in-obj
            composed_bg_mask = previous_bg_mask * sample_bg_weights_for_object
            final_bg_mask = tf.tensor_scatter_nd_update(final_bg_mask, fg_pixel_indices_for_object, composed_bg_mask)

    if composed_output:
        return final_pixels, 1. - final_bg_mask
    else:
        return tf.stack(final_pixels_by_object, axis=0)


def render_meshes_over_background(
    object_vertices,  # eib, obj, vertex, x/y/z/w
    object_textures,  # eib, obj, v, u, r/g/b; may be the string 'depth' to render a depthmap
    object_faces,  # face, vertex-in-face
    object_uvs,  # vertex, u/v
    object_positions_world,  # eib, fie, obj, x/y/z
    object_azimuths,  # eib, fie, obj
    object_sizes,  # eib, obj, x/y/z
    object_presences,  # eib, obj
    background_pixels,  # eib, fie, y, x, r/g/b
    background_depth,  # eib, fie, y, x
    world_to_view_matrices,  # eib, fie, x/y/z/w (in), x/y/z/w (out)
    projection_matrix,  # x/y/z/w (in), x/y/z/w (out)
    composed_output=True  # if set to false, we render each object into its own image
):
    object_count = object_vertices.get_shape()[1].value

    object_initial_distances = -tf.matmul(utils.to_homogeneous(object_positions_world[:, 0]), world_to_view_matrices[:, 0])[..., 2]  # :: eib, obj
    object_indices_farthest_to_nearest = tf.argsort(object_initial_distances, axis=-1, direction='DESCENDING')  # :: eib, obj-by-depth
    object_vertices = tf.gather(object_vertices, object_indices_farthest_to_nearest, batch_dims=1)
    if object_textures != 'depth':
        object_textures = tf.gather(object_textures, object_indices_farthest_to_nearest, batch_dims=1)
    object_positions_world = tf.gather(object_positions_world, object_indices_farthest_to_nearest, axis=2, batch_dims=1)
    object_azimuths = tf.gather(object_azimuths, object_indices_farthest_to_nearest, axis=2, batch_dims=1)
    object_sizes = tf.gather(object_sizes, object_indices_farthest_to_nearest, batch_dims=1)
    object_presences = tf.gather(object_presences, object_indices_farthest_to_nearest, batch_dims=1)

    object_to_clip_matrices = dirt.matrices.compose(
        dirt.matrices.scale(object_sizes[:, None, :] * [1., -1., 1.] / 2.),
        dirt.matrices.rodrigues(object_azimuths[..., None] * [0., 1., 0.]),
        dirt.matrices.translation(object_positions_world),
        world_to_view_matrices[:, :, None],
        projection_matrix[None, None, None]
    )  # eib, fie, obj, x/y/z/w (in), x/y/z/w (out)

    object_vertices_clip = tf.einsum('eovi,efoij->efovj', object_vertices, object_to_clip_matrices)  # eib, fie, obj, vertex, x/y/z/w

    def do_shading(mask_and_uvs_and_depth, object_presence, object_texture, background):

        mask = mask_and_uvs_and_depth[..., :1] * object_presence[:, None, None, None]
        uvs = mask_and_uvs_and_depth[..., 1:3]
        depth = mask_and_uvs_and_depth[..., 3:]

        if object_textures == 'depth':
            texture_samples = depth
        else:
            uvs = tf.reshape(uvs, [episodes_per_batch, frames_per_episode, frame_height, frame_width, 2])
            texture_samples = tf.reshape(
                sample_texture(object_texture, uvs),
                [episodes_per_batch * frames_per_episode, frame_height, frame_width, 3]
            )

        return texture_samples * mask + background * (1 - mask)

    final_pixels_flat = tf.reshape(background_pixels, [episodes_per_batch * frames_per_episode, frame_height, frame_width, 3])
    if not composed_output:  # this is used for evaluating segmentation, when we require one mask per object
        final_pixels_flat_by_object = []
    for object_index in range(object_count):  # note that objects are already (approximately) sorted farthest-to-nearest

        vertices_for_object_clip_flat = tf.reshape(
            object_vertices_clip[:, :, object_index],
            [-1] + object_vertices_clip.shape[3:].as_list()
        )  # eib * fie, vertex, x/y/z/w
        far_depth_value = 1.e2  # this is the 'background' value for the depth-map; typically all pixels are covered by a wall, so it rarely appears
        final_pixels_flat = dirt.rasterise_batch_deferred(
            background_attributes=tf.zeros([episodes_per_batch * frames_per_episode, frame_height, frame_width, 4]) + [0., 0., 0., far_depth_value],
            vertices=vertices_for_object_clip_flat,
            vertex_attributes=tf.concat([
                tf.ones([episodes_per_batch * frames_per_episode, vertices_for_object_clip_flat.get_shape()[1], 1]),
                tf.tile(object_uvs[None], [episodes_per_batch * frames_per_episode, 1, 1]),
                vertices_for_object_clip_flat[..., 3:]
            ], axis=-1),
            faces=tf.tile(object_faces[None], [episodes_per_batch * frames_per_episode, 1, 1]),
            shader_fn=do_shading,
            shader_additional_inputs=[
                tf.reshape(
                    tf.tile(object_presences[:, None, object_index], [1, frames_per_episode]),
                    [episodes_per_batch * frames_per_episode]
                ),
                object_textures[:, object_index] if object_textures != 'depth' else tf.zeros([]),
                final_pixels_flat if composed_output else tf.zeros([episodes_per_batch * frames_per_episode, frame_height, frame_width, 3])
            ]
        )
        if not composed_output:
            final_pixels_flat_by_object.append(final_pixels_flat)

    if composed_output:
        return tf.reshape(
            final_pixels_flat,
            [episodes_per_batch, frames_per_episode, frame_height, frame_width, 3]
        )
    else:
        return tf.reshape(
            tf.stack(final_pixels_flat_by_object, axis=0),
            [object_count, episodes_per_batch, frames_per_episode, frame_height, frame_width, 3]
        )


def get_voxel_bboxes_at_threshold(threshold, object_alphas, object_presences, voxel_to_view_matrices, voxel_coordinates):

    # threshold is a python float, giving the presence * alpha threshold that should be used to
    # determine which part of an object counts as 'solid'
    # object_alphas :: eib, obj, z, y, x
    # object_presences :: eib, obj
    # voxel_to_view_matrices :: eib, fie, obj, x/y/z/w (in), x/y/z/w (out)
    # voxel_coordinates :: z, y, x, x/y/z; this is a constant tensor passed to avoid repetition, that
    # stores the voxel-space locations of the voxels for any object
    # result :: eib, fie, obj, min-x/-y/-z / max-x/-y/-z / score, where score is presence * max(alpha)
    # and is constant over frames

    # Note that this function includes all objects regardless of visibility; the evaluation code
    # filters to only those which reproject into the frame

    # 1. Downsample alphas and coordinates, to reduce memory usage
    downsample_factor = 3
    object_alphas = tf.nn.max_pool3d(
        tf.reshape(object_alphas, [-1] + object_alphas.get_shape()[2:].as_list() + [1]),
        ksize=downsample_factor, strides=downsample_factor,
        padding='VALID'
    )
    object_alphas = tf.reshape(object_alphas, object_presences.get_shape().as_list() + object_alphas.get_shape()[1:-1].as_list())
    voxel_coordinates = tf.nn.avg_pool3d(
        tf.reshape(voxel_coordinates, [1] + voxel_coordinates.get_shape().as_list()),
        ksize=downsample_factor, strides=downsample_factor,
        padding='VALID'
    )[0]

    # 2. For every (downsampled) point in every object, push it through the voxel-to-view
    # transform and drop the w-coordinate
    voxel_coordinates_view = tf.einsum('zyxi,efoij->efozyxj', utils.to_homogeneous(voxel_coordinates), voxel_to_view_matrices)[..., :3]

    # 3. For each object, find which points have alpha*presence exceeding the threshold,
    # and calculate the bounding-box of such points
    object_alpha_times_presence = object_alphas * object_presences[:, :, None, None, None]
    masks = tf.greater(object_alpha_times_presence, threshold)  # :: eib, obj, z, y, x
    masks_f = tf.cast(masks, tf.float32)[:, None, :, :, :, :, None]  # :: eib, 1, obj, z, y, x, 1
    large_coord = 1.e6  # this should be larger than any actually-occurring coordinate!
    bbox_min_corners = tf.reduce_min(
        voxel_coordinates_view * masks_f + tf.ones_like(voxel_coordinates_view) * large_coord * (1. - masks_f),
        axis=[3, 4, 5]
    )  # :: eib, fie, obj, x/y/z
    bbox_max_corners = tf.reduce_max(
        voxel_coordinates_view * masks_f + tf.ones_like(voxel_coordinates_view) * -large_coord * (1. - masks_f),
        axis=[3, 4, 5]
    )  # :: eib, fie, obj, x/y/z

    # 4. Calculate the object scores, ensuring that any objects that had no greater-than-threshold
    # voxels are assigned exactly zero score (so the evaluation can ignore them)
    scores = tf.reduce_max(object_alpha_times_presence, axis=[2, 3, 4])  # :: eib, obj
    object_partly_visible = tf.reduce_any(masks, axis=[2, 3, 4])  # :: eib, obj
    scores = tf.where(object_partly_visible, scores, tf.zeros_like(scores))

    return tf.concat([
        bbox_min_corners, bbox_max_corners,
        tf.tile(scores[:, None, :, None], [1, frames_per_episode, 1, 1])
    ], axis=-1)


def get_mesh_bboxes(object_vertices, object_positions_world, object_azimuths, object_sizes, object_presences, world_to_view_matrices):

    # ** this duplicates render_meshes_over_background!
    object_to_view_matrices = dirt.matrices.compose(
        dirt.matrices.scale(object_sizes[:, None, :] * [1., -1., 1.] / 2.),
        dirt.matrices.rodrigues(object_azimuths[..., None] * [0., 1., 0.]),
        dirt.matrices.translation(object_positions_world),
        world_to_view_matrices[:, :, None],
    )  # eib, fie, obj, x/y/z/w (in), x/y/z/w (out)

    object_vertices_view = tf.einsum('eovi,efoij->efovj', object_vertices, object_to_view_matrices)[..., :3]  # eib, fie, obj, vertex, x/y/z/w

    bbox_min_corners = tf.reduce_min(object_vertices_view, axis=3)
    bbox_max_corners = tf.reduce_max(object_vertices_view, axis=3)

    return tf.concat([
        bbox_min_corners, bbox_max_corners,
        tf.tile(object_presences[:, None, :, None], [1, frames_per_episode, 1, 1])
    ], axis=-1)


def binarise_and_modalise_masks(masks_by_depth, thresholds):

    # masks_by_depth :: obj eib, fie, y, x
    # thresholds :: *
    # result :: *, obj, eib, fie, y, x

    binarised_masks_by_threshold = tf.cast(tf.greater(masks_by_depth, thresholds[..., None, None, None, None, None]), tf.float32)  # threshold, obj, eib, fie, y, x
    nearer_object_exists = tf.minimum(tf.cumsum(binarised_masks_by_threshold, axis=-5, reverse=True, exclusive=True), 1.)
    return binarised_masks_by_threshold * (1. - nearer_object_exists)


class TfRecordsGroundTruth:

    subfolder = 'Town02-R19_fs-1_{}-fr_1-4-obj'.format(frames_per_episode)
    data_string = 'carla_' + subfolder
    max_bboxes = 5  # may differ from maximum objects visible per frame, as bboxes include out-of-frame instances

    base_scale_xz = 37.5
    base_scale_y = 10.

    def __init__(self, path):

        def decode_serialised_examples(serialised_examples):

            features = tf.io.parse_single_example(
                serialised_examples,
                features={
                    'frames': tf.io.FixedLenFeature([frames_per_episode], tf.string),
                    'segmentations': tf.io.FixedLenFeature([frames_per_episode], tf.string),
                    'depths': tf.io.FixedLenFeature([frames_per_episode], tf.string),
                    'bboxes': tf.io.FixedLenFeature([frames_per_episode, self.max_bboxes, 8, 3], tf.float32),
                    'camera_matrices': tf.io.FixedLenFeature([frames_per_episode, 4, 4], tf.float32),
                })

            frames = tf.map_fn(tf.image.decode_jpeg, features['frames'], tf.uint8)
            frames.set_shape([frames_per_episode, frame_height, frame_width, 3])
            frames = tf.cast(frames, tf.float32) / 255.

            depths = tf.map_fn(tf.image.decode_png, features['depths'], tf.uint8)
            depths.set_shape([frames_per_episode, frame_height, frame_width, 3])
            depths = tf.cast(depths, tf.int32)
            depths = depths[..., 0] + 256 * (depths[..., 1] + 256 * depths[..., 2])
            depths = tf.cast(depths, tf.float32) * 1000 / (256 ** 3 - 1)

            segmentations = tf.map_fn(tf.image.decode_png, features['segmentations'], tf.uint8)
            segmentations.set_shape([frames_per_episode, frame_height, frame_width, 3])

            oriented_bboxes = features['bboxes']  # these are eight vertices in view-space, (redundantly) representing the oriented bbox

            aligned_bboxes = tf.concat([
                tf.reduce_min(oriented_bboxes, axis=2, keepdims=True),
                tf.reduce_max(oriented_bboxes, axis=2, keepdims=True),
            ], axis=2)  # fie, obj, min/max, x/y/z

            return frames, features['camera_matrices'], depths, tf.zeros_like(frames), segmentations, aligned_bboxes

        def get_dataset_for_split(split):
            filenames = tf.data.Dataset.list_files(path + '/' + self.subfolder + '/' + split + '/*')
            dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=4) \
                .map(decode_serialised_examples)
            if split == 'train':  # i.e. val/test datasets are iterated only once, and in deterministic order
                dataset = dataset \
                    .repeat() \
                    .shuffle(episodes_per_batch * 50)
            dataset = dataset.batch(episodes_per_batch, drop_remainder=True)
            if split == 'train':  # i.e. val/test datasets are not subbatched
                dataset = dataset.batch(subbatch_count, drop_remainder=True)
            return dataset.prefetch(10)

        self._split_to_dataset = {
            split: get_dataset_for_split(split)
            for split in ['train', 'val', 'test']
        }

    def get_projection_matrix(self):

        near_plane_distance = 0.05
        fov_angle = 100. * np.pi / 180.
        near_plane_right = near_plane_distance * tf.tan(fov_angle / 2.)
        return dirt.matrices.perspective_projection(
            near_plane_distance,
            far=100.,
            right=near_plane_right,
            aspect=frame_height / frame_width
        )  # :: x/y/x/z (in), x/y/z/w (out)

    def batches(self, split):

        for batch in self._split_to_dataset[split]:
            yield batch


class RoomVae(tf.keras.Model):

    beta = hyper(5.e-1, 'beta')
    beta_anneal_start = hyper(0, 'beta-anneal-start', int)
    beta_anneal_duration = hyper(0, 'beta-anneal-duration', int)
    initial_beta = hyper(beta, 'initial-beta')
    room_circumference_segments = 64
    room_vertical_segments = 24
    scene_embedding_channels = hyper(64, 'sec', int)
    bg_embedding_channels = hyper(24, 'bgec', int)
    bg_shape_bottleneck_channels = 4
    bg_texture_bottleneck_channels = 64

    grid_cells = [5, 1, 7]  # along z, y, x
    total_grid_cells = grid_cells[0] * grid_cells[1] * grid_cells[2]
    object_embedding_channels = 16
    object_representation = hyper('voxels', 'obj-repr', str)
    assert object_representation in {'voxels', 'mesh'}
    if object_representation == 'mesh':
        object_circumference_segments = 16
        object_vertical_segments = 8

    param_string = 'se-{}'.format(scene_embedding_channels)
    param_string += '_bg[det-e-{}_b-{}-{}_sphere_{}c-{}l]'.format(bg_embedding_channels, bg_shape_bottleneck_channels, bg_texture_bottleneck_channels, room_circumference_segments, room_vertical_segments)
    param_string += '_objects[{}det-e-{}]'.format('mesh_' if object_representation == 'mesh' else '', object_embedding_channels)
    param_string += '_beta-{}_4-pyr'.format(beta)

    def build_mesh(self, phi_min, phi_max, theta_max, vertical_segments, circumference_segments):

        vertices = []
        faces = []
        uvs = []
        adjacencies = np.zeros([vertical_segments * circumference_segments, vertical_segments * circumference_segments], dtype=np.float32)
        for phi_index, phi in enumerate(np.linspace(phi_min, phi_max, vertical_segments)):
            for theta_index, theta in enumerate(np.linspace(-theta_max, theta_max, circumference_segments)):
                vertices.append([
                    np.sin(theta) * np.cos(phi),
                    np.sin(phi),
                    -np.cos(theta) * np.cos(phi)
                ])
                uvs.append([
                    theta_index / (circumference_segments - 1),
                    phi_index / (vertical_segments - 1)
                ])
                if phi_index > 0 and theta_index > 0:
                    bottom_right = phi_index * circumference_segments + theta_index
                    assert bottom_right == len(vertices) - 1
                    bottom_left = bottom_right - 1
                    top_right = bottom_right - circumference_segments
                    top_left = top_right - 1
                    faces.extend([
                        [top_left, top_right, bottom_right],
                        [top_left, bottom_right, bottom_left]
                    ])
                    adjacencies[top_left, top_right] = adjacencies[top_right, top_left] = 1.
                    adjacencies[top_right, bottom_right] = adjacencies[bottom_right, top_right] = 1.
                    adjacencies[bottom_right, bottom_left] = adjacencies[bottom_left, bottom_right] = 1.
                    adjacencies[bottom_left, top_left] = adjacencies[top_left, bottom_left] = 1.
                    adjacencies[top_left, bottom_right] = adjacencies[bottom_right, top_left] = 1.

        return vertices, faces, uvs, adjacencies

    def __init__(self, gt):

        super(RoomVae, self).__init__()

        vertices, faces, uvs, adjacency_matrix = self.build_mesh(phi_min=-np.pi / 2., phi_max=1.4, theta_max=np.pi, vertical_segments=self.room_vertical_segments, circumference_segments=self.room_circumference_segments)
        print('room: {} vertices, {} faces'.format(len(vertices), len(faces)))
        self.canonical_vertices = tf.constant(vertices, dtype=tf.float32)
        self.canonical_faces = tf.constant(faces, dtype=tf.int32)
        self.canonical_uvs = tf.constant(uvs, dtype=tf.float32)

        self.canonical_vertices *= [gt.base_scale_xz, gt.base_scale_y, gt.base_scale_xz]

        room_pitch_matrix = dirt.matrices.rodrigues([0. * np.pi / 180, 0., 0.])
        self.canonical_vertices = tf.matmul(utils.to_homogeneous(self.canonical_vertices), room_pitch_matrix)[:, :3]

        self.adjacency_matrix = tf.constant(adjacency_matrix, dtype=tf.float32)
        self.laplacian = tf.eye(int(self.adjacency_matrix.shape[0])) - self.adjacency_matrix / tf.reduce_sum(self.adjacency_matrix, axis=1, keepdims=True)
        self.creases = tf.constant(self.get_creases(faces), dtype=tf.int32)  # ** we could replace get_creases, and just construct the crease data in build_cylinder, alongside the adjacency matrix

        if self.object_representation == 'mesh':
            object_vertices, object_faces, object_uvs, object_adjacency_matrix = self.build_mesh(phi_min=-np.pi / 2, phi_max=np.pi / 2, theta_max=np.pi, vertical_segments=self.object_vertical_segments, circumference_segments=self.object_circumference_segments)
            print('objects: {} vertices, {} faces'.format(len(object_vertices), len(object_faces)))
            self.object_canonical_vertices = tf.constant(object_vertices, dtype=tf.float32) * [1., 1., -1.]  # flip along z so seam is initially away from camera
            self.object_faces = tf.constant(object_faces, dtype=tf.int32)
            self.object_uvs = tf.constant(object_uvs, dtype=tf.float32)

            self.object_adjacency_matrix = tf.constant(object_adjacency_matrix, dtype=tf.float32)
            self.object_laplacian = tf.eye(int(self.object_adjacency_matrix.shape[0])) - self.object_adjacency_matrix / tf.reduce_sum(self.object_adjacency_matrix, axis=1, keepdims=True)
            self.object_creases = tf.constant(self.get_creases(object_faces), dtype=tf.int32)  # ** we could replace get_creases, and just construct the crease data in build_cylinder, alongside the adjacency matrix

        self.projection_matrix = gt.get_projection_matrix()
        self.base_scale_xz = gt.base_scale_xz
        self.base_scale_y = gt.base_scale_y

        self.grid_centres = (tf.stack(tf.meshgrid(
            (tf.range(self.grid_cells[0], dtype=tf.float32) + 0.5) / self.grid_cells[0],
            (tf.range(self.grid_cells[1], dtype=tf.float32) + 0.5) / self.grid_cells[1],
            (tf.range(self.grid_cells[2], dtype=tf.float32) + 0.5) / self.grid_cells[2],
            indexing='ij'
        ), axis=-1)[..., ::-1] - 0.5)  # indexed by grid-z, grid-y, grid-x, x/y/z
        self.grid_centres = self.grid_centres * [gt.base_scale_xz * 0.8, gt.base_scale_y, gt.base_scale_xz * 0.55] - [0., 0.6, 3.5]
        self.grid_mask = np.ones(self.grid_cells, dtype=np.float32)
        self.grid_mask[3, 0, 2:5] = 0.
        self.grid_mask = tf.constant(self.grid_mask)

        if not hyper(1, 'large-enc', int):
            self.temporal_encoder = tf.keras.Sequential(name='temporal_encoder', layers=[
                tf.keras.layers.Conv3D(32, kernel_size=[1, 7, 7], strides=[1, 2, 2], activation=tf.nn.relu),
                utils.GroupNormalization(groups=4, reduction_axes=[-4, -3, -2]),
                tf.keras.layers.Conv3D(48, kernel_size=[1, 3, 3], activation=tf.nn.relu),
                tf.keras.layers.Conv3D(64, kernel_size=[3 if frames_per_episode >= 8 else 2, 1, 1], activation=tf.nn.relu),
                tf.keras.layers.MaxPool3D([2 if frames_per_episode >= 8 else 1, 2, 2]),
                utils.GroupNormalization(groups=4, reduction_axes=[-4, -3, -2]),
                tf.keras.layers.Conv3D(48, kernel_size=[1, 3, 3], activation=tf.nn.relu),
                tf.keras.layers.Conv3D(64, kernel_size=[3 if frames_per_episode >= 8 else 2, 1, 1], activation=tf.nn.relu),
                tf.keras.layers.MaxPool3D([1, 2, 2]),
                utils.GroupNormalization(groups=4, reduction_axes=[-4, -3, -2]),
                tf.keras.layers.Conv3D(128, kernel_size=[1, 3, 3], activation=tf.nn.relu),
                tf.keras.layers.Flatten(),
                utils.LayerNormalization(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                utils.ResDense(activation=tf.nn.relu),
            ])
        else:
            self.temporal_encoder = tf.keras.Sequential(name='temporal_encoder', layers=[
                tf.keras.layers.Conv3D(32, kernel_size=[1, 7, 7], strides=[1, 2, 2], activation=tf.nn.relu),
                utils.GroupNormalization(groups=4, reduction_axes=[-4, -3, -2]),
                tf.keras.layers.Conv3D(64, kernel_size=[1, 3, 3], strides=[1, 2, 2], activation=tf.nn.relu),
                utils.Residual(tf.keras.layers.Conv3D(64, kernel_size=[1, 3, 3], activation=tf.nn.relu, padding='SAME')),
                utils.GroupNormalization(groups=4, reduction_axes=[-4, -3, -2]),
                tf.keras.layers.Conv3D(96, kernel_size=[3 if frames_per_episode >= 8 else 2, 1, 1], activation=tf.nn.relu),
                utils.GroupNormalization(groups=6, reduction_axes=[-4, -3, -2]),
                tf.keras.layers.Conv3D(128, kernel_size=[1, 3, 3], strides=[1, 2, 2], activation=tf.nn.relu),
                utils.Residual(tf.keras.layers.Conv3D(128, kernel_size=[1, 3, 3], activation=tf.nn.relu, padding='SAME')),
                utils.GroupNormalization(groups=4, reduction_axes=[-4, -3, -2]),
                tf.keras.layers.Conv3D(192, kernel_size=[3 if frames_per_episode >= 8 else 2, 1, 1], strides=[2, 1, 1]  if frames_per_episode >= 8 else [1, 1, 1], activation=tf.nn.relu),
                utils.GroupNormalization(groups=6, reduction_axes=[-4, -3, -2]),
                tf.keras.layers.Conv3D(256, kernel_size=[1, 3, 3], activation=tf.nn.relu),
                tf.keras.layers.Flatten(),
                utils.LayerNormalization(),
                tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                utils.ResDense(activation=tf.nn.relu),
            ])
        self.camera_motion_encoder = tf.keras.Sequential(name='camera_motion_encoder', layers=[
            tf.keras.layers.Dense(128, activation=tf.nn.elu),
            utils.LayerNormalization(),
            utils.ResDense(activation=tf.nn.elu),
            utils.LayerNormalization(),
            utils.ResDense(activation=tf.nn.elu),
            utils.LayerNormalization(),
        ])
        self.encoding_to_scene_embedding = tf.keras.layers.Dense(self.scene_embedding_channels * 2)
        self.scene_embedding_to_bg_embedding = tf.keras.Sequential(name='scene_embedding_to_bg_embedding', layers=[
            tf.keras.layers.Dense(128, activation=tf.nn.elu),
            utils.ResDense(tf.nn.elu),
            tf.keras.layers.Dense(self.bg_embedding_channels)
        ])
        self.scene_embedding_to_object_params = tf.keras.Sequential(name='scene_embedding_to_object_params', layers=[
            tf.keras.layers.Dense(128, activation=tf.nn.elu),
            utils.ResDense(tf.nn.elu),
            tf.keras.layers.Dense(
                self.total_grid_cells * (self.object_embedding_channels + 3 + 2 + (frames_per_episode - 1) * (1 + 1) + 1 + 1),
                kernel_initializer=tf.keras.initializers.Zeros()
            )  # appearance-embedding mean/sigma, xyz offset, xz velocity, speed-factors, angular velocities, presence, azimuth
        ])
        self.bg_embedding_to_bottlenecks = tf.keras.layers.Dense(self.bg_shape_bottleneck_channels + self.bg_texture_bottleneck_channels)
        # ** Note this is tied to the values of room_circumference_segments and room_vertical_segments
        self.bg_bottleneck_to_offsets = tf.keras.Sequential(name='bg_embedding_to_offsets', layers=[
            tf.keras.layers.Dense(480, activation=tf.nn.elu),
            tf.keras.layers.Reshape([3, 8, -1]),
            tf.keras.layers.Conv2D(96, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
            utils.Residual(tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation=tf.nn.elu)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(64, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
            utils.Residual(tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation=tf.nn.elu)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(48, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
            utils.Residual(tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation=tf.nn.elu)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(32, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
            utils.Residual(tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation=tf.nn.elu)),
            tf.keras.layers.Conv2D(4, kernel_size=[3, 3], padding='same'),
        ])
        self.bg_bottleneck_to_texture = tf.keras.Sequential(name='bg_embedding_to_texture', layers=[
            tf.keras.layers.Dense(720, activation=tf.nn.elu),
            tf.keras.layers.Reshape([6, 12, -1]),
            tf.keras.layers.Conv2D(128, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
            utils.Residual(tf.keras.layers.Conv2D(filters=128, kernel_size=1, activation=tf.nn.elu)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(96, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
            utils.Residual(tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation=tf.nn.elu)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(64, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
            utils.Residual(tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation=tf.nn.elu)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(48, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
            utils.Residual(tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation=tf.nn.elu)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(32, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
            utils.Residual(tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation=tf.nn.elu)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(24, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
            utils.Residual(tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=tf.nn.elu)),
            tf.keras.layers.Conv2D(3, kernel_size=[3, 3])
        ])
        if self.object_representation == 'voxels':
            if hyper(1, 'large-voxels', int):
                voxel_dec_initial_size = 3
            else:
                voxel_dec_initial_size = 2
            voxel_dec_final_size = voxel_dec_initial_size * 8
            self.object_embedding_to_appearance = tf.keras.Sequential(name='object_embedding_to_appearance', layers=[
                tf.keras.layers.Dense(voxel_dec_initial_size * voxel_dec_initial_size * 64, activation=tf.nn.elu),
                tf.keras.layers.Reshape([voxel_dec_initial_size, voxel_dec_initial_size, 64]),
                tf.keras.layers.UpSampling2D(),
                tf.keras.layers.Conv2D(64, kernel_size=3, padding='SAME', activation=tf.nn.elu),
                tf.keras.layers.UpSampling2D(),
                tf.keras.layers.Conv2D(48, kernel_size=3, padding='SAME', activation=tf.nn.elu),
                tf.keras.layers.UpSampling2D(),
                tf.keras.layers.Conv2D(32, kernel_size=3, padding='SAME', activation=tf.nn.elu),
                tf.keras.layers.Conv2D(voxel_dec_final_size * 4, kernel_size=4, padding='SAME', activation=None),
                tf.keras.layers.Reshape([voxel_dec_final_size, voxel_dec_final_size, voxel_dec_final_size, 4]),
                tf.keras.layers.Permute([3, 1, 2, 4])
            ])
        elif self.object_representation == 'mesh':
            # ** Note this is tied to the values of object_circumference_segments and object_vertical_segments
            self.object_embedding_to_offsets = tf.keras.Sequential(name='object_embedding_to_offsets', layers=[
                tf.keras.layers.Dense(128, activation=tf.nn.elu),
                tf.keras.layers.Reshape([2, 4, -1]),
                tf.keras.layers.Conv2D(96, kernel_size=[2, 2], activation=tf.nn.elu, padding='same'),
                tf.keras.layers.UpSampling2D(),
                tf.keras.layers.Conv2D(64, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
                utils.Residual(tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation=tf.nn.elu)),
                tf.keras.layers.UpSampling2D(),
                tf.keras.layers.Conv2D(48, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
                tf.keras.layers.Conv2D(4, kernel_size=1, kernel_initializer='zeros')
            ])
            self.object_embedding_to_texture = tf.keras.Sequential(name='object_embedding_to_texture', layers=[
                tf.keras.layers.Dense(180, activation=tf.nn.elu),
                tf.keras.layers.Reshape([3, 6, -1]),
                tf.keras.layers.Conv2D(96, kernel_size=[2, 2], activation=tf.nn.elu, padding='same'),
                utils.Residual(tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation=tf.nn.elu)),
                tf.keras.layers.UpSampling2D(),
                tf.keras.layers.Conv2D(64, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
                utils.Residual(tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation=tf.nn.elu)),
                tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear'),
                tf.keras.layers.Conv2D(24, kernel_size=[3, 3], activation=tf.nn.elu, padding='same'),
                tf.keras.layers.Conv2D(3, kernel_size=1)
            ])

        if self.beta == self.initial_beta:
            beta = self.beta
        else:
            def beta():
                assert self.beta > self.initial_beta and self.beta_anneal_duration > 0
                iteration = tf.cast(tf.train.get_global_step(), tf.float32)
                annealed_beta = self.initial_beta + (self.beta - self.initial_beta) * (iteration - self.beta_anneal_start) / self.beta_anneal_duration
                return tf.clip_by_value(annealed_beta, self.initial_beta, self.beta)

        self.inferencer = IntegratedEagerKlqp(
            self.generative,
            self.variational,
            integrated_name_to_values={},
            conditioning_names=['maybe_camera_matrices', 'gt_pixels_for_regularisation_and_debug'],
            beta=beta,
            verbose=True
        )
        self.optimiser = tf.train.AdamOptimizer(1.e-4)
        tf.train.create_global_step()

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimiser, model=self)

        restore_suffix_and_iteration = hyper('', 'restore', str)  # path@iteration, path is either the part after param-string but before 'checkpoints', i.e. seed and timestamp, or the whole identifier
        if restore_suffix_and_iteration != '':
            restore_suffix, restore_iteration = restore_suffix_and_iteration.split('@')
            if restore_suffix.count('/') > 1:
                self.restored_job_name = os.path.join(output_base_path, restore_suffix)
            else:
                self.restored_job_name = os.path.join(output_base_path, gt.data_string, self.param_string, restore_suffix)
            checkpoint_path = os.path.join(self.restored_job_name, 'checkpoints')
            self.temporary_path_for_deletion = tempfile.mkdtemp()  # this is necesssary as our original paths are often too long for tensorflow to handle
            shortened_path = os.path.join(self.temporary_path_for_deletion, 'ckpt')
            os.symlink(checkpoint_path, shortened_path)
            shortened_path = os.path.join(shortened_path, 'ckpt-{}'.format(restore_iteration))
            print('restoring checkpoint ' + checkpoint_path + ' via symlink ' + shortened_path)
            self.checkpoint.restore(shortened_path)
            self.first_iteration = int(restore_iteration) + 1  # add one as we save the checkpoint after the gradient update for that iteration, so we start with the next one
        else:
            self.temporary_path_for_deletion = None
            self.restored_job_name = None
            self.first_iteration = 0

    def get_room_vertices(self, vertex_offsets, transform_clip_and_hinge):

        vertex_offsets = tf.reshape(vertex_offsets, [episodes_per_batch, self.room_vertical_segments * self.room_circumference_segments, 4])
        room_vertices = self.canonical_vertices * tf.exp(transform_clip_and_hinge(vertex_offsets[..., :1], -2., 2.5))
        room_vertices += tf.tanh(vertex_offsets[..., 1:]) * 1.
        room_vertices = tf.concat([room_vertices, tf.ones_like(room_vertices[..., :1])], axis=-1)  # :: eib, vertex, x/y/z/w

        return room_vertices

    def get_object_vertices(self, vertex_offsets, transform_clip_and_hinge):

        # ** note this is rather similar to get_room_vertices!

        vertex_offsets = tf.reshape(
            vertex_offsets,
            [episodes_per_batch, self.total_grid_cells, self.object_vertical_segments * self.object_circumference_segments, 4]
        )
        object_vertices = self.object_canonical_vertices * tf.exp(transform_clip_and_hinge(vertex_offsets[..., :1], -0.7, 0.7))
        object_vertices += tf.tanh(vertex_offsets[..., 1:]) * 0.2
        object_vertices = tf.concat([object_vertices, tf.ones_like(object_vertices[..., :1])], axis=-1)  # :: eib, obj, vertex, x/y/z/w

        return object_vertices

    def get_l2_laplacian_loss(self, vertices, laplacian):

        delta_coordinates = tf.matmul(tf.tile(laplacian[None, :, :], [int(vertices.shape[0]), 1, 1]), vertices[:, :, :-1])  # indexed by eib, vertex, x/y/z
        delta_norms = tf.norm(delta_coordinates, axis=2)
        return tf.reduce_mean(delta_norms, axis=1)

    def get_creases(self, faces):

        # Result is indexed by crease-index, 1st endpoint / 2nd endpoint / 'left' other-vertex / 'right' other-vertex
        # There is one entry per edge in the mesh; pairs of endpoints correspond to the actual edges; 'left'/'right' vertices are the
        # other vertices that belong to the two face that including the relevant edge
        # Note that this assumes 'simple' topology, with exactly one or two faces touching each edge; edges with only one face
        # touching are not included in the result (as they do not constitute a crease)

        creases = []

        faces = np.asarray(faces)

        for first_face_index in range(len(faces)):
            for second_face_index in range(first_face_index):
                # If any two vertices of 1st face are in 2nd face too, then we've found a crease
                second_as_list = faces[second_face_index].tolist()
                indices_in_second_of_first = [
                    second_as_list.index(first) if first in second_as_list else -1
                    for first in faces[first_face_index]
                ]

                def other_in_second():
                    if 0 not in indices_in_second_of_first:
                        return second_as_list[0]
                    elif 1 not in indices_in_second_of_first:
                        return second_as_list[1]
                    elif 2 not in indices_in_second_of_first:
                        return second_as_list[2]
                    else:
                        assert False

                if indices_in_second_of_first[0] != -1 and indices_in_second_of_first[1] != -1:
                    creases.append((faces[first_face_index, 0], faces[first_face_index, 1], faces[first_face_index, 2], other_in_second()))
                elif indices_in_second_of_first[1] != -1 and indices_in_second_of_first[2] != -1:
                    creases.append((faces[first_face_index, 1], faces[first_face_index, 2], faces[first_face_index, 0], other_in_second()))
                elif indices_in_second_of_first[2] != -1 and indices_in_second_of_first[0] != -1:
                    creases.append((faces[first_face_index, 2], faces[first_face_index, 0], faces[first_face_index, 1], other_in_second()))

        return np.int32(creases)

    def get_crease_loss(self, vertices, creases):

        # vertices is indexed by eib, vertex-index, x/y/z/w
        # creases is indexed by crease-index, 1st-endpoint / 2nd-endpoint / 1st-other / 2nd-other

        # ** this could be converted to a single gather instead of 4*eppb; see get_global_equilaterality_loss!
        first_endpoints, second_endpoints, lefts, rights = tf.map_fn(
            lambda vertices_for_iib: (
                tf.gather(vertices_for_iib, creases[:, 0]),
                tf.gather(vertices_for_iib, creases[:, 1]),
                tf.gather(vertices_for_iib, creases[:, 2]),
                tf.gather(vertices_for_iib, creases[:, 3]),
            ),
            vertices[:, :, :3], (tf.float32, tf.float32, tf.float32, tf.float32)
        )  # each indexed by eib, crease-index, x/y/z

        line_displacements = second_endpoints - first_endpoints
        line_directions = line_displacements / tf.norm(line_displacements, axis=-1, keep_dims=True)
        lefts_projected = first_endpoints + tf.reduce_sum((lefts - first_endpoints) * line_directions, axis=-1, keep_dims=True) * line_directions
        rights_projected = first_endpoints + tf.reduce_sum((rights - first_endpoints) * line_directions, axis=-1, keep_dims=True) * line_directions

        left_directions = lefts - lefts_projected
        left_directions /= tf.norm(left_directions, axis=-1, keep_dims=True)
        right_directions = rights - rights_projected
        right_directions /= tf.norm(right_directions, axis=-1, keep_dims=True)
        cosines = tf.reduce_sum(left_directions * right_directions, axis=-1)
        loss = tf.reduce_mean(tf.sqrt(tf.nn.relu(cosines + 1.)), axis=1)

        return loss

    def get_global_equilaterality_loss(self, vertices):

        edge_endpoints = tf.gather(vertices[:, :, :3], self.creases[:, :2], axis=1)  # eib, edge, start/end, x/y/z
        edge_lengths = tf.linalg.norm(edge_endpoints[:, :, 1] - edge_endpoints[:, :, 0], axis=2)  # eib, edge
        mean_edge_length = tf.reduce_mean(edge_lengths, axis=1, keepdims=True)
        return tf.reduce_mean(tf.square(edge_lengths - mean_edge_length), axis=1)  # i.e. return the variance of edge lengths, for each eib

    def get_edge_matching_loss(self, fg_mask, gt_pixels):

        gaussian_ksize = 5
        gaussian_kernel_1d = tf.tile(
            tf.constant(cv2.getGaussianKernel(ksize=gaussian_ksize, sigma=1., ktype=cv2.CV_32F), dtype=tf.float32)[:, :, None, None],
            [1, 1, 3, 1]
        )

        smoothed_pixels = tf.nn.depthwise_conv2d(
            tf.nn.depthwise_conv2d(
                tf.reshape(gt_pixels, [-1, frame_height, frame_width, 3]),
                gaussian_kernel_1d,
                strides=[1, 1, 1, 1], padding='VALID'
            ),
            tf.transpose(gaussian_kernel_1d, [1, 0, 2, 3]),
            strides=[1, 1, 1, 1], padding='VALID'
        )
        cropped_fg_mask = fg_mask[:, :, gaussian_ksize // 2 : -(gaussian_ksize // 2), gaussian_ksize // 2 : -(gaussian_ksize // 2)]  # this should match the cropping the happens to the image during the above convolution

        derivative_kernel_1d = tf.constant(cv2.getDerivKernels(dx=1, dy=1, ksize=3, normalize=True, ktype=cv2.CV_32F)[0][:, 0], dtype=tf.float32)
        derivative_kernel_x = tf.tile(derivative_kernel_1d[None, :, None, None], [3, 1, 3, 1])
        derivative_kernel_y = tf.tile(derivative_kernel_1d[:, None, None, None], [1, 3, 3, 1])
        derivatives_kernel = tf.concat([derivative_kernel_x, derivative_kernel_y], axis=-1)

        gt_pixel_derivatives = tf.nn.depthwise_conv2d(smoothed_pixels, derivatives_kernel, strides=[1, 1, 1, 1], padding='VALID')
        gt_pixel_derivatives = tf.reshape(
            gt_pixel_derivatives,
            [episodes_per_batch, frames_per_episode] + gt_pixel_derivatives.get_shape()[1:3].as_list() + [3, 2]
        )  # eib, fie, y, x, r/g/b, x-/y-derivative
        fg_mask_derivatives = tf.nn.depthwise_conv2d(
            tf.reshape(cropped_fg_mask, [-1] + cropped_fg_mask.get_shape()[2:].as_list() + [1]),
            derivatives_kernel[:, :, :1, :],
            strides=[1, 1, 1, 1], padding='VALID'
        )
        fg_mask_derivatives = tf.reshape(
            fg_mask_derivatives,
            [episodes_per_batch, frames_per_episode] + fg_mask_derivatives.shape[1:3] + [2]
        )  # eib, fie, y, x, x-/y-derivative

        gt_pixel_derivative_magnitudes = tf.reduce_max(tf.abs(gt_pixel_derivatives), axis=-2)  # max is over colour channels
        fg_mask_derivative_magnitudes = tf.abs(fg_mask_derivatives)

        zeta = hyper(1.e1, 'edge-matching-zeta')
        return tf.reduce_mean(fg_mask_derivative_magnitudes * tf.exp(-zeta * gt_pixel_derivative_magnitudes))

    def get_3d_object_positions(
        self,
        initial_offsets,   # :: eib, grid-z, grid-y, grid-x, x/y/z
        velocities
    ):
        initial_positions = self.grid_centres + initial_offsets

        relative_positions = tf.concat([
            tf.zeros([episodes_per_batch, 1] + self.grid_cells + [3]),
            tf.cumsum(velocities, axis=1)
        ], axis=1)

        return initial_positions[:, None] + relative_positions

    def get_object_appearances(self, object_embeddings):

        object_appearances_flat = self.object_embedding_to_appearance(
            tf.reshape(object_embeddings, [-1, self.object_embedding_channels])
        )
        object_appearances = tf.reshape(
            object_appearances_flat,
            [episodes_per_batch] + self.grid_cells + object_appearances_flat.shape[1:].as_list()
        )

        return object_appearances

    def get_view_matrices_and_maybe_camera_encoding(self, maybe_camera_matrices):

        world_to_view_matrices = tf.linalg.inv(maybe_camera_matrices)

        camera_encoding = self.camera_motion_encoder(tf.reshape(world_to_view_matrices, [episodes_per_batch, frames_per_episode * 16]))

        return world_to_view_matrices, camera_encoding

    def generative(self, rv, mode, add_loss, maybe_camera_matrices, gt_pixels_for_regularisation_and_debug):

        def transform_clip_and_hinge(x, min, max):
            loss_weight = 0.3
            x = (min + max) / 2. + (max - min) * x
            if mode == GenerativeMode.CONDITIONED:
                add_loss(tf.reduce_mean(
                    tf.nn.relu(x - max) + tf.nn.relu(min - x),
                    axis=tuple(range(1, x.get_shape().ndims))
                ) * loss_weight / (max - min))
            return tf.clip_by_value(x, min, max)

        def maybe_add_loss(loss_fn, name, default_strength):
            strength = hyper(default_strength, name)
            if strength != 0.:
                add_loss(loss_fn() * strength)

        scene_embedding = rv.Normal('scene_embedding', tf.zeros([episodes_per_batch, self.scene_embedding_channels]), 1.)

        world_to_view_matrices, camera_encoding = self.get_view_matrices_and_maybe_camera_encoding(maybe_camera_matrices)
        if camera_encoding is not None:
            scene_embedding = tf.concat([scene_embedding, camera_encoding], axis=-1)

        bg_embedding = self.scene_embedding_to_bg_embedding(scene_embedding)

        bg_bottlenecks = self.bg_embedding_to_bottlenecks(bg_embedding)
        bg_shape_bottleneck = bg_bottlenecks[:, :self.bg_shape_bottleneck_channels]
        bg_texture_bottleneck = bg_bottlenecks[:, self.bg_shape_bottleneck_channels:]
        vertex_offsets = self.bg_bottleneck_to_offsets(bg_shape_bottleneck) * 0.1
        assert vertex_offsets.shape[1:3] == [self.room_vertical_segments, self.room_circumference_segments]
        bg_texture = transform_clip_and_hinge(self.bg_bottleneck_to_texture(bg_texture_bottleneck) * 0.25, 0., 1.)

        room_vertices = self.get_room_vertices(vertex_offsets, transform_clip_and_hinge)
        room_vertices_view = tf.einsum('evi,efij->efvj', room_vertices, world_to_view_matrices)
        projected_vertices = tf.einsum('efvj,jk->efvk', room_vertices_view, self.projection_matrix)

        if mode == GenerativeMode.CONDITIONED:
            maybe_add_loss(lambda: self.get_l2_laplacian_loss(room_vertices, self.laplacian), 'l2-lapl', 0.)
            maybe_add_loss(lambda: self.get_crease_loss(room_vertices, self.creases), 'crease', 5.e1)
            maybe_add_loss(lambda: self.get_global_equilaterality_loss(room_vertices), 'glob-equil', 1.e1)

        def do_shading(mask_and_uvs_and_depth, texture):

            mask = mask_and_uvs_and_depth[..., 0]
            uvs = mask_and_uvs_and_depth[..., 1:3]
            depth = mask_and_uvs_and_depth[..., 3:]
            uvs = tf.reshape(uvs, [episodes_per_batch, frames_per_episode, frame_height, frame_width, 2])

            texture_samples = tf.reshape(
                sample_texture(texture, uvs),
                [episodes_per_batch * frames_per_episode, frame_height, frame_width, 3]
            )

            pixels = texture_samples * mask[..., None]

            return tf.concat([pixels, depth], axis=-1)

        vertex_count = self.canonical_vertices.shape[0]
        projected_vertices_flat = tf.reshape(projected_vertices, [-1] + projected_vertices.shape[2:].as_list())
        far_depth_value = 2.e1  # this is the 'background' value for the depth-map; typically all pixels are covered by a wall, so it rarely appears
        background_pixels_and_depths = tf.reshape(
            dirt.rasterise_batch_deferred(
                background_attributes=tf.zeros([episodes_per_batch * frames_per_episode, frame_height, frame_width, 4]) + [0., 0., 0., far_depth_value],
                vertices=projected_vertices_flat,
                vertex_attributes=tf.concat([
                    tf.ones([episodes_per_batch * frames_per_episode, vertex_count, 1]),
                    tf.tile(self.canonical_uvs[None], [episodes_per_batch * frames_per_episode, 1, 1]),
                    projected_vertices_flat[..., 3:]
                ], axis=-1),
                faces=tf.tile(self.canonical_faces[None], [episodes_per_batch * frames_per_episode, 1, 1]),
                shader_fn=do_shading,
                shader_additional_inputs=[bg_texture]
            ),
            [episodes_per_batch, frames_per_episode, frame_height, frame_width, 4]
        )
        background_pixels = background_pixels_and_depths[..., :3]
        background_depths = background_pixels_and_depths[..., 3]

        (
            object_embeddings,
            object_initial_offsets,
            object_xz_velocities,
            object_speed_factors,
            object_angular_velocities,
            object_presence_logits,
            object_initial_azimuths
        ) = tf.split(
            tf.reshape(
                self.scene_embedding_to_object_params(scene_embedding),
                [episodes_per_batch] + self.grid_cells + [-1]
            ),
            [self.object_embedding_channels, 3, 2, frames_per_episode - 1, frames_per_episode - 1, 1, 1],
            axis=-1
        )

        object_presences = tf.nn.sigmoid(tf.squeeze(object_presence_logits, axis=-1) + hyper(-0., 'obj-pres-bias'))  # :: eib, grid-z, grid-y, grid-x
        if mode == GenerativeMode.CONDITIONED:
            maybe_add_loss(lambda: tf.reduce_mean(tf.nn.relu(hyper(0., 'obj-min-pres') - object_presences)), 'obj-min-pres-reg', 0.)
            maybe_add_loss(lambda: tf.reduce_mean(object_presences), 'obj-pres-reg', 0.)
        object_presences *= self.grid_mask

        object_initial_offsets = transform_clip_and_hinge(object_initial_offsets * [0.5, 0., 0.5], -1., 1.) * 0.5 * tf.abs(self.grid_centres[1, 0, 1] - self.grid_centres[0, 0, 0])

        object_xz_velocities *= hyper(1.5, 'obj-linear-vel-scale')
        object_velocities = tf.stack([object_xz_velocities[..., 0], tf.zeros_like(object_xz_velocities[..., 0]), object_xz_velocities[..., 1]], axis=-1)

        object_speed_factors = tf.exp(tf.transpose(object_speed_factors, [0, 4, 1, 2, 3]) * hyper(0.25, 'obj-speed-variation'))
        object_speed_factors = transform_clip_and_hinge(object_speed_factors, 0.5, 2.)
        object_velocities = object_velocities[:, None] * object_speed_factors[..., None]
        if mode == GenerativeMode.CONDITIONED:
            maybe_add_loss(lambda: tf.reduce_mean(tf.linalg.norm(object_initial_offsets, axis=-1)), 'obj-offset-reg', 0.)
            maybe_add_loss(lambda: tf.reduce_mean(tf.linalg.norm(object_velocities, axis=-1)), 'obj-vel-reg', 1.e0)
        object_velocities -= [1., 0., 0.]

        object_initial_azimuths *= hyper(0.1, 'obj-azimuth-scale')
        object_angular_velocities = tf.transpose(object_angular_velocities, [0, 4, 1, 2, 3])
        object_angular_velocities *= hyper(0., 'obj-angular-vel-scale')
        object_azimuths = tf.squeeze(object_initial_azimuths, axis=-1)[:, None] + tf.concat([
            tf.zeros([episodes_per_batch, 1] + self.grid_cells),
            tf.cumsum(object_angular_velocities, axis=1)
        ], axis=1)

        object_sizes = tf.ones([episodes_per_batch] + self.grid_cells + [3]) * [4.5, 2.2, 4.5]

        object_positions_world = self.get_3d_object_positions(object_initial_offsets, object_velocities)

        if self.object_representation == 'voxels':

            object_appearances = transform_clip_and_hinge(self.get_object_appearances(object_embeddings), 0., 1.)

            object_appearances_flat = tf.reshape(object_appearances, [episodes_per_batch, -1] + object_appearances.get_shape()[-4:].as_list())
            pixel_means, fg_mask = render_3d_objects_over_background(
                object_appearances_flat[..., :3], object_appearances_flat[..., 3],
                tf.reshape(object_positions_world, [episodes_per_batch, frames_per_episode, -1, 3]),
                tf.reshape(object_azimuths, [episodes_per_batch, frames_per_episode, -1]),
                tf.reshape(object_sizes, [episodes_per_batch, -1, 3]),
                tf.reshape(object_presences, [episodes_per_batch, -1]),
                background_pixels, background_depths,
                world_to_view_matrices, self.projection_matrix
            )

            if mode == GenerativeMode.CONDITIONED:
                maybe_add_loss(lambda: self.get_edge_matching_loss(fg_mask, gt_pixels_for_regularisation_and_debug), 'edge-matching-reg', 1.e1)
                maybe_add_loss(lambda: tf.reduce_mean(fg_mask), 'obj-coverage-reg', 0.)

            if mode != GenerativeMode.CONDITIONED:

                objects, _ = render_3d_objects_over_background(
                    object_appearances_flat[..., :3], object_appearances_flat[..., 3],
                    tf.reshape(object_positions_world, [episodes_per_batch, frames_per_episode, -1, 3]),
                    tf.reshape(object_azimuths, [episodes_per_batch, frames_per_episode, -1]),
                    tf.reshape(object_sizes, [episodes_per_batch, -1, 3]),
                    tf.reshape(object_presences, [episodes_per_batch, -1]),
                    utils.make_chequerboard(frame_width, frame_height, batch_dims=[episodes_per_batch, frames_per_episode], spacing=10),
                    background_depths,
                    world_to_view_matrices, self.projection_matrix
                )

                if True:  # enable this to render black, presence-weighted boxes around each object

                    far_depth_value = 2.e1
                    object_depths, object_rotation_matrices = get_3d_object_depthmaps(
                        tf.reshape(object_positions_world, [episodes_per_batch, frames_per_episode, -1, 3]),
                        tf.reshape(object_azimuths, [episodes_per_batch, frames_per_episode, -1]),
                        tf.reshape(object_sizes, [episodes_per_batch, -1, 3]),
                        world_to_view_matrices, self.projection_matrix,
                        far_depth_value
                    )  # :: eib, fie, obj, y, x

                    object_masks = tf.cast(tf.not_equal(object_depths, far_depth_value), tf.float32)  # :: eib, fie, obj, y, x

                    object_outlines = tf.reshape(
                        tf.nn.max_pool(tf.reshape(object_masks, [-1, frame_height, frame_width, 1]), [1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME'),
                        [episodes_per_batch, frames_per_episode, -1, frame_height, frame_width, 1]
                     ) - object_masks[..., None]  # :: eib * fie * obj, y, x, singleton
                    object_outlines = tf.reshape(object_outlines, [episodes_per_batch, frames_per_episode, -1, frame_height, frame_width])

                    object_outlines *= tf.reshape(object_presences, [episodes_per_batch, 1, self.total_grid_cells, 1, 1])
                    object_outlines *= tf.reduce_max(object_appearances_flat[..., 3], axis=[2, 3, 4])[:, None, :, None, None]
                    object_outlines = tf.reduce_max(object_outlines, axis=2)
                    objects = objects * (1. - object_outlines[..., None])

                object_masks_by_depth = render_3d_objects_over_background(
                    tf.ones_like(object_appearances_flat[..., :3]), object_appearances_flat[..., 3],
                    tf.reshape(object_positions_world, [episodes_per_batch, frames_per_episode, -1, 3]),
                    tf.reshape(object_azimuths, [episodes_per_batch, frames_per_episode, -1]),
                    tf.reshape(object_sizes, [episodes_per_batch, -1, 3]),
                    tf.reshape(object_presences, [episodes_per_batch, -1]),
                    tf.zeros_like(background_pixels), background_depths,
                    world_to_view_matrices, self.projection_matrix,
                    composed_output=False
                )[..., 0]

                # ** note this is (roughly) duplicated from render_3d_objects_over_background!
                voxel_to_view_matrices = dirt.matrices.compose(
                    dirt.matrices.scale(tf.reshape(object_sizes, [episodes_per_batch, 1, -1, 3]) * [1., -1., 1.] / 2.),
                    dirt.matrices.pad_3x3_to_4x4(object_rotation_matrices),
                    dirt.matrices.translation(tf.reshape(object_positions_world, [episodes_per_batch, frames_per_episode, -1, 3])),
                    world_to_view_matrices[:, :, None]
                )  # eib, fie, obj, x/y/z/w (in), x/y/z/w (out)
                voxel_coordinates = tf.stack(tf.meshgrid(
                    tf.linspace(-1., 1., object_appearances.shape[4]),  # z
                    tf.linspace(-1., 1., object_appearances.shape[5]),  # y
                    tf.linspace(-1., 1., object_appearances.shape[6]),  # x
                    indexing='ij'
                ), axis=-1)[..., ::-1]  # z, y, x, x/y/z

                voxel_coordinate_map, _ = render_3d_objects_over_background(
                    tf.tile(voxel_coordinates[None, None], [episodes_per_batch, self.total_grid_cells, 1, 1, 1, 1]),
                    object_appearances_flat[..., 3],
                    tf.reshape(object_positions_world, [episodes_per_batch, frames_per_episode, -1, 3]),
                    tf.reshape(object_azimuths, [episodes_per_batch, frames_per_episode, -1]),
                    tf.reshape(object_sizes, [episodes_per_batch, -1, 3]),
                    tf.reshape(object_presences, [episodes_per_batch, -1]),
                    tf.tile(background_depths[..., None], [1, 1, 1, 1, 3]), background_depths,
                    world_to_view_matrices, self.projection_matrix,
                    object_colour_transforms=tf.matmul(
                        voxel_to_view_matrices,
                        tf.diag([1., 1., -1., 1.])  # flip z, so we get +ve depth rather than -ve view-space z-coordinate
                    )
                )
                depth = voxel_coordinate_map[..., 2]

                threshold_to_object_bboxes = {
                    threshold: get_voxel_bboxes_at_threshold(
                        threshold,
                        object_appearances_flat[..., 3],
                        tf.reshape(object_presences, [episodes_per_batch, -1]),
                        voxel_to_view_matrices,
                        voxel_coordinates
                    )
                    for threshold in [0.05, 0.1, 0.2, 0.4]
                }

            else:
                depth = objects = object_masks_by_depth = threshold_to_object_bboxes = None

        elif self.object_representation == 'mesh':

            object_embeddings_flat = tf.reshape(object_embeddings, [episodes_per_batch * self.total_grid_cells, self.object_embedding_channels])
            object_vertex_offsets = tf.reshape(
                self.object_embedding_to_offsets(object_embeddings_flat),
                [episodes_per_batch, self.total_grid_cells, -1, 4]
            )
            object_textures = self.object_embedding_to_texture(object_embeddings_flat)
            object_textures = transform_clip_and_hinge(
                tf.reshape(object_textures, [episodes_per_batch, self.total_grid_cells] + object_textures.shape[1:].as_list()),
                0., 1.
            )
            object_vertices = self.get_object_vertices(object_vertex_offsets, transform_clip_and_hinge)

            if mode == GenerativeMode.CONDITIONED:
                maybe_add_loss(lambda: tf.reduce_mean(tf.reshape(
                    self.get_l2_laplacian_loss(tf.reshape(object_vertices, [-1] + object_vertices.get_shape()[2:].as_list()), self.object_laplacian),
                    [episodes_per_batch, self.total_grid_cells]
                ), axis=1), 'obj-l2-lapl', 0.)
                maybe_add_loss(lambda: tf.reduce_mean(tf.reshape(
                    self.get_crease_loss(tf.reshape(object_vertices, [-1] + object_vertices.get_shape()[2:].as_list()), self.object_creases),
                    [episodes_per_batch, self.total_grid_cells]
                ), axis=1), 'obj-crease', 0.)

            pixel_means = render_meshes_over_background(
                object_vertices, object_textures, self.object_faces, self.object_uvs,
                tf.reshape(object_positions_world, [episodes_per_batch, frames_per_episode, -1, 3]),
                tf.reshape(object_azimuths, [episodes_per_batch, frames_per_episode, -1]),
                tf.reshape(object_sizes, [episodes_per_batch, -1, 3]),
                tf.reshape(object_presences, [episodes_per_batch, -1]),
                background_pixels, background_depths,
                world_to_view_matrices, self.projection_matrix
            )

            if mode != GenerativeMode.CONDITIONED:

                objects = render_meshes_over_background(
                    object_vertices, object_textures, self.object_faces, self.object_uvs,
                    tf.reshape(object_positions_world, [episodes_per_batch, frames_per_episode, -1, 3]),
                    tf.reshape(object_azimuths, [episodes_per_batch, frames_per_episode, -1]),
                    tf.reshape(object_sizes, [episodes_per_batch, -1, 3]),
                    tf.reshape(object_presences, [episodes_per_batch, -1]),
                    utils.make_chequerboard(frame_width, frame_height, batch_dims=[episodes_per_batch, frames_per_episode], spacing=10),
                    background_depths,
                    world_to_view_matrices, self.projection_matrix
                )

                depth = render_meshes_over_background(
                    object_vertices, 'depth', self.object_faces, self.object_uvs,
                    tf.reshape(object_positions_world, [episodes_per_batch, frames_per_episode, -1, 3]),
                    tf.reshape(object_azimuths, [episodes_per_batch, frames_per_episode, -1]),
                    tf.reshape(object_sizes, [episodes_per_batch, -1, 3]),
                    tf.reshape(object_presences, [episodes_per_batch, -1]),
                    tf.tile(background_depths[..., None], [1, 1, 1, 1, 3]), background_depths,
                    world_to_view_matrices, self.projection_matrix
                )[..., 0]

                object_masks_by_depth = render_meshes_over_background(
                    object_vertices, tf.ones_like(object_textures), self.object_faces, self.object_uvs,
                    tf.reshape(object_positions_world, [episodes_per_batch, frames_per_episode, -1, 3]),
                    tf.reshape(object_azimuths, [episodes_per_batch, frames_per_episode, -1]),
                    tf.reshape(object_sizes, [episodes_per_batch, -1, 3]),
                    tf.reshape(object_presences, [episodes_per_batch, -1]),
                    tf.zeros_like(background_pixels), background_depths,
                    world_to_view_matrices, self.projection_matrix,
                    composed_output=False
                )[..., 0]

                threshold_to_object_bboxes = {0: get_mesh_bboxes(
                    object_vertices,
                    tf.reshape(object_positions_world, [episodes_per_batch, frames_per_episode, -1, 3]),
                    tf.reshape(object_azimuths, [episodes_per_batch, frames_per_episode, -1]),
                    tf.reshape(object_sizes, [episodes_per_batch, -1, 3]),
                    tf.reshape(object_presences, [episodes_per_batch, -1]),
                    world_to_view_matrices
                )}

            else:
                depth = objects = object_masks_by_depth = threshold_to_object_bboxes = None

        if mode != GenerativeMode.CONDITIONED:

            room_vertices_split, canonical_faces_split = dirt.lighting.split_vertices_by_face(room_vertices, self.canonical_faces)
            room_vertices_view_split = tf.einsum('evi,efij->efvj', room_vertices_split, world_to_view_matrices)
            projected_vertices_split = tf.einsum('efij,jk->efik', room_vertices_view_split, self.projection_matrix)
            vertex_normals_split = tf.tile(dirt.lighting.vertex_normals_pre_split(room_vertices_split, canonical_faces_split, static=True)[:, None], [1, frames_per_episode, 1, 1])
            normals = tf.reshape(
                dirt.rasterise_batch(
                    background=tf.zeros([episodes_per_batch * frames_per_episode, frame_height, frame_width, 3]),
                    vertices=tf.reshape(projected_vertices_split, [-1] + projected_vertices_split.shape[2:].as_list()),
                    vertex_colors=tf.reshape(vertex_normals_split, [episodes_per_batch * frames_per_episode, vertex_normals_split.shape[2], 3]),
                    faces=tf.tile(canonical_faces_split[None], [episodes_per_batch * frames_per_episode, 1, 1])
                ),
                [episodes_per_batch, frames_per_episode, frame_height, frame_width, 3]
            )

        else:
            normals = None

        rv.NormalPyramid('pixels', pixel_means, 0.01, levels=hyper(4, 'pyr-levels', int))
        return pixel_means, depth, normals, background_pixels, objects, object_masks_by_depth, room_vertices, threshold_to_object_bboxes

    def variational(self, rv, add_loss, pixels, maybe_camera_matrices, gt_pixels_for_regularisation_and_debug):

        pixel_encoding = self.temporal_encoder(pixels)
        _, camera_encoding = self.get_view_matrices_and_maybe_camera_encoding(maybe_camera_matrices)

        if camera_encoding is not None:
            scene_embedding_mean_and_sigma_logit = self.encoding_to_scene_embedding(tf.concat([pixel_encoding, camera_encoding], axis=-1))
        else:
            scene_embedding_mean_and_sigma_logit = self.encoding_to_scene_embedding(pixel_encoding)

        scene_embedding_mean = scene_embedding_mean_and_sigma_logit[:, :self.scene_embedding_channels]
        scene_embedding_sigma_logit = scene_embedding_mean_and_sigma_logit[:, self.scene_embedding_channels:]
        rv.Normal('scene_embedding', scene_embedding_mean, tf.nn.softplus(scene_embedding_sigma_logit) + 1.e-6)

    def train(self, gt, output_path, visualise_frequency):

        def save_checkpoint():
            checkpoint_path = self.checkpoint.write(file_prefix=output_path + '/checkpoints/ckpt-{}'.format(iteration))
            print('checkpoint saved to ' + checkpoint_path)

        iteration = None
        try:

            @tf.function(autograph=False)
            def get_loss_and_gradients(gt_pixels, gt_camera_matrices):

                with tf.GradientTape() as tape:
                    loss = self.inferencer.train(pixels=gt_pixels, check_prior=False, maybe_camera_matrices=gt_camera_matrices, gt_pixels_for_regularisation_and_debug=gt_pixels)

                grads = tape.gradient(loss, self.trainable_weights)
                grads = [tf.clip_by_value(grad, -10., 10.) for grad in grads if grad is not None]

                return loss, grads

            @tf.function(autograph=False)
            def visualise(iteration, gt_pixels, gt_camera_matrices, gt_depth, gt_masks):

                recon_pixels, recon_depth, recon_normals, recon_background, recon_objects, recon_masks_by_depth, recon_room_vertices, _ = self.inferencer.reconstruct(pixels=gt_pixels, maybe_camera_matrices=gt_camera_matrices, gt_pixels_for_regularisation_and_debug=gt_pixels)
                gen_pixels, gen_depth, _, gen_background, gen_objects, gen_masks_by_depth, _, _ = self.inferencer.sample_prior(maybe_camera_matrices=gt_camera_matrices, gt_pixels_for_regularisation_and_debug=gt_pixels)

                self.visualise_batch(
                    iteration, output_path,
                    gt, gt_depth, gt_masks, gt_pixels,
                    recon_background, recon_depth, recon_normals, recon_objects, recon_masks_by_depth, recon_pixels, recon_room_vertices,
                    gen_background, gen_depth, gen_objects, gen_masks_by_depth, gen_pixels
                )

                if tf.executing_eagerly():

                    for episode_index in range(1):  # ...just one episode as write_obj is rather slow!
                        utils.write_obj('{}/objs/{:06}_{:02}.obj'.format(output_path, iteration, episode_index),
                            recon_room_vertices[episode_index], self.canonical_faces
                        )

            for iteration, (gt_pixels, gt_camera_matrices, gt_depth, gt_normals, gt_masks, gt_bboxes) in enumerate(gt.batches('train'), start=self.first_iteration):

                if iteration == 400000:
                    break

                total_loss = 0.
                total_grads = None
                for subbatch_index in range(subbatch_count):
                    loss, grads = get_loss_and_gradients(gt_pixels[subbatch_index], gt_camera_matrices[subbatch_index])
                    total_loss += loss / subbatch_count
                    if total_grads is None:
                        total_grads = [grad / subbatch_count for grad in grads]
                    else:
                        total_grads = [total_grad + grad / subbatch_count for grad, total_grad in zip(grads, total_grads)]
                self.optimiser.apply_gradients(zip(total_grads, self.trainable_weights), tf.train.get_global_step())

                if tf.is_nan(total_loss):
                    print('aborting due to NaN loss')
                    visualise(tf.constant(iteration), gt_pixels[0], gt_camera_matrices[0], gt_depth[0], gt_masks[0])
                    break

                if iteration == self.first_iteration + 1:  # i.e. wait till the generative and variational have been run in full
                    hyper.verify_args()
                    if self.temporary_path_for_deletion is not None:
                        shutil.rmtree(self.temporary_path_for_deletion)
                        self.temporary_path_for_deletion = None

                if iteration % visualise_frequency == 0 or iteration == self.first_iteration:
                    visualise(tf.constant(iteration), gt_pixels[0], gt_camera_matrices[0], gt_depth[0], gt_masks[0])

                if iteration > 0 and iteration % 5000 == 0:
                    save_checkpoint()

                if iteration > 0 and iteration % 50000 == 0:
                    print('evaluating at iteration {}...'.format(iteration))
                    self.test_on_split(gt, output_path, split='val', output_suffix=f'{iteration:06}')

        except KeyboardInterrupt:
            pass

        if iteration is not None and iteration > 100:
            save_checkpoint()

    def test_on_split(self, gt, output_path, output_suffix, split='val'):

        class PixelAccuracy:

            def __init__(self):
                self._mean_sq_errors = []  # each entry is a scalar mean over all pixels in the batch
                self._mean_abs_errors = []

            def record_batch(self, gt_pixels, recon_pixels):
                self._mean_sq_errors.append(tf.reduce_mean(tf.square(gt_pixels - recon_pixels).numpy()))
                self._mean_abs_errors.append(tf.reduce_mean(tf.abs(gt_pixels - recon_pixels).numpy()))

            PixelStatistics = namedtuple('PixelStatistics', ['mean_sq_error', 'mean_abs_error'])

            def get_statistics(self):
                return self.PixelStatistics(
                    mean_sq_error=np.mean(self._mean_sq_errors),
                    mean_abs_error=np.mean(self._mean_abs_errors)
                )

        class SegmentationAccuracy:

            # This implements standard fg/bg segmentation metrics -- mean (per-frame) foreground IOU, and pixel
            # classification accuracy -- and the segmentation covering (SC) and mean segmentation covering (mSC) metrics of
            # Engelcke, "GENESIS: Generative Scene Inference and Sampling with Object-Centric Latent Representations"
            # We also include 'tracking aware' equivalents that evaluate the IOU per-episode rather than per-frame -- possible
            # as both gt and recon have persistent object identities across frames
            # Each metric is calculated for several different alpha*presence binarisation thresholds

            SegmentationStatisticsForThreshold = namedtuple('SegmentationStatisticsForThreshold', [
                'fg_iou', 'pixel_accuracy',
                'framewise_weighted_segmentation_covering', 'framewise_mean_segmentation_covering',
                'tracked_weighted_segmentation_covering', 'tracked_mean_segmentation_covering'
            ])

            def __init__(self, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
                self._threshold_to_statistics = {threshold: self.SegmentationStatisticsForThreshold([], [], [], [], [], []) for threshold in thresholds}

            def record_batch(self, gt_masks, recon_masks_by_depth):

                # gt_masks is of type uint8; each non-black colour represents a distinct object
                # recon_masks_by_depth is indexed by object, ordered from farthest to nearest

                ordered_thresholds = sorted(self._threshold_to_statistics.keys())
                thresholds_tensor = tf.constant(ordered_thresholds)

                recon_masks_binarised_by_threshold = binarise_and_modalise_masks(recon_masks_by_depth, thresholds_tensor)  # threshold, obj, eib, fie, y, x

                gt_masks = tf.cast(gt_masks, tf.int32)
                gt_masks_int = gt_masks[..., 0] + gt_masks[..., 1] * 256 + gt_masks[..., 2] * 256 * 256

                self.record_segmentation_covering(gt_masks_int, recon_masks_binarised_by_threshold, ordered_thresholds)
                self.record_fg_bg_accuracy(gt_masks_int, recon_masks_binarised_by_threshold, ordered_thresholds)

            @staticmethod
            def iou(a, b):
                # a and b are assumed to be binary, indexed by *, y, x
                return tf.reduce_sum(a * b, axis=[-2, -1]) / (tf.reduce_sum(tf.maximum(a, b), axis=[-2, -1]) + 1.e-12)

            def record_segmentation_covering(self, gt_masks_int, recon_masks_binarised_by_threshold, ordered_thresholds):

                for episode_index in range(episodes_per_batch):

                    gt_object_colours, gt_object_indices_flat = tf.unique(tf.reshape(gt_masks_int[episode_index], [-1]))
                    gt_object_indices = tf.reshape(gt_object_indices_flat, [frames_per_episode, frame_height, frame_width])
                    gt_masks_binarised = tf.transpose(
                        tf.one_hot(gt_object_indices, depth=gt_object_colours.shape[0]),  # fie, y, x, obj
                        [3, 0, 1, 2]
                    )  # obj, fie, y, x
                    gt_masks_binarised = tf.boolean_mask(gt_masks_binarised, tf.not_equal(gt_object_colours, 0))  # i.e. remove the 'object' corresponding to the black background

                    if len(gt_masks_binarised) == 0:
                        continue  # or, could record as perfect coverage, which it is...

                    def do_framewise_or_tracked(gt_masks_binarised, recon_masks_binarised_by_threshold):

                        # The parameters to this are indexed by ...fie, y, x; a mean over the fie axis is taken at the end, while
                        # the y & x axes are reduced over in the IoU calculation
                        # With 'natural' input, this yields the framewise statistics; if run with a singleton frame dimension
                        # and time expanded into the y-axis, it computes the tracked version, where entire episode is treated
                        # like a single image

                        ious = self.iou(
                            gt_masks_binarised[None, :, None],
                            recon_masks_binarised_by_threshold[:, None]
                        )  # threshold, gt-obj, recon-obj, fie-or-singleton
                        max_iou_per_gt = tf.reduce_max(ious, axis=2)  # threshold, gt-obj, fie-or-singleton
                        gt_areas = tf.reduce_sum(gt_masks_binarised, axis=[2, 3])  # gt-obj, fie-or-singleton
                        gt_present = tf.cast(tf.greater(gt_areas, 0), tf.float32)

                        mean_segmentation_covering = tf.reduce_mean(
                            tf.reduce_sum(max_iou_per_gt, axis=1) / (tf.reduce_sum(gt_present, axis=0) + 1.e-12),  # sums are over gt-obj
                            axis=1  # mean is over frames (unless tracked mode)
                        )  # :: threshold
                        weighted_segmentation_covering = tf.reduce_mean(
                            tf.reduce_sum(max_iou_per_gt * gt_areas, axis=1) / (tf.reduce_sum(gt_areas, axis=0) + 1.e-12),  # sums are over gt-obj
                            axis=1  # mean is over frames (unless tracked mode)
                        )

                        return mean_segmentation_covering, weighted_segmentation_covering

                    framewise_mean_segmentation_covering, framewise_weighted_segmentation_covering = do_framewise_or_tracked(gt_masks_binarised, recon_masks_binarised_by_threshold[:, :, episode_index])
                    tracked_mean_segmentation_covering, tracked_weighted_segmentation_covering = do_framewise_or_tracked(
                        tf.reshape(gt_masks_binarised, [-1, 1, frames_per_episode * frame_height, frame_width]),
                        tf.reshape(recon_masks_binarised_by_threshold[:, :, episode_index], [len(ordered_thresholds), -1, 1, frames_per_episode * frame_height, frame_width])
                    )

                    for threshold_index, threshold in enumerate(ordered_thresholds):
                        statistics = self._threshold_to_statistics[threshold]
                        statistics.framewise_mean_segmentation_covering.append(float(framewise_mean_segmentation_covering[threshold_index]))
                        statistics.framewise_weighted_segmentation_covering.append(float(framewise_weighted_segmentation_covering[threshold_index]))
                        statistics.tracked_mean_segmentation_covering.append(float(tracked_mean_segmentation_covering[threshold_index]))
                        statistics.tracked_weighted_segmentation_covering.append(float(tracked_weighted_segmentation_covering[threshold_index]))

            def record_fg_bg_accuracy(self, gt_masks_int, recon_masks_binarised_by_threshold, ordered_thresholds):

                recon_fg_mask_binarised_by_threshold = tf.reduce_max(recon_masks_binarised_by_threshold, axis=1)  # threshold, eib, fie, y, x

                gt_cars_mask = tf.cast(tf.not_equal(gt_masks_int, 0), tf.float32)  # eib, fie, y, x

                ious = self.iou(gt_cars_mask[None], recon_fg_mask_binarised_by_threshold)  # threshold, eib, fie
                accuracies = tf.reduce_mean(tf.cast(tf.equal(gt_cars_mask, recon_fg_mask_binarised_by_threshold), tf.float32), axis=[-2, -1])  # threshold, eib, fie

                ious = tf.reduce_mean(ious, axis=[1, 2])  # threshold
                accuracies = tf.reduce_mean(accuracies, axis=[1, 2])

                for threshold_index, threshold in enumerate(ordered_thresholds):
                    statistics = self._threshold_to_statistics[threshold]
                    statistics.fg_iou.append(float(ious[threshold_index]))
                    statistics.pixel_accuracy.append(float(accuracies[threshold_index]))

            def get_statistics(self):
                return {
                    threshold: self.SegmentationStatisticsForThreshold(
                        fg_iou=np.mean(statistics.fg_iou),
                        pixel_accuracy=np.mean(statistics.pixel_accuracy),
                        framewise_weighted_segmentation_covering=np.mean(statistics.framewise_weighted_segmentation_covering),
                        framewise_mean_segmentation_covering=np.mean(statistics.framewise_mean_segmentation_covering),
                        tracked_weighted_segmentation_covering=np.mean(statistics.tracked_weighted_segmentation_covering),
                        tracked_mean_segmentation_covering=np.mean(statistics.tracked_mean_segmentation_covering)
                    )
                    for threshold, statistics in self._threshold_to_statistics.items()
                }

        class DepthAccuracy:

            # This follows Eigen, "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"; the
            # same metrics are used in Eigen, "Predicting Depth, Surface Normals and Semantic Labels
            # with a Common Multi-Scale Convolutional Architecture", Garg, "Unsupervised CNN for Single View
            # Depth Estimation: Geometry to the Rescue", and Zhan, "Self-supervised Learning for Single View Depth
            # and Surface Normal Estimation"
            # We use the entire image, i.e. both objects and background
            # Given the gt and recon linear depths in world-space, we evaluate:
            # - mean absolute relative error (gt in the denominator)
            # - mean squared relative error (non-squared gt in the denominator)
            # - RMS error
            # - RMS difference of logs
            # - fraction of pixels s.t. max(gt/recon, recon/gt) < {1.25, 1.25**2, 1.25**3}

            def __init__(self):
                self._gt_by_batch = []  # each entry is a flat array, as we have different numbers of non-sky pixels per image
                self._recon_by_batch = []

            def record_batch(self, gt_depths, recon_depths):
                non_sky_mask = tf.logical_and(tf.not_equal(gt_depths, 1.e3), tf.not_equal(recon_depths, 0.))
                non_sky_gt_depths = tf.boolean_mask(gt_depths, non_sky_mask)
                non_sky_recon_depths = tf.boolean_mask(recon_depths, non_sky_mask)
                self._gt_by_batch.append(non_sky_gt_depths.numpy())
                self._recon_by_batch.append(non_sky_recon_depths.numpy())

            DepthStatistics = namedtuple('DepthStatistics', ['mean_rel_error', 'mean_sq_rel_error', 'rms_error', 'rms_diff_logs', 'threshold_to_fraction_delta_less'])

            def get_statistics(self):
                all_gt = np.concatenate(self._gt_by_batch, axis=0)
                all_recon = np.concatenate(self._recon_by_batch, axis=0)
                abs_errors = np.abs(all_recon - all_gt)
                sq_errors = np.square(abs_errors)
                deltas = np.maximum(all_recon / all_gt, all_gt / all_recon)
                return self.DepthStatistics(
                    mean_rel_error=np.mean(abs_errors / all_gt),
                    mean_sq_rel_error=np.mean(sq_errors / all_gt),
                    rms_error=np.sqrt(np.mean(sq_errors)),
                    rms_diff_logs=np.sqrt(np.mean(np.square(np.log(all_recon) - np.log(all_gt)))),
                    threshold_to_fraction_delta_less={
                        threshold: np.mean(deltas < threshold)
                        for threshold in {1.25, 1.25 ** 2, 1.25 ** 3}
                    }
                )

        class DetectionAccuracy:

            # This measures the 3D bbox detection accuracy, i.e. AP at some overlap
            # threshold, allowing each gt box to be matched at most once. We use
            # axis-aligned bounding-boxes in view space (hence slightly different to KITTI).
            # Ordering / scores are defined by mean(presence * alpha); the bbox itself is
            # defined by thresholding presence * alpha

            DetectionsForThreshold = namedtuple('DetectionsForThreshold', [
                'true_positives', 'false_positives', 'scores'  # each is a list of floats of equal length; true_positive and false_positive are always in {0, 1}
            ])

            def __init__(self):
                # alpha-thresholds are applied to voxel alphas to determine bounds; they are
                # unrelated to the IOU threshold in the AP calculation
                self._alpha_threshold_to_detections = defaultdict(lambda: self.DetectionsForThreshold([], [], []))
                self._gt_true_count = 0
                self._iou_threshold = 0.3  # threshold for overlap of 3D bboxes
                self._mask_visibility_count_threshold = 40  # for a gt/recon object to count as visible, this many pixels of its bbox-silhouette must be visible

            @staticmethod
            def _render_bboxes(bboxes, projection_matrix):

                # bboxes :: im, obj, min/max, x/y/z; coordinates are in view-space
                # projection_matrix :: x/y/z/w (in), x/y/z/w (out)
                # result :: im, obj, y, x

                bbox_vertices_view = tf.stack([
                    tf.stack([
                        bboxes[:, :, x_min_or_max, 0],
                        bboxes[:, :, y_min_or_max, 1],
                        bboxes[:, :, z_min_or_max, 2]
                    ], axis=-1)
                    for z_min_or_max in [0, 1] for y_min_or_max in [0, 1] for x_min_or_max in [0, 1]
                ], axis=2)  # im, obj, vertex, x/y/z

                bbox_vertices_clip = tf.matmul(
                    utils.to_homogeneous(tf.reshape(bbox_vertices_view, [-1, 8, 3])),
                    projection_matrix
                )  # im * obj, vertex, x/y/z/w

                bbox_quads = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5], [0, 4, 5, 1], [2, 6, 7, 3]]
                bbox_faces = tf.constant(sum([[[a, b, c], [c, d, a]] for [a, b, c, d] in bbox_quads], []), dtype=tf.int32)

                result = dirt.rasterise_batch(
                    background=tf.zeros([bbox_vertices_clip.get_shape()[0], frame_height, frame_width, 1]),
                    vertices=bbox_vertices_clip,
                    vertex_colors=tf.ones_like(bbox_vertices_clip[:, :, :1]),
                    faces=tf.tile(bbox_faces[None], [bbox_vertices_clip.get_shape()[0], 1, 1])
                )  # im * obj, y, x, 1
                return tf.reshape(result, bboxes.get_shape()[:2].as_list() + [frame_height, frame_width])

            def record_batch(self, gt_bboxes, threshold_to_recon_bboxes, projection_matrix):

                # gt_bboxes :: eib, fie, obj, min/max, x/y/z
                # values of threshold_to_recon_bboxes :: eib, fie, obj, min/max * x/y/z / score
                # These are axis-aligned and given in view (camera) space; however, there is no guarantee
                # the object is actually visible in the relevant frame

                # 1. Flatten together frames and episodes, as we treat images independently
                gt_bboxes = tf.reshape(gt_bboxes, [-1] + gt_bboxes.get_shape()[2:].as_list())  # im, gt-obj, min/max, x/y/z
                ordered_thresholds = sorted(threshold_to_recon_bboxes.keys())
                recon_bboxes_by_threshold = tf.reshape(
                    tf.stack([threshold_to_recon_bboxes[threshold] for threshold in ordered_thresholds]),
                    [len(ordered_thresholds), episodes_per_batch * frames_per_episode, -1, 7]
                )  # threshold, im, recon-obj, min-x/-y/-z / max-x/-y/-z / score

                # 2. Determine which gt objects 'exist' in each frame -- i.e. are not all-zero (used by the
                # generator to indicate an unused slot), and have some projection in the frame
                # ** ideally we would use the gt masks, but these are not 'linked' to the bboxes!
                gt_unused_slot = tf.reduce_all(tf.equal(gt_bboxes, 0.), axis=[2, 3])  # im, gt-obj
                gt_bbox_images = self._render_bboxes(gt_bboxes, projection_matrix)  # im, gt-obj, y, x
                gt_visible = tf.greater(tf.reduce_sum(gt_bbox_images, axis=[2, 3]), self._mask_visibility_count_threshold)  # im, gt-obj
                gt_valid = tf.logical_and(tf.logical_not(gt_unused_slot), gt_visible)  # im, gt-obj

                # 3. Determine which recon objects 'exist' in each frame -- i.e. have non-zero score (used
                # by the bbox calculation to indicate no voxels over threshold), and whose corresponding
                # segmentation in the frame is non-empty
                recon_bbox_images = tf.reshape(
                    self._render_bboxes(tf.reshape(recon_bboxes_by_threshold[..., :-1], [-1, recon_bboxes_by_threshold.get_shape()[2], 2, 3]), projection_matrix),
                    recon_bboxes_by_threshold.get_shape()[:3].as_list() + [frame_height, frame_width]
                )  # threshold, im, recon-obj, y, x
                recon_visible = tf.greater(tf.reduce_sum(recon_bbox_images, axis=[3, 4]), self._mask_visibility_count_threshold)  # threshold, im, recon-obj
                nonzero_score = tf.greater(recon_bboxes_by_threshold[..., -1], 0.)  # threshold, im, recon-obj
                recon_valid = tf.logical_and(recon_visible, nonzero_score)

                # 4. Calculate IOUs between all pairs of valid objects in each frame, and find most-and-
                # sufficiently-overlapping gt-object for each recon-object
                def iou1d(gt, recon):
                    # gt and recon both :: threshold, im, recon-obj, gt-obj, min/max (up to broadcasting)
                    # result :: threshold, im, recon-obj, gt-obj
                    gt_min, gt_max = gt[..., 0], gt[..., 1]
                    recon_min, recon_max = recon[..., 0], recon[..., 1]
                    intersection = tf.nn.relu(tf.minimum(gt_max, recon_max) - tf.maximum(gt_min, recon_min))
                    union = gt_max - gt_min + recon_max - recon_min - intersection
                    return intersection / tf.maximum(union, 1.e-12)
                def iou3d(gt, recon):
                    # gt and recon both :: threshold, im, recon-obj, gt-obj, min/max, x/y/z (up to broadcasting)
                    # result :: threshold, im, recon-obj, gt-obj
                    return iou1d(gt[..., 0], recon[..., 0]) * iou1d(gt[..., 1], recon[..., 1]) * iou1d(gt[..., 2], recon[..., 2])
                ious = iou3d(
                    gt_bboxes[None, :, None],
                    tf.reshape(recon_bboxes_by_threshold[..., :6], recon_bboxes_by_threshold.get_shape()[:3].as_list() + [1, 2, 3])
                )  # :: threshold, im, recon-obj, gt-obj
                ious = tf.where(
                    tf.broadcast_to(gt_valid[:, None, :], ious.get_shape()),
                    ious,
                    tf.zeros_like(ious)
                )
                best_gt_by_recon = tf.argmax(ious, axis=3)  # threshold, im, recon-obj
                has_match_by_recon = tf.greater_equal(tf.reduce_max(ious, axis=3), self._iou_threshold)  # ditto

                # 5. For each valid detection in decreasing order of score, if matched gt exists and is not-yet-
                # detected, count the detection as true-positive and mark the gt as detected; otherwise,
                # count the detection as false-positive
                recon_valid = recon_valid.numpy()
                best_gt_by_recon = best_gt_by_recon.numpy()
                has_match_by_recon = has_match_by_recon.numpy()
                scores = recon_bboxes_by_threshold[..., -1].numpy()  # threshold, im, recon-obj
                for threshold_index in range(scores.shape[0]):
                    detections = self._alpha_threshold_to_detections[ordered_thresholds[threshold_index]]
                    for image_index in range(scores.shape[1]):
                        score_ordering = np.argsort(-scores[threshold_index, image_index])
                        gt_used = np.zeros([gt_valid.shape[1]], dtype=np.bool)  # gt-obj
                        for recon_obj in score_ordering:
                            if not recon_valid[threshold_index, image_index, recon_obj]:
                                continue
                            true_positive = 0.
                            if has_match_by_recon[threshold_index, image_index, recon_obj]:
                                matched_gt_obj = best_gt_by_recon[threshold_index, image_index, recon_obj]
                                if not gt_used[matched_gt_obj]:
                                    true_positive = 1.
                                    gt_used[matched_gt_obj] = True
                            detections.true_positives.append(true_positive)
                            detections.false_positives.append(1. - true_positive)
                            detections.scores.append(scores[threshold_index, image_index, recon_obj])

                # 6. Record the total number of valid ground-truth objects
                self._gt_true_count += int(tf.reduce_sum(tf.cast(gt_valid, tf.int32)))

            def _calculate_ap(self, detections):

                true_positives, false_positives, scores = detections
                order = np.argsort(-np.float32(scores))
                true_positives = np.cumsum(np.float32(true_positives)[order])
                false_positives = np.cumsum(np.float32(false_positives)[order])
                precisions = true_positives / (true_positives + false_positives)
                recalls = true_positives / self._gt_true_count

                # The following is based on https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py

                # first append sentinel values at the end
                mpre = np.concatenate([[0.], precisions, [0.]])
                mrec = np.concatenate([[0.], recalls, [1.]])

                # compute the precision envelope
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

                return ap

            DetectionStatisticsForThreshold = namedtuple('DetectionStatisticsForThreshold', [
                'average_precision'
            ])

            def get_statistics(self):

                return {threshold: self.DetectionStatisticsForThreshold(
                    average_precision=self._calculate_ap(detections)
                ) for threshold, detections in self._alpha_threshold_to_detections.items()}

        try:

            pixel_accuracy = PixelAccuracy()
            segmentation_accuracy = SegmentationAccuracy()
            depth_accuracy = DepthAccuracy()
            detection_accuracy = DetectionAccuracy()

            def print_statistics():
                pixel_statistics = pixel_accuracy.get_statistics()
                threshold_to_segmentation_statistics = segmentation_accuracy.get_statistics()
                threshold_to_detection_statistics = detection_accuracy.get_statistics()
                depth_statistics = depth_accuracy.get_statistics()
                print(pixel_statistics)
                print(threshold_to_segmentation_statistics)
                print(threshold_to_detection_statistics)
                print(depth_statistics)
                if final_eval:
                    segmentation_thresholds = sorted(threshold_to_segmentation_statistics.keys())
                    detection_thresholds = sorted(threshold_to_detection_statistics.keys())
                    names = pixel_statistics._fields + depth_statistics._fields[:-1] + tuple([
                        f'fraction_delta_less_{threshold}'
                        for threshold in depth_statistics.threshold_to_fraction_delta_less
                    ]) + sum([
                        tuple([
                            name + f'@{threshold:.2f}'
                            for name in threshold_to_segmentation_statistics[threshold]._fields
                        ])
                        for threshold in segmentation_thresholds
                    ], ()) + sum([
                        tuple([
                            name + f'@{threshold:.2f}'
                            for name in threshold_to_detection_statistics[threshold]._fields
                        ])
                        for threshold in detection_thresholds
                    ], ())
                    values = pixel_statistics + depth_statistics[:-1] + tuple([
                        depth_statistics.threshold_to_fraction_delta_less[threshold]
                        for threshold in depth_statistics.threshold_to_fraction_delta_less
                    ]) + sum([
                        threshold_to_segmentation_statistics[threshold]
                        for threshold in segmentation_thresholds
                    ], ()) + sum([
                        threshold_to_detection_statistics[threshold]
                        for threshold in detection_thresholds
                    ], ())
                    print(','.join(names))
                    print(','.join(map(str, values)))

            @tf.function(autograph=False)
            def reconstruct_and_maybe_visualise(batch_index, gt_pixels, gt_camera_matrices, gt_depth, gt_segmentation):

                recon_pixels, recon_depth, recon_normals, recon_background, recon_objects, recon_masks_by_depth, recon_room_vertices, threshold_to_recon_bboxes = \
                    self.inferencer.reconstruct(pixels=gt_pixels, maybe_camera_matrices=gt_camera_matrices, gt_pixels_for_regularisation_and_debug=gt_pixels)

                if output_path is not None:

                    gen_pixels, gen_depth, _, gen_background, gen_objects, gen_masks_by_depth, _ , _= self.inferencer.sample_prior(maybe_camera_matrices=gt_camera_matrices, gt_pixels_for_regularisation_and_debug=gt_pixels)

                    # Write the 'overview' images

                    self.visualise_batch(
                        batch_index, output_path + '/' + split + '-' + output_suffix,
                        gt, gt_depth, gt_segmentation, gt_pixels,
                        recon_background, recon_depth, recon_normals, recon_objects, recon_masks_by_depth, recon_pixels, recon_room_vertices,
                        gen_background, gen_depth, gen_objects, gen_masks_by_depth, gen_pixels
                    )

                    # Write the FID/FVD eval images

                    def write_frechet_image(pixels, subfolder):
                        tf.io.write_file(
                            output_path + '/' + split + '-' + output_suffix + '/' + subfolder + '/' + tf.strings.as_string(batch_index * episodes_per_batch + episode_index, width=6, fill='0') + '_{:02}.png'.format(frame_index),
                            tf.image.encode_png(tf.cast(tf.clip_by_value(pixels[episode_index, frame_index] * 255, 0, 255), tf.uint8))
                        )

                    for episode_index in range(episodes_per_batch):
                        for frame_index in range(frames_per_episode):
                            write_frechet_image(gt_pixels, 'gt')
                            write_frechet_image(gen_pixels, 'gen')

                return recon_pixels, recon_depth, recon_normals, recon_background, recon_objects, recon_masks_by_depth, recon_room_vertices, threshold_to_recon_bboxes

            for batch_index, (gt_pixels, gt_camera_matrices, gt_depth, gt_normals, gt_segmentation, gt_bboxes) in enumerate(gt.batches(split)):

                recon_pixels, recon_depth, recon_normals, recon_background, recon_objects, recon_masks_by_depth, recon_room_vertices, threshold_to_recon_bboxes = \
                    reconstruct_and_maybe_visualise(tf.constant(batch_index), gt_pixels, gt_camera_matrices, gt_depth, gt_segmentation)

                pixel_accuracy.record_batch(gt_pixels, recon_pixels)
                segmentation_accuracy.record_batch(gt_segmentation, recon_masks_by_depth)
                depth_accuracy.record_batch(gt_depth, recon_depth)
                detection_accuracy.record_batch(gt_bboxes, threshold_to_recon_bboxes, self.projection_matrix)

                if batch_index > 0 and batch_index % 100 == 0:
                    print('subtotal statistics after {} batches:'.format(batch_index + 1))
                    print_statistics()

                if batch_index == 1:  # i.e. wait till the generative and variational have been run in full
                    if self.temporary_path_for_deletion is not None:
                        shutil.rmtree(self.temporary_path_for_deletion)
                        self.temporary_path_for_deletion = None

                if batch_index * episodes_per_batch >= 10000:
                    print('stopping at 10K episodes')
                    break

            print('final statistics:')
            print_statistics()

        except KeyboardInterrupt:
            pass

    def visualise_batch(
        self, iteration_or_index, output_path,
        gt, gt_depth, gt_segmentation, gt_pixels,
        recon_background, recon_depth, recon_normals, recon_objects, recon_masks_by_depth, recon_pixels, recon_room_vertices,
        gen_background, gen_depth, gen_objects, gen_masks_by_depth, gen_pixels
    ):
        min_depth = gt.base_scale_xz * 0.01
        max_depth = gt.base_scale_xz * 1.5
        depth_colourmap = tf.constant(matplotlib.cm.get_cmap('magma').colors, dtype=tf.float32)
        def depth_to_rgb(depth):
            depth = tf.clip_by_value(1. - (depth - min_depth) / (max_depth - min_depth), 0., 1.)
            depth = depth ** 3.
            depth_quantised = tf.cast(tf.round(depth * 255.), tf.int32)
            depth = tf.gather(depth_colourmap, depth_quantised)
            return depth

        segmentation_colours = tf.image.hsv_to_rgb(tf.random.uniform([self.total_grid_cells, 3], [0., 0.5, 0.5], [1., 1., 1.]))
        def masks_to_segmentation(masks_by_depth):
            modal_masks = binarise_and_modalise_masks(masks_by_depth, tf.constant(0.5))  # obj, eib, fie, y, x
            return tf.reduce_sum(modal_masks[..., None] * segmentation_colours[:, None, None, None, None, :], axis=0)

        vis_pixels = tf.concat([
            gt_pixels,
            tf.clip_by_value(recon_pixels, 0., 1.),
            tf.clip_by_value(recon_background, 0., 1.),
            tf.clip_by_value(recon_objects, 0., 1.),
            (recon_normals + 1.) / 2,
            depth_to_rgb(gt_depth),
            depth_to_rgb(recon_depth),
            tf.cast(gt_segmentation, tf.float32) / 255.,
            masks_to_segmentation(recon_masks_by_depth),
            tf.clip_by_value(gen_pixels, 0., 1.),
            tf.clip_by_value(gen_background, 0., 1.),
            tf.clip_by_value(gen_objects, 0., 1.),
            depth_to_rgb(gen_depth),
            masks_to_segmentation(gen_masks_by_depth)
        ], axis=2)[:8]

        for frame_index in range(frames_per_episode):
            vis_frame = tf.reshape(tf.transpose(vis_pixels[:, frame_index], [1, 0, 2, 3]), [vis_pixels.shape[2], -1, 3])
            tf.io.write_file(
                output_path + '/' + tf.strings.as_string(iteration_or_index, width=6, fill='0') + '_{:02}.jpg'.format(frame_index),
                tf.image.encode_jpeg(tf.cast(tf.clip_by_value(vis_frame * 255, 0, 255), tf.uint8))
            )


def main():

    array_job_id = os.getenv('SLURM_ARRAY_JOB_ID')
    if array_job_id is None:
        run_id = os.getenv('SLURM_JOB_ID', str(os.getpid()))
    else:
        run_id = array_job_id + '-' + os.getenv('SLURM_ARRAY_TASK_ID')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '_' + run_id

    tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    tf.random.set_random_seed(random_seed)

    gt = TfRecordsGroundTruth('./data/carla/preprocessed')
    model = RoomVae(gt)

    if final_eval:

        assert model.restored_job_name is not None
        output_path = os.path.join(output_base_path, model.restored_job_name)
        model.test_on_split(gt, output_path, split='test', output_suffix=f'{model.first_iteration - 1:06}')

    else:

        output_subfolder = os.path.join(gt.data_string, model.param_string, 'seed-{}'.format(random_seed), timestamp)
        output_path = os.path.join(output_base_path, output_subfolder)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('output subfolder: ' + output_subfolder)

        model.train(gt, output_path, visualise_frequency=100 if array_job_id is None else 500)


if __name__ == '__main__':

    main()

