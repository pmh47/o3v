
import os
import cv2
import pickle
import argparse
import numpy as np
import tensorflow as tf
import skimage.color
import skimage.measure
import skimage.morphology
from tqdm import tqdm

import dirt.matrices
from crop_extraction_common import ShardedRecordWriter, float32_feature


def main():

    tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', required=True)
    parser.add_argument('--output-folder', required=True)
    parser.add_argument("--num-scenes-per-file", type=int, default=2000)
    parser.add_argument("--num-observations-per-scene", type=int, default=3)
    parser.add_argument("--frame-step", type=int, default=6)
    parser.add_argument("--max-num-objects", type=int, default=3)
    parser.add_argument("--min-num-objects", type=int, default=1)
    parser.add_argument("--max-bboxes", type=int, default=5)  # this includes bboxes for cars that exist, but are not visible in a given particular frame, hence may be greater than max-num-objects
    parser.add_argument("--val-percent", type=float, default=10)
    parser.add_argument("--test-percent", type=float, default=10)
    args = parser.parse_args()

    all_sequence_ids = sorted([
        filename[:filename.index('_')]
        for filename in os.listdir(args.input_folder)
        if filename.endswith('_camera.txt')
    ])
    print('found {} sequences'.format(len(all_sequence_ids)))

    first_val_index = len(all_sequence_ids) * (100 - args.val_percent - args.test_percent) // 100
    first_test_index = len(all_sequence_ids) * (100 - args.test_percent) // 100

    do_for_split('train', all_sequence_ids[:first_val_index], args)
    do_for_split('val', all_sequence_ids[first_val_index : first_test_index], args)
    do_for_split('test', all_sequence_ids[first_test_index:], args)


def get_projection_matrix():

    near_plane_distance = 0.05
    fov_angle = 100. * np.pi / 180.
    near_plane_right = near_plane_distance * tf.tan(fov_angle / 2.)
    return dirt.matrices.perspective_projection(
        near_plane_distance,
        far=100.,
        right=near_plane_right,
        aspect=72 / 96
    ).numpy()  # :: x/y/x/z (in), x/y/z/w (out)


def get_instance_segmentations(fg_segmentations, bboxes_view, vehicle_colours):

    # fg_segmentations :: frame, y, x (boolean)
    # bboxes_view :: frame, vehicle, bbox-corner, x/y/z
    # vehicle_colours :: vehicle, r/g/b (uint8)

    projection_matrix = get_projection_matrix()
    bboxes_clip = np.matmul(bboxes_view, projection_matrix)
    frame_height, frame_width = fg_segmentations.shape[1:]

    cube_faces = np.int32([
        [0, 4, 6], [0, 6, 2],  # bottom
        [1, 5, 7], [1, 7, 3],  # top
        [3, 7, 6], [3, 6, 2],  # right
        [1, 5, 4], [1, 4, 0],  # left
        [6, 4, 5], [6, 5, 7],  # front
        [2, 0, 1], [2, 1, 3],  # back
    ])
    bbox_masks = dirt.rasterise_batch(
        background=np.zeros([bboxes_view.shape[0] * bboxes_view.shape[1], frame_height, frame_width, 1]),
        vertices=np.reshape(bboxes_clip, [-1, 8, 4]),
        vertex_colors=np.ones([bboxes_view.shape[0] * bboxes_view.shape[1], 8, 1]),
        faces=np.tile(cube_faces[None], [bboxes_view.shape[0] * bboxes_view.shape[1], 1, 1])
    ).numpy().reshape([bboxes_view.shape[0], bboxes_view.shape[1], frame_height, frame_width]) > 0.5

    vehicle_depths = np.mean(bboxes_clip[:, :, 3], axis=2)

    result = np.zeros([fg_segmentations.shape[0], frame_height, frame_width, 3], dtype=np.uint8)
    for frame_index, (vehicle_depths_for_frame, bbox_masks_for_frame, fg_mask_for_frame) in enumerate(zip(vehicle_depths, bbox_masks, fg_segmentations)):
        order = np.argsort(vehicle_depths_for_frame)
        for bbox_mask, colour_for_vehicle in zip(bbox_masks_for_frame[order], vehicle_colours[order]):
            composed_mask = bbox_mask * fg_mask_for_frame
            result[frame_index] += composed_mask[:, :, None].astype(np.uint8) * colour_for_vehicle
            fg_mask_for_frame *= np.logical_not(composed_mask)

    return result


def do_for_split(split, sequence_ids, args):

    print('preprocessing {} sequences for split {}...'.format(len(sequence_ids), split))
    split_folder = args.output_folder + '/' + split
    os.makedirs(split_folder, exist_ok=True)
    with ShardedRecordWriter(split_folder + '/{:04d}.tfrecords', args.num_scenes_per_file) as writer:

        total_examples = 0
        for sequence_id in tqdm(sequence_ids):

            with open('{}/{}_camera.txt'.format(args.input_folder, sequence_id), 'r') as f:
                camera_transforms = map(lambda line: list(map(float, line[:-1].split(' '))), f.readlines())
            camera_transforms = np.float32(list(camera_transforms))

            with open('{}/{}_bboxes.pkl'.format(args.input_folder, sequence_id), 'rb') as f:
                vehicle_bboxes_by_frame = pickle.load(f)  # [frame], [vehicle], bbox-corner, x/y/z
            vehicle_bboxes_by_frame = np.asarray(vehicle_bboxes_by_frame)
            vehicle_segmentation_colours = (skimage.color.hsv2rgb(np.random.uniform([0., 0.5, 0.5], [1., 1., 1.], size=[1, vehicle_bboxes_by_frame.shape[1], 3]))[0] * 255).astype(np.uint8)

            # Note that the following samples overlapping sequences, i.e. they are non-independent; this effectively means
            # we sample starting at arbitrary offsets rather than regularly-spaced ones
            # ** could instead sample longer sequences and rely on TfRecordsGroundTruth to extract subsequences; this
            # ** results in less duplication in the tfrecord files, but slightly more restricted choice of initial offset (as
            # ** subsequences cannot cross a full-sequence boundary), hence makes slightly less efficient use of the data
            for first_frame_index in range(len(camera_transforms) - args.num_observations_per_scene * args.frame_step):

                last_frame_index = first_frame_index + args.num_observations_per_scene * args.frame_step  # exclusive

                frames = []
                fg_segmentations = []
                depths = []
                for frame_index in range(first_frame_index, last_frame_index, args.frame_step):

                    image = cv2.imread('{}/{}_{:03}.jpg'.format(args.input_folder, sequence_id, frame_index))
                    semantic_segmentation = cv2.imread('{}/{}_{:03}_ss.png'.format(args.input_folder, sequence_id, frame_index))
                    depth = cv2.imread('{}/{}_{:03}_depth.png'.format(args.input_folder, sequence_id, frame_index))

                    def crop_and_scale(image, interpolation):
                        return cv2.resize(image, (96, 72), interpolation=interpolation)

                    image = crop_and_scale(image, cv2.INTER_AREA)
                    segmentation = crop_and_scale(semantic_segmentation, cv2.INTER_NEAREST)
                    depth = crop_and_scale(depth, cv2.INTER_NEAREST)

                    cars_mask = np.all(segmentation == [142, 0, 0], axis=-1)
                    _, car_count = skimage.measure.label(skimage.morphology.binary_erosion(cars_mask), return_num=True)

                    if car_count < args.min_num_objects or car_count > args.max_num_objects:
                        break

                    frames.append(cv2.imencode('.jpg', image)[1].tostring())
                    depths.append(cv2.imencode('.png', depth)[1].tostring())
                    fg_segmentations.append(cars_mask)

                if len(frames) < args.num_observations_per_scene:
                    continue

                # Note that rotations are stored in degrees, and ordered yaw, pitch, roll
                locations_ue, rotations = np.split(camera_transforms[first_frame_index : last_frame_index : args.frame_step], 2, axis=1)
                expected_pitch = -5.
                if np.max(np.abs(rotations[:, 2])) > 1.:
                    continue  # skip the episode if there is significant camera roll
                if np.max(np.abs(rotations[:, 1] - expected_pitch)) > 1.:
                    continue  # skip the episode if there is significant additional camera pitch due to the road

                # UE4 uses the convention that +ve x is forward, +ve y is rightward, +ve z is upward
                ue_to_gl_matrix = tf.constant([
                    [0., 0., -1.],
                    [1., 0., 0.],
                    [0., 1., 0.],
                ])
                locations_gl = tf.matmul(tf.constant(locations_ue), ue_to_gl_matrix)

                relative_locations = locations_gl - locations_gl[:1]
                relative_yaws = rotations[:, 0] - rotations[:1, 0]
                initial_yaw_matrix = dirt.matrices.rodrigues([0., 1., 0.] * -tf.constant(rotations)[:1, 0] * np.pi / 180., three_by_three=True)
                relative_locations = tf.matmul(relative_locations, initial_yaw_matrix)

                camera_matrices = dirt.matrices.compose(
                    dirt.matrices.rodrigues([-expected_pitch * np.pi / 180., 0, 0]),
                    dirt.matrices.rodrigues([0., 1., 0.] * tf.constant(relative_yaws)[:, None] * np.pi / 180.),
                    dirt.matrices.translation(relative_locations),
                ).numpy()

                world_bboxes_ue = vehicle_bboxes_by_frame[first_frame_index : last_frame_index : args.frame_step]
                world_bboxes_gl = np.matmul(world_bboxes_ue, ue_to_gl_matrix.numpy())
                relative_bboxes = np.matmul(world_bboxes_gl - locations_gl[0], initial_yaw_matrix)
                bboxes_view = np.matmul(
                    np.concatenate([relative_bboxes, np.ones_like(relative_bboxes[..., :1])], axis=-1),
                    np.linalg.inv(camera_matrices)[:, None]
                )

                instance_segmentations = get_instance_segmentations(np.asarray(fg_segmentations), bboxes_view, vehicle_segmentation_colours)
                instance_segmentations = [cv2.imencode('.png', segmentation)[1].tostring() for segmentation in instance_segmentations]

                assert bboxes_view.shape[1] <= args.max_bboxes
                bboxes_view_padded = np.concatenate([
                    bboxes_view[..., :3],  # drop the homogeneous w-coordinate
                    np.zeros([bboxes_view.shape[0], args.max_bboxes - bboxes_view.shape[1], 8, 3])
                ], axis=1)

                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        'frames': tf.train.Feature(bytes_list=tf.train.BytesList(value=frames)),
                        'segmentations': tf.train.Feature(bytes_list=tf.train.BytesList(value=instance_segmentations)),
                        'depths': tf.train.Feature(bytes_list=tf.train.BytesList(value=depths)),
                        'camera_matrices': float32_feature(camera_matrices),
                        'bboxes': float32_feature(bboxes_view_padded)
                    })
                )
                writer.write(example.SerializeToString())
                total_examples += 1

            # # ** debug!
            # if total_examples > 100:
            #     break

        print('wrote {} examples'.format(total_examples))


if __name__ == '__main__':

    main()

