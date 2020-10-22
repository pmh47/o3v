
# Invoke with e.g. python rooms_ring_camera.py --output-directory ../../data/gqn-rooms/1-4-obj --image-size 80 --min-num-objects 1 --max-num-objects 4 --total-scenes 100000

import argparse
import colorsys
import math
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyglet
import trimesh
from PIL import Image, ImageEnhance
from tqdm import tqdm

from OpenGL.GL import GL_LINEAR_MIPMAP_LINEAR

import pyrender
from pyrender import (DirectionalLight, Mesh, Node, OffscreenRenderer,
                      PerspectiveCamera, PointLight, RenderFlags, Scene,
                      Primitive)
from pyrender.material import SpecularGlossinessMaterial

import tensorflow as tf
from crop_extraction_common import ShardedRecordWriter, float32_feature

texture_directory = os.path.join(os.path.dirname(__file__), "..", "textures")
object_directory = os.path.join(os.path.dirname(__file__), "objects")

floor_textures = [
    "{}/lg_floor_d.tga".format(texture_directory),
    "{}/lg_style_01_floor_blue_d.tga".format(texture_directory),
    "{}/lg_style_01_floor_orange_bright_d.tga".format(texture_directory),
]

wall_textures = [
    "{}/lg_style_01_wall_cerise_d.tga".format(texture_directory),
    "{}/lg_style_01_wall_green_bright_d.tga".format(texture_directory),
    "{}/lg_style_01_wall_red_bright_d.tga".format(texture_directory),
    "{}/lg_style_02_wall_yellow_d.tga".format(texture_directory),
    "{}/lg_style_03_wall_orange_bright_d.tga".format(texture_directory),
]


def bottomless_capsule():

    capsule = pyrender.objects.Capsule()
    primitive = capsule.mesh.primitives[0]
    primitive.positions = np.stack([primitive.positions[:, 0], primitive.positions[:, 1], np.clip(primitive.positions[:, 2], -np.inf, 0.75 - 1.e-4)], axis=1)
    return capsule


objects = [
    bottomless_capsule,
    pyrender.objects.Cylinder,
    pyrender.objects.Icosahedron,
    pyrender.objects.Box,
    pyrender.objects.Sphere,
]


def set_random_texture(node, path):
    texture_image = Image.open(path).convert("RGB")
    primitive = node.mesh.primitives[0]
    assert isinstance(primitive, Primitive)
    primitive.material.baseColorTexture.source = texture_image
    primitive.material.baseColorTexture.sampler.minFilter = GL_LINEAR_MIPMAP_LINEAR


def make_normals_node(trimesh, node, normal_direction):
    normals_mesh = Mesh.from_trimesh(trimesh, smooth=False)
    normals_mesh.primitives[0].material = SpecularGlossinessMaterial(diffuseFactor=list(np.asarray(normal_direction) / 2 + 0.5) + [1.])
    return Node(
        mesh=normals_mesh,
        rotation=node.rotation,
        translation=node.translation
    )


def build_scene(floor_textures, wall_textures, fix_light_position=False):
    full_scene = Scene(
        bg_color=np.array([153 / 255, 226 / 255, 249 / 255]),
        ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))
    normals_scene = Scene(
        bg_color=np.array([0., 0., 0.]),
        ambient_light=np.array([1., 1., 1., 1.]))
    masks_scene = Scene(
        bg_color=np.array([0., 0., 0.]),
        ambient_light=np.array([1., 1., 1., 1.]))

    floor_trimesh = trimesh.load("{}/floor.obj".format(object_directory))
    mesh = Mesh.from_trimesh(floor_trimesh, smooth=False)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_pitch(-math.pi / 2),
        translation=np.array([0, 0, 0]))
    texture_path = random.choice(floor_textures)
    set_random_texture(node, texture_path)
    full_scene.add_node(node)
    normals_scene.add_node(make_normals_node(floor_trimesh, node, [0., 1., 0.]))

    texture_path = random.choice(wall_textures)

    wall_trimesh = trimesh.load("{}/wall.obj".format(object_directory))
    mesh = Mesh.from_trimesh(wall_trimesh, smooth=False)
    node = Node(mesh=mesh, translation=np.array([0, 1.15, -3.5]))
    set_random_texture(node, texture_path)
    full_scene.add_node(node)
    normals_scene.add_node(make_normals_node(wall_trimesh, node, [0., 0., 1.]))

    mesh = Mesh.from_trimesh(wall_trimesh, smooth=False)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_yaw(math.pi),
        translation=np.array([0, 1.15, 3.5]))
    set_random_texture(node, texture_path)
    full_scene.add_node(node)
    normals_scene.add_node(make_normals_node(wall_trimesh, node, [0., 0., -1.]))

    mesh = Mesh.from_trimesh(wall_trimesh, smooth=False)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_yaw(-math.pi / 2),
        translation=np.array([3.5, 1.15, 0]))
    set_random_texture(node, texture_path)
    full_scene.add_node(node)
    normals_scene.add_node(make_normals_node(wall_trimesh, node, [-1., 0., 0.]))

    mesh = Mesh.from_trimesh(wall_trimesh, smooth=False)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_yaw(math.pi / 2),
        translation=np.array([-3.5, 1.15, 0]))
    set_random_texture(node, texture_path)
    full_scene.add_node(node)
    normals_scene.add_node(make_normals_node(wall_trimesh, node, [1., 0., 0.]))

    light = DirectionalLight(color=np.ones(3), intensity=10)
    if fix_light_position == True:
        translation = np.array([1, 1, 1])
    else:
        xz = np.random.uniform(-1, 1, size=2)
        translation = np.array([xz[0], 1, xz[1]])
    yaw, pitch = compute_yaw_and_pitch(translation)
    node = Node(
        light=light,
        rotation=genearte_camera_quaternion(yaw, pitch),
        translation=translation)
    full_scene.add_node(node)

    return full_scene, normals_scene, masks_scene


def place_objects(full_scene,
                  masks_scene,
                  colors,
                  objects,
                  max_num_objects=3,
                  min_num_objects=1,
                  discrete_position=False,
                  rotate_object=False):
    # Place objects
    directions = [-1.5, 0.0, 1.5]
    available_positions = []
    for z in directions:
        for x in directions:
            available_positions.append((x, z))
    available_positions = np.array(available_positions)
    num_objects = random.choice(range(min_num_objects, max_num_objects + 1))
    indices = np.random.choice(
        np.arange(len(available_positions)), replace=False, size=num_objects)
    parents = []
    mask_parents = []
    for xz in available_positions[indices]:
        object_ctor = random.choice(objects)
        node = object_ctor()
        node.mesh.primitives[0].color_0 = random.choice(colors)
        mask_node = object_ctor()
        mask_node.mesh.primitives[0].color_0 = list(np.random.uniform(0.05, 1., size=[3])) + [1.]
        if discrete_position == False:
            xz += np.random.uniform(-0.3, 0.3, size=xz.shape)
        if rotate_object:
            yaw = np.random.uniform(0, math.pi * 2, size=1)[0]
            rotation = pyrender.quaternion.from_yaw(yaw)
            parent = Node(
                children=[node],
                rotation=rotation,
                translation=np.array([xz[0], 0, xz[1]]))
            mask_parent = Node(
                children=[mask_node],
                rotation=rotation,
                translation=np.array([xz[0], 0, xz[1]]))
        else:
            parent = Node(
                children=[node], translation=np.array([xz[0], 0, xz[1]]))
            mask_parent = Node(
                children=[mask_node], translation=np.array([xz[0], 0, xz[1]]))
        parent.scale = [args.object_scale] * 3
        mask_parent.scale = [args.object_scale] * 3
        full_scene.add_node(parent)
        masks_scene.add_node(mask_parent)
        parents.append(parent)
        mask_parents.append(mask_parent)
    return parents, mask_parents


def udpate_vertex_buffer(cube_nodes):
    for node in (cube_nodes):
        node.mesh.primitives[0].update_vertex_buffer_data()


def compute_yaw_and_pitch(vec):
    x, y, z = vec
    norm = np.linalg.norm(vec)
    if z < 0:
        yaw = math.pi + math.atan(x / z)
    elif x < 0:
        if z == 0:
            yaw = math.pi * 1.5
        else:
            yaw = math.pi * 2 + math.atan(x / z)
    elif z == 0:
        yaw = math.pi / 2
    else:
        yaw = math.atan(x / z)
    pitch = -math.asin(y / norm)
    return yaw, pitch


def genearte_camera_quaternion(yaw, pitch):
    quaternion_yaw = pyrender.quaternion.from_yaw(yaw)
    quaternion_pitch = pyrender.quaternion.from_pitch(pitch)
    quaternion = pyrender.quaternion.multiply(quaternion_pitch, quaternion_yaw)
    quaternion = quaternion / np.linalg.norm(quaternion)
    return quaternion


def main():

    # Colors
    colors = []
    for n in range(args.num_colors):
        hue = n / args.num_colors
        saturation = 1
        lightness = 1
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        colors.append(np.array((red, green, blue, 1)))

    renderer = OffscreenRenderer(
        viewport_width=args.image_size, viewport_height=args.image_size)

    os.makedirs(args.output_directory, exist_ok=True)

    with ShardedRecordWriter(args.output_directory + '/{:04d}.tfrecords', args.num_scenes_per_file) as writer:

        for scene_index in tqdm(range(args.total_scenes)):
            full_scene, normals_scene, masks_scene = build_scene(
                floor_textures,
                wall_textures,
                fix_light_position=args.fix_light_position)
            object_nodes, object_mask_nodes = place_objects(
                full_scene,
                masks_scene,
                colors,
                objects,
                max_num_objects=args.max_num_objects,
                min_num_objects=args.min_num_objects,
                discrete_position=args.discrete_position,
                rotate_object=args.rotate_object)
            object_velocities = np.random.uniform(-1., 1., [len(object_nodes), 3]) * [1., 0., 1.]
            camera_distance = np.random.uniform(3., 4.8)
            camera = PerspectiveCamera(yfov=math.pi / 4)
            camera_node = Node(camera=camera, translation=np.array([0, 1, 1]))
            full_scene.add_node(camera_node)
            normals_scene.add_node(camera_node)
            masks_scene.add_node(camera_node)
            initial_yaw = np.random.uniform(-np.pi, np.pi)
            delta_yaw = np.random.normal(0.3, 0.05) * (np.random.randint(2) * 2 - 1.)
            pitch = 0.  # np.random.normal(0., 0.1) - 0.03
            all_frames = []
            all_depths = []
            all_masks = []
            all_normals = []
            all_bboxes = []
            all_camera_positions = []
            all_camera_yaws = []
            all_camera_pitches = []
            all_camera_matrices = []
            for observation_index in range(args.num_observations_per_scene):

                yaw = initial_yaw + delta_yaw * observation_index

                camera_xz = camera_distance * np.array(
                    (math.sin(yaw), math.cos(yaw)))

                camera_node.rotation = genearte_camera_quaternion(yaw, pitch)
                camera_position = np.array( [camera_xz[0], 1, camera_xz[1]])
                camera_node.translation = camera_position

                # Image and depths (latter are linear float32 depths in world space, with zero for sky)
                flags = RenderFlags.NONE if args.no_shadows else RenderFlags.SHADOWS_DIRECTIONAL
                if args.anti_aliasing:
                    flags |= RenderFlags.ANTI_ALIASING
                image, depths = renderer.render(full_scene, flags=flags)

                # Background (wall/floor) normals in view space
                normals_world = renderer.render(normals_scene, flags=RenderFlags.NONE)[0]
                normals_world = np.where(np.sum(normals_world, axis=2, keepdims=True) == 0, 0., (normals_world.astype(np.float32) / 255. - 0.5) * 2)  # this has zeros for the sky
                normals_view = np.einsum('ij,yxj->yxi', np.linalg.inv(camera_node.matrix[:3, :3]), normals_world)

                # Instance segmentation masks
                masks_image = renderer.render(masks_scene, flags=RenderFlags.NONE)[0]

                # Instance 3D bboxes in view space (axis-aligned)
                def get_mesh_node_bbox(node):
                    object_to_view_matrix = np.dot(np.linalg.inv(camera_node.matrix), full_scene.get_pose(node))
                    assert len(node.mesh.primitives) == 1
                    vertices_object = np.concatenate([node.mesh.primitives[0].positions, np.ones_like(node.mesh.primitives[0].positions[:, :1])], axis=1)
                    vertices_view = np.einsum('ij,vj->vi', object_to_view_matrix, vertices_object)[:, :3]
                    return np.min(vertices_view, axis=0), np.max(vertices_view, axis=0)
                object_bboxes_view = [
                    get_mesh_node_bbox(object_parent.children[0])
                    for object_parent in object_nodes
                ]

                all_frames.append(cv2.imencode('.jpg', image[..., ::-1])[1].tostring())
                all_masks.append(cv2.imencode('.png', masks_image[..., ::-1])[1].tostring())
                all_depths.append(depths)
                all_normals.append(normals_view)
                all_bboxes.append(object_bboxes_view)
                all_camera_positions.append(camera_position)
                all_camera_yaws.append(yaw)
                all_camera_pitches.append(pitch)
                all_camera_matrices.append(camera_node.matrix)

                if args.visualize:
                    plt.clf()
                    plt.imshow(image)
                    plt.pause(1e-10)

                if args.moving_objects:
                    for object_node, object_velocity in zip(object_nodes, object_velocities):
                        new_translation = object_node.translation + object_velocity
                        new_translation = np.clip(new_translation, -3., 3.)
                        object_node.translation = new_translation

            all_bboxes = np.asarray(all_bboxes)  # :: frame, obj, min/max, x/y/z
            all_bboxes = np.concatenate([all_bboxes, np.zeros([all_bboxes.shape[0], args.max_num_objects - all_bboxes.shape[1], 2, 3])], axis=1)

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'frames': tf.train.Feature(bytes_list=tf.train.BytesList(value=all_frames)),
                    'masks': tf.train.Feature(bytes_list=tf.train.BytesList(value=all_masks)),
                    'depths': float32_feature(all_depths),
                    'normals': float32_feature(all_normals),
                    'bboxes': float32_feature(all_bboxes),
                    'camera_positions': float32_feature(all_camera_positions),
                    'camera_yaws': float32_feature(all_camera_yaws),
                    'camera_pitches': float32_feature(all_camera_pitches),
                    'camera_matrices': float32_feature(all_camera_matrices),
                })
            )
            writer.write(example.SerializeToString())

    renderer.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-scenes", "-total", type=int, default=2000000)
    parser.add_argument("--num-scenes-per-file", type=int, default=2000)
    parser.add_argument("--num-observations-per-scene", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-num-objects", type=int, default=3)
    parser.add_argument("--min-num-objects", type=int, default=1)
    parser.add_argument("--num-colors", type=int, default=6)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--anti-aliasing", default=False, action="store_true")
    parser.add_argument(
        "--discrete-position", default=False, action="store_true")
    parser.add_argument("--rotate-object", default=False, action="store_true")
    parser.add_argument(
        "--fix-light-position", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    parser.add_argument("--no-shadows", default=False, action="store_true")
    parser.add_argument("--moving-objects", default=False, action="store_true")
    parser.add_argument("--object-scale", type=float, default=1.)
    args = parser.parse_args()
    main()
