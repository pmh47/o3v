
import os
import re
import sys
import time
import glob
import math
import queue
import pickle
import random
import logging
import argparse
import numpy as np


egg_pattern = '/opt/carla/PythonAPI/carla/dist/carla-*{}.{}-{}.egg'.format(
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'
)
egg_matches = glob.glob(egg_pattern)
if len(egg_matches) == 0:
    raise RuntimeError('no carla egg matching pattern ' + egg_pattern)
elif len(egg_matches) > 1:
    raise RuntimeError('found multiple carla eggs: ' + ', '.join(egg_matches))
else:
    print('found carla egg at ' + egg_matches[0])
sys.path.append(egg_matches[0])

import carla


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self._tm = kwargs.get('traffic_manager', None)

    def __enter__(self):
        if self._tm is not None:
            self._tm.set_synchronous_mode(True)
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)
        if self._tm is not None:
            self._tm.set_synchronous_mode(False)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-o', '--output-folder',
        help='folder to write images to')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '-m', '--map',
        default='Town02',
        help='name of the map to load (default: "Town02")')
    argparser.add_argument(
        '-w', '--weather',
        default='CloudyNoon',
        help='name of the weather preset (default: "CloudyNoon")')
    argparser.add_argument(
        '-tm_p', '--tm_port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '-e', '--episodes',
        default=5,
        type=int,
        help='number of episodes to record')
    argparser.add_argument(
        '--first-episode',
        default=0,
        type=int,
        help='offset to add to episode indices')
    argparser.add_argument(
        '--skip_frames',
        default=30,
        type=int,
        help='number of initial frames to skip')
    argparser.add_argument(
        '--frames_per_episode',
        default=80,
        type=int,
        help='number of frames to record per episode')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    os.makedirs(args.output_folder, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    map_prefix = '/Game/Carla/Maps/'
    map_path = map_prefix + args.map
    if map_path not in client.get_available_maps():
        raise RuntimeError('map {} not in allowed set {}'.format(args.map, [map_name.replace(map_prefix, '') for map_name in sorted(client.get_available_maps())]))
    world = client.load_world(args.map)

    if not hasattr(carla.WeatherParameters, args.weather):
        raise RuntimeError('weather preset {} not in allowed set {}'.format(args.weather, [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]))
    else:
        print('set weather preset %r.' % args.weather)
        world.set_weather(getattr(carla.WeatherParameters, args.weather))

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)

    for episode_index in range(args.episodes):
        print('episode {} / {} (#{})'.format(episode_index + 1, args.episodes, episode_index + args.first_episode))
        try:
            capture_episode(world, traffic_manager, args.output_folder + '/{:04d}'.format(episode_index + args.first_episode), args)
        except Exception as e:
            print('skipping episode due to exception:\n' + str(e))


def capture_episode(world, traffic_manager, output_prefix, args):

    rgb_camera = None
    ss_camera = None
    depth_camera = None
    all_vehicle_ids = []
    try:

        blueprints = world.get_blueprint_library().filter(args.filterv)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('rubicon')]
            blueprints = [x for x in blueprints if not x.id.endswith('nissan.patrol')]
            blueprints = [x for x in blueprints if not x.id.endswith('mustang')]
            blueprints = [x for x in blueprints if not x.id.endswith('police')]

        world_map = world.get_map()

        if 'Town01' in world_map.name:
            locations = [
                (15, [-1, 1], 100., 24),  # verge / front gardens
                (15, [-1, 1], 200., 24),  # ditto
                (12, [-1, 1], 115., 24),  # buildings, pavement, cafe tables
                (4, [-1, 1], 115., 24),  # front gardens / river
            ]
        elif 'Town02' in world_map.name:
            locations = [
                # (12, [-1, 1], 20., 12),  # top edge, valley, lane -1 is leftward
                (19, [-1], 20., 4),  # bottom edge, terraced houses, just after junction; lane -1 is rightward
                # (19, [1, -1], 120., 8),  # bottom edge, terraced houses, just after bend; lane 1 is leftward
            ]
        elif 'Town03' in world_map.name:
            locations = [
                (76, [3, 2, -1, -2], 175., 32),
                (76, [3, 2, -1, -2], 90., 24),
                (76, [3, 2, -1, -2], 230., 32),
                (90, [1, -1], 100., 20),
                (21, [1, -1], 90., 20),
            ]
        elif 'Town04' in world_map.name:
            locations = [
                (38, [6, 5, 4, 3, -1, -2, -3, -4], 100., 48),  # highway
            ]
        elif 'Town05' in world_map.name:
            locations = [
                (19, [-1, 1], 120., 12),  # houses / gardens
                (20, [-1, 1], 90., 12),  # ditto
                (34, [-3, -2, -1, 2, 3, 4], 120., 18),  # houses / verge / highway
                (38, [-3, -2, -1, 2, 3, 4], 270., 18),  # ditto
                (37, [-3, -2, -1, 2, 3, 4], 100., 18),  # ditto
                # (37, [-3, -2, -1, 2, 3, 4], 200., 18),  # ditto
                # (37, [-3, -2, -1, 2, 3, 4], 550., 18),  # ditto
                # (37, [-3, -2, -1, 2, 3, 4], 700., 18),  # ditto
                # (34, [-3, -2, -1, 2, 3, 4], 120., 48),  # houses / verge / highway
                # (38, [-3, -2, -1, 2, 3, 4], 270., 48),  # ditto
                # (37, [-3, -2, -1, 2, 3, 4], 100., 48),  # ditto
                # (37, [-3, -2, -1, 2, 3, 4], 200., 48),  # ditto
                # (37, [-3, -2, -1, 2, 3, 4], 550., 48),  # ditto
                # (37, [-3, -2, -1, 2, 3, 4], 700., 48),  # ditto
            ]
        else:
            raise RuntimeError('unknown map')

        road_id, lane_ids, base_s, number_of_vehicles = locations[np.random.randint(len(locations))]

        if number_of_vehicles % len(lane_ids) != 0:
            print('warning: number of vehicles does not divide number of lanes')
        vehicles_per_lane = number_of_vehicles // len(lane_ids)

        delta_ss = np.random.uniform(12., 16., size=[len(lane_ids), vehicles_per_lane])
        ss = np.cumsum(delta_ss, axis=1) + np.random.uniform(-10., 10., size=[len(lane_ids), 1])
        ss = ss - (np.min(ss) + np.max(ss)) / 2. + base_s
        if len(lane_ids) == 2:
            ss[np.asarray(lane_ids) < 0, :] -= 75

        def color_allowed(color):
            bits = list(map(int, color.split(',')))
            greyness_threshold = 15
            return abs(bits[1] - bits[0]) > greyness_threshold or abs(bits[2] - bits[0]) > greyness_threshold or (bits[0] > 220 and bits[1] > 220 and bits[2] > 220)

        def sample_car_blueprint():

            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                colors = blueprint.get_attribute('color').recommended_values
                colors = list(filter(color_allowed, colors))
                color = random.choice(colors)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            return blueprint

        vehicle_ids_by_lane = [[] for _ in lane_ids]
        for lane_index, lane_id in enumerate(lane_ids):

            for vehicle_index_in_lane in range(vehicles_per_lane):

                blueprint = sample_car_blueprint()

                waypoint = world_map.get_waypoint_xodr(road_id, lane_id, ss[lane_index, vehicle_index_in_lane])
                if waypoint is None:
                    print('warning: failed to get waypoint; probably ran out of road (road_id = {}, base_s = {:.1f})'.format(road_id, base_s))
                    continue
                transform = waypoint.transform

                transform.location.z += 1.25
                transform.location.x += np.random.normal(0., 0.2)
                transform.location.y += np.random.normal(0., 0.2)
                transform.rotation.yaw += np.random.normal(0., 2.)

                min_percent_speed_limit = 65
                max_percent_speed_limit = 85

                try:
                    vehicle = world.spawn_actor(blueprint, transform)
                    vehicle.set_autopilot(True)
                    traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(100 - max_percent_speed_limit, 100 - min_percent_speed_limit))
                    vehicle_ids_by_lane[lane_index].append(vehicle.id)
                    print('spawned ' + vehicle.type_id + ' at ' + str(transform.location))
                except RuntimeError as e:
                    print('warning: failed to spawn vehicle: ' + str(e))

        all_vehicle_ids = sum(vehicle_ids_by_lane, [])
        print('spawned {} vehicles'.format(len(all_vehicle_ids)))

        ego_lane_index = 0
        ego_vehicle_id = vehicle_ids_by_lane[ego_lane_index][len(vehicle_ids_by_lane[ego_lane_index]) // 2]
        ego_vehicle = world.get_actor(ego_vehicle_id)
        traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 100 - max_percent_speed_limit)
        traffic_manager.distance_to_leading_vehicle(ego_vehicle, 1.)

        time.sleep(0.2)

        if len(lane_ids) == 1 and random.uniform(0., 1.) < 0.5:
            companion_vehicles = []
            ego_vehicle_transform = ego_vehicle.get_transform()
            companion_relative_velocity = 1. * (-1 if random.uniform(0., 1.) < 0.5 else 1)
            min_companion_distance = 6.
            max_companion_distance = 10.
            companion_distance = random.uniform(min_companion_distance, max_companion_distance)
            blueprint = sample_car_blueprint()
            transform = carla.Transform(
                carla.Location(x=ego_vehicle_transform.location.x - companion_distance, y=ego_vehicle_transform.location.y, z=1.25),
                ego_vehicle_transform.rotation
            )
            companion_vehicle = world.spawn_actor(blueprint, transform)
            companion_vehicles.append(companion_vehicle)
            all_vehicle_ids.append(companion_vehicle.id)
            print('spawned companion {} at {}'.format(companion_vehicle.type_id, transform.location))
        else:
            companion_vehicles = []

        def get_camera_transform(yaw, pitch, centre_offset):
            return carla.Transform(
                carla.Location(x=-6. * math.cos(math.radians(yaw)) + centre_offset[0], y=-3. * math.sin(math.radians(yaw)) + centre_offset[1], z=1.),
                carla.Rotation(pitch=pitch, yaw=yaw, roll=0.)
            )

        min_yaw = 50
        max_yaw = 130
        if lane_ids[ego_lane_index] > 0:
            min_yaw, max_yaw = -max_yaw, -min_yaw
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '320')
        camera_bp.set_attribute('image_size_y', '240')
        camera_bp.set_attribute('fov', '100')
        camera_yaw = random.uniform(min_yaw, max_yaw)
        camera_pitch = -5.
        if len(companion_vehicles) == 0:
            camera_centre_offset_x = random.uniform(-2., 2.)  # +ve x = forward
            camera_centre_offset_y = random.uniform(-1.5, 0.25)  # -ve y = leftward
        else:
            camera_centre_offset_x = random.uniform(-6., -2.)  # +ve x = forward
            camera_centre_offset_y = random.uniform(-2.5, -0.75)  # -ve y = leftward
        if lane_ids[ego_lane_index] > 0:
            camera_centre_offset_y = -camera_centre_offset_y
        camera_centre_offset = [camera_centre_offset_x, camera_centre_offset_y]
        rgb_camera = world.spawn_actor(camera_bp, get_camera_transform(camera_yaw, camera_pitch, camera_centre_offset), attach_to=ego_vehicle)
        time.sleep(0.2)
        print('spawned ' + rgb_camera.type_id + ' at ' + str(rgb_camera.get_transform()) + ' attached to ego-vehicle with transform ' + str(ego_vehicle.get_transform()))

        camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '320')
        camera_bp.set_attribute('image_size_y', '240')
        camera_bp.set_attribute('fov', '100')
        identity_transform = carla.Transform(carla.Location(x=0., y=0., z=0.), carla.Rotation(pitch=0., yaw=0., roll=0.))
        ss_camera = world.spawn_actor(camera_bp, identity_transform, attach_to=rgb_camera)
        print('spawned ' + ss_camera.type_id)

        camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
        camera_bp.set_attribute('image_size_x', '320')
        camera_bp.set_attribute('image_size_y', '240')
        camera_bp.set_attribute('fov', '100')
        depth_camera = world.spawn_actor(camera_bp, identity_transform, attach_to=rgb_camera)
        print('spawned ' + depth_camera.type_id)

        first_lane_change_frame = 5
        second_lane_change_frame = 50
        lane_change_sense = random.uniform(0., 1.) < 0.5

        yaw_direction = 1. if random.uniform(0., 1.) < 0.5 else -1.

        def location_to_np(location):
            return np.float32([location.x, location.y, location.z])

        print('collecting frames...', end='', flush=True)

        camera_transforms = []
        vehicle_bboxes_by_frame = []  # [frame], [vehicle], bbox-corner, x/y/z
        with CarlaSyncMode(world, rgb_camera, ss_camera, depth_camera, fps=20, traffic_manager=traffic_manager) as sync_mode:

            frame_counter = 0  # this counts frames at the fps rate specified above
            frame_frequency = 5  # further downsampling rate applied to these frames
            previous_location = np.float32([0., 0., 0.])
            ego_vehicle_initial_yaw = None  # we use the yaw at the moment we actually start capturing
            while frame_counter // frame_frequency <= args.skip_frames + args.frames_per_episode:

                snapshot, image_rgb, image_ss, image_depth = sync_mode.tick(timeout=2.)

                ego_vehicle_transform = ego_vehicle.get_transform()

                if frame_counter % frame_frequency == 0:

                    current_location = location_to_np(ego_vehicle_transform.location)
                    if ego_vehicle.get_traffic_light_state() == carla.TrafficLightState.Red:

                        # print('skipping frame {}: red traffic light'.format(frame_counter // frame_frequency))
                        pass

                    else:

                        if frame_counter // frame_frequency >= args.skip_frames:

                            current_yaw = ego_vehicle_transform.rotation.yaw
                            if ego_vehicle_initial_yaw is None:
                                ego_vehicle_initial_yaw = current_yaw
                            elif abs(current_yaw - ego_vehicle_initial_yaw) > 5:
                                print('note: ending sequence due to vehicle turning')
                                break

                            image_rgb.save_to_disk('{}_{:03d}.jpg'.format(output_prefix, frame_counter // frame_frequency - args.skip_frames), carla.ColorConverter.Raw)
                            image_ss.save_to_disk('{}_{:03d}_ss.png'.format(output_prefix, frame_counter // frame_frequency - args.skip_frames), carla.ColorConverter.CityScapesPalette)
                            image_depth.save_to_disk('{}_{:03d}_depth.png'.format(output_prefix, frame_counter // frame_frequency - args.skip_frames), carla.ColorConverter.Raw)

                            current_camera_transform = rgb_camera.get_transform()
                            camera_transforms.append(current_camera_transform)

                            some_vehicle_unavailable = False
                            vehicle_bboxes_for_frame = []
                            for vehicle_id in all_vehicle_ids:
                                vehicle = world.get_actor(vehicle_id)
                                if vehicle is None:
                                    some_vehicle_unavailable = True
                                    break
                                bbox = vehicle.bounding_box
                                vehicle_bboxes_for_frame.append(
                                    tuple(map(location_to_np, bbox.get_world_vertices(vehicle.get_transform())))
                                )
                            if some_vehicle_unavailable:
                                print('warning: ending sequence due to missing vehicle')
                                break
                            vehicle_bboxes_by_frame.append(vehicle_bboxes_for_frame)

                        if len(lane_ids) > 2:
                            if frame_counter // frame_frequency - args.skip_frames == first_lane_change_frame:
                                traffic_manager.force_lane_change(ego_vehicle, lane_change_sense)
                            elif frame_counter // frame_frequency - args.skip_frames == second_lane_change_frame:
                                traffic_manager.force_lane_change(ego_vehicle, not lane_change_sense)

                camera_yaw += snapshot.timestamp.delta_seconds * 20. * yaw_direction
                if (yaw_direction < 0 and camera_yaw < min_yaw) or (yaw_direction > 0 and camera_yaw > max_yaw):
                    yaw_direction = -yaw_direction
                rgb_camera.set_transform(get_camera_transform(camera_yaw, camera_pitch, camera_centre_offset))

                if frame_counter // frame_frequency >= args.skip_frames - 1:
                    for companion_vehicle_index, companion_vehicle in enumerate(companion_vehicles):
                        companion_vehicle.set_transform(carla.Transform(
                            carla.Location(
                                ego_vehicle_transform.location.x - companion_distance,
                                ego_vehicle_transform.location.y,
                                companion_vehicle.get_location().z
                            ),
                            ego_vehicle_transform.rotation
                        ))
                        companion_distance += snapshot.timestamp.delta_seconds * 1. * companion_relative_velocity
                        if (companion_relative_velocity < 0 and companion_distance < min_companion_distance) or (companion_relative_velocity > 0 and companion_distance > max_companion_distance):
                            companion_relative_velocity = -companion_relative_velocity

                frame_counter += 1
                previous_location = current_location

        print(' done')
        
        with open(output_prefix + '_camera.txt', 'w') as f:
            for transform in camera_transforms:
                f.write('{} {} {} {} {} {}\n'.format(transform.location.x, transform.location.y, transform.location.z, transform.rotation.yaw, transform.rotation.pitch, transform.rotation.roll))
        with open(output_prefix + '_bboxes.pkl', 'wb') as f:
            pickle.dump(vehicle_bboxes_by_frame, f)

    finally:

        if depth_camera is not None:
            depth_camera.destroy()
        if ss_camera is not None:
            ss_camera.destroy()
        if rgb_camera is not None:
            rgb_camera.destroy()
        for vehicle_id in all_vehicle_ids:
            maybe_vehicle = world.get_actor(vehicle_id)
            if maybe_vehicle is not None:
                maybe_vehicle.destroy()
            else:
                print('warning: vehicle #{} disappeared prematurely'.format(vehicle_id))

        time.sleep(0.5)


if __name__ == '__main__':

    main()
