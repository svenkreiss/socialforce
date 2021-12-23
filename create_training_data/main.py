"""
Simulate crowd flow by Social Force Model
Provide training data for GNS
"""

import math
from os import times
import random
import json
from typing import List, Tuple
from numpy.core.defchararray import array
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import socialforce

tf.enable_eager_execution()

Position = Tuple[float, float]
State = np.ndarray
States = np.ndarray  # time-series data of State
Space = List[np.ndarray]


def create_random_agent_pos() -> Position:
    """Return initial agent position"""
    x_lower_lim = -10.0
    x_upper_lim = -5.0
    y_lower_lim = -5.0
    y_upper_lim = 10.0
    rand_x = random.uniform(x_lower_lim, x_upper_lim)
    rand_y = random.uniform(y_lower_lim, y_upper_lim)
    return rand_x, rand_y


def get_xy_from_rd(radius: float, degree: float) -> Position:
    """Return XY coordinates from polar coodinates"""
    rad = math.radians(degree)
    x = radius * math.cos(rad)
    y = radius * math.sin(rad)
    return x, y


def create_circular_random_agent_pos() -> Position:
    """Return initial agent position in a circle"""
    lower_radius = 50.0
    upper_radius = 80.0
    lower_angle = 0.0
    upper_angle = 360.0
    rand_radius = random.uniform(lower_radius, upper_radius)
    rand_angle = random.uniform(lower_angle, upper_angle)
    return get_xy_from_rd(rand_radius, rand_angle)


def setup_state(agent_num: int, destination: Tuple[float, float]) -> State:
    """Return initial agent state"""
    return np.array([(*create_circular_random_agent_pos(), 1.0, 0.0, *destination) for _ in range(agent_num)])


def add_space(space: Space, obstacle: Tuple[Position, Position]) -> Space:
    """Add obstacle to the space"""
    space.append(np.linspace(*obstacle, 100))
    return space


def add_hole(space: Space, hole: Tuple[float, float], x_pos: float) -> Space:
    """Add obstacle with a hole to the space"""
    add_space(space, ((x_pos, -10), (x_pos, hole[0])))
    add_space(space, ((x_pos, hole[1]), (x_pos, 10)))
    return space


def _bytes_feature(value):
    """string / byte 型から byte_list を返す"""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """float / double 型から float_list を返す"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """bool / enum / int / uint 型から Int64_list を返す"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def visualize(states: States, space: Space, output_filename: str) -> None:
    """Visualize social force simulation"""
    with socialforce.show.animation(
            len(states),
            output_filename,
            writer='imagemagick') as context:
        ax = context['ax']
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim(-200, 150)
        ax.set_ylim(-100, 100)

        for s in space:
            ax.plot(s[:, 0], s[:, 1], 'o', color='black', markersize=2.5)

        actors = []
        for ped in range(states.shape[1]):
            speed = np.linalg.norm(states[0, ped, 2:4])
            radius = 0.2 + speed / 2.0 * 0.3
            p = plt.Circle(states[0, ped, 0:2], radius=radius,
                           facecolor='black' if states[0,
                                                       ped, 4] > 0 else 'white',
                           edgecolor='black')
            actors.append(p)
            ax.add_patch(p)

        def update(i):
            for ped, p in enumerate(actors):
                # p.set_data(states[i:i+5, ped, 0], states[i:i+5, ped, 1])
                p.center = states[i, ped, 0:2]
                speed = np.linalg.norm(states[i, ped, 2:4])
                p.set_radius(0.2 + speed / 2.0 * 0.3)

        context['update_function'] = update


def setup(simulation_length: int, destination: Tuple[float, float]) -> Tuple[States, Space]:
    """Set up space and states"""
    # agent_num = random.randrange(3, 8)
    agent_num = 230
    initial_state = setup_state(agent_num, destination)
    # y_min = -2.0
    # y_max = 2.0
    # hole_width_min = 1.4
    # hole_width_max = 2.0
    # hole_width = random.uniform(hole_width_min, hole_width_max)
    # hole_start = random.uniform(y_min, y_max - hole_width)
    # hole_end = hole_start + hole_width
    # hole = (hole_start, hole_end)
    # x_pos = random.uniform(-3.0, 3.0)
    space = []
    # space = add_hole(space, hole, x_pos)
    s = socialforce.Simulator(
        initial_state, socialforce.PedSpacePotential(space))
    states = np.stack([s.step().state.copy()
                      for _ in range(simulation_length)])
    return states, space


def create_obstacle_agents(space: Space, simulation_length: int) -> np.ndarray:
    """Return obstacle agents' position sequence as np.float32"""
    obstacle_agents = np.array(space).astype(np.float32).reshape((-1, 2))
    return np.array([obstacle_agents] * simulation_length)


def create_moving_agents(states: States) -> np.ndarray:
    """Return agents' position sequence as np.float32"""
    return states[:, :, 0:2].astype(np.float32)


def create_json(metadata: dict) -> None:
    """Create json formatted metadata file"""
    file_path = "/tmp/datasets/SFM/metadata.json"
    with open(file_path, 'w') as f:
        json.dump(metadata, f)


def create_tfrecord(position_list: np.ndarray, particle_type: np.ndarray, destination_x: np.ndarray, destination_y: np.ndarray) -> None:
    """Create tfrecord formatted feature data file"""
    # file name is train.tfrecord/test.tfrecord/valid.tfrecord
    file_path = "/tmp/datasets/SFM/train.tfrecord"
    with tf.python_io.TFRecordWriter(file_path) as w:
        context = tf.train.Features(feature={
            'particle_type': _bytes_feature(particle_type.tobytes()),
            'destination_x': _bytes_feature(destination_x.tobytes()),
            'destination_y': _bytes_feature(destination_y.tobytes()),
            'key': _int64_feature(np.int64(0))
        })
        description_feature = [
            _bytes_feature(v.tobytes()) for v in position_list
        ]
        feature_lists = tf.train.FeatureLists(feature_list={
            "position": tf.train.FeatureList(feature=description_feature)
        })

        sequence_example = tf.train.SequenceExample(context=context,
                                                    feature_lists=feature_lists)
        w.write(sequence_example.SerializeToString())


def main():
    """Output multiple simulation results and each animations"""
    output_num = 1
    simulation_length = 100
    destination = (0, 0)
    timestep_num_list = np.array([])
    agents_num_list = np.array([])
    vel_mean_list = np.empty((0,2))
    vel_var_list = np.empty((0,2))
    acc_mean_list = np.empty((0,2))
    acc_var_list = np.empty((0,2))
    for i in range(output_num):
        print(f"Dealing with ({i + 1}/{output_num}) simulation")
        states, space = setup(simulation_length, destination)
        timestep_num_list = np.append(timestep_num_list, states.shape[0])
        agents_num_list = np.append(agents_num_list, states.shape[1])
        vel_mean_list = np.append(
            vel_mean_list,
            np.array([np.mean(states[:, :, 2:4], axis=(0, 1))]),
            axis=0
        )
        vel_var_list = np.append(
            vel_var_list,
            np.array([np.mean(
                (states[:, :, 2:4] - vel_mean_list[i]) ** 2,
                axis=(0, 1))]),
            axis=0
        )
        first_step_vel_mean = np.mean(states[0, :, 2:4], axis=0)
        final_step_vel_mean = np.mean(states[-1, :, 2:4], axis=0)
        try:
            # acc_mean = (\bar{v_(L+1)} - \bar{v_1}) / L
            acc_mean_list = np.append(acc_mean_list, np.array(
                [(final_step_vel_mean - first_step_vel_mean) / (timestep_num_list[i] - 1)]), axis=0)
        except ZeroDivisionError as e:
            print(e)
            print("Simulation timestep should be more than 1.")
        acc_var_list = np.append(
            acc_var_list,
            np.array([np.mean(
                (np.diff(states[:, :, 2:4], axis=0) - acc_mean_list[i]) ** 2,
                axis=(0, 1))]),
            axis=0
        )
        obstacle_agents = create_obstacle_agents(space, simulation_length)
        moving_agents = create_moving_agents(states)
        agents = np.concatenate([obstacle_agents, moving_agents], axis=1)
        print(obstacle_agents.shape[1])
        print(moving_agents.shape[1])
        obstacle_row = [np.int64(3)] * obstacle_agents.shape[1]
        moving_row = [np.int64(8)] * moving_agents.shape[1]
        agents_row = obstacle_row + moving_row
        print(agents.shape)
        destination_x = [np.float32(destination[0])] * agents.shape[1]
        destination_y = [np.float32(destination[1])] * agents.shape[1]
        create_tfrecord(agents, np.array(agents_row), np.array(destination_x), np.array(destination_y))
        visualize(states, space, f'create_training_data/img/output{str(i + 1)}.gif')
    vel_mean = np.sum(
        vel_mean_list * timestep_num_list.reshape((-1, 1)) * agents_num_list.reshape((-1, 1)),
        axis=0) / np.sum(timestep_num_list * agents_num_list)
    acc_mean = np.sum(
        acc_mean_list * (timestep_num_list.reshape((-1, 1)) - 1) * agents_num_list.reshape((-1, 1)),
        axis=0) / np.sum((timestep_num_list - 1) * agents_num_list)
    vel_var = np.sum(
        (vel_var_list + vel_mean_list ** 2) * timestep_num_list.reshape((-1, 1)) *
        agents_num_list.reshape((-1, 1)), axis=0) / np.sum(timestep_num_list * agents_num_list)
    acc_var = np.sum(
        (acc_var_list + acc_mean_list ** 2) * (timestep_num_list.reshape((-1, 1)) - 1) *
        agents_num_list.reshape((-1, 1)), axis=0) / np.sum((timestep_num_list - 1) * agents_num_list)
    vel_std = np.sqrt(vel_var)
    acc_std = np.sqrt(acc_var)
    metadata: dict = {
        'bounds': [[-150.0, 150.0], [-100.0, 100.0]],
        'sequence_length': simulation_length - 1,
        'default_connectivity_radius': 0.2,
        'dim': 2,
        'dt': 0.4,
        'vel_mean': vel_mean.tolist(),
        'vel_std': vel_std.tolist(),
        'acc_mean': acc_mean.tolist(),
        'acc_std': acc_std.tolist()
    }
    print(metadata)
    create_json(metadata)


if __name__ == '__main__':
    main()
