"""
Simulate crowd flow by Social Force Model
Provide training data for GNS
"""

import random
import json
from typing import List, Tuple
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import socialforce

Position = Tuple[float, float]
State = np.ndarray
States = np.ndarray # time-series data of State
Space = List[np.ndarray]

# Create a description of the features.
_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}


def random_agent_pos() -> Position:
    """Return initial agent position"""
    x_lower_lim = -10.0
    x_upper_lim = -5.0
    y_lower_lim = -5.0
    y_upper_lim = 10.0
    rand_x = random.uniform(x_lower_lim, x_upper_lim)
    rand_y = random.uniform(y_lower_lim, y_upper_lim)
    return rand_x, rand_y


def setup_state(agent_num: int) -> State:
    """Return initial agent state"""
    return np.array([(*random_agent_pos(), 1.0, 0.0, 10.0, 0.0) for _ in range(agent_num)])


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
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
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


def setup(simulation_length: int) -> Tuple[States, Space]:
    """Set up space and states"""
    agent_num = random.randrange(3, 8)
    initial_state = setup_state(agent_num)
    y_min = -2.0
    y_max = 2.0
    hole_width_min = 1.4
    hole_width_max = 2.0
    hole_width = random.uniform(hole_width_min, hole_width_max)
    hole_start = random.uniform(y_min, y_max - hole_width)
    hole_end = hole_start + hole_width
    hole = (hole_start, hole_end)
    x_pos = random.uniform(-3.0, 3.0)
    space = []
    space = add_hole(space, hole, x_pos)
    s = socialforce.Simulator(
        initial_state, socialforce.PedSpacePotential(space))
    states = np.stack([s.step().state.copy() for _ in range(simulation_length)])
    return states, space


def create_obstacle_agents(space: Space, simulation_length: int) -> np.ndarray:
    """Return obstacle agents' position sequence as np.float32"""
    obstacle_agents = np.array(space).astype(np.float32).reshape((-1, 2))
    return np.array([obstacle_agents] * simulation_length)


def create_moving_agents(states: States) -> np.ndarray:
    """Return agents' position sequence as np.float32"""
    return states[:,:, 0:2].astype(np.float32)


def create_json(metadata: dict) -> None:
    file_path = "/tmp/datasets/SocialForceModel/metadata.json"
    with open(file_path, 'w') as f:
        json.dump(metadata, f)


def create_tfrecord(position_list: np.ndarray) -> None:
    file_path = "/tmp/datasets/SocialForceModel/train.tfrecord"
    with tf.python_io.TFRecordWriter(file_path) as w:
        context = tf.train.Features(feature={
            'key': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
            'particle_type': tf.train.Feature(bytes_list=tf.train.BytesList(value=[]))
        })
        description_feature = [tf.train.Feature(
            int64_list=tf.train.Int64List(value=[v])) for v in position_list
        ]
        feature_lists = tf.train.FeatureLists(feature_list={
            "position": tf.train.FeatureList(feature=description_feature)
        })

        sequence_example = tf.train.SequenceExample(context=context,
                                                    feature_lists=feature_lists)
        w.write(sequence_example.SerializeToString())

def main():
    """Output multiple simulation results and each animations"""
    output_num = 6
    base_file_name = 'output'
    simulation_length = 150
    vel_mean_list = np.array([[]])
    agents_list = np.array([])
    for i in range(1, output_num + 1):
        print(f"creating images({i}/{output_num})")
        states, space = setup(simulation_length)
        np.append(vel_mean_list, np.mean(states, axis=2), axis=0)
        np.append(agents_list, states.shape(1))
        obstacle_agents = create_obstacle_agents(space, simulation_length)
        moving_agent = create_moving_agents(states)
        visualize(states, space, f'mycode/img/{base_file_name + str(i)}.gif')
    acc_mean_list = np.diff(vel_mean_list, axis=0)
    vel_mean = np.mean(vel_mean_list, axis=1)
    acc_mean = np.mean(acc_mean_list, axis=1)
    vel_std = np.mean((vel_mean_list - vel_mean) ** 2, axis=3)
    acc_std = np.mean((acc_mean_list - acc_mean) ** 2, axis=3)
    


if __name__ == '__main__':
    main()