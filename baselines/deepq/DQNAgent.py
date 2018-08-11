import numpy as np

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.schedules import LinearSchedule
from baselines.deepq.utils import PlaceholderTfInput

import baselines.common.tf_util as U

import tensorflow as tf
import tensorflow.contrib.layers as layers


class TrafficTfInput(PlaceholderTfInput):
    def __init__(self, shape, name=None):
        """
        Parameters
        ----------
        shape: [int]
            shape of the tensor.
        name: str
            name of the underlying placeholder
        """

        super().__init__(tf.placeholder(tf.float32, [None] + list(shape), name=name))
        self._shape = shape


def model(img_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)

        with tf.variable_scope("action_value"):
            value_out = layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
            value_out = tf.nn.relu(value_out)
            value_out = layers.fully_connected(value_out, num_outputs=num_actions, activation_fn=None)
        return value_out


def dueling_model(img_in, num_actions, scope, reuse=False):
    """As described in https://arxiv.org/abs/1511.06581"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)

        with tf.variable_scope("state_value"):
            state_hidden = layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
            state_hidden = tf.nn.relu(state_hidden)
            state_score = layers.fully_connected(state_hidden, num_outputs=1, activation_fn=None)
        with tf.variable_scope("action_value"):
            actions_hidden = layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
            actions_hidden = tf.nn.relu(actions_hidden)
            action_scores = layers.fully_connected(actions_hidden, num_outputs=num_actions, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores = action_scores - tf.expand_dims(action_scores_mean, 1)
        return state_score + action_scores


class DQNAgent:

    def __init__(self, identifier, actions, observation_shape, num_steps, x=0.0, y=0.0):
        self.id = identifier
        self.actions = actions
        self.x = x
        self.y = y
        self.yellow_steps = 0
        self.postponed_action = None
        self.obs = None
        self.current_action = None
        self.weights = np.ones(32)
        self.td_errors = np.ones(32)

        self.pre_train = 2500
        self.prioritized = True
        self.prioritized_eps = 1e-4
        self.batch_size = 32
        self.buffer_size = 30000
        self.learning_freq = 500
        self.target_update = 5000

        # Create all the functions necessary to train the model
        self.act, self.train, self.update_target, self.debug = deepq.build_train(
            make_obs_ph=lambda name: TrafficTfInput(observation_shape, name=name),
            q_func=dueling_model,
            num_actions=len(actions),
            optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4),
            gamma=0.99,
            double_q=True,
            scope="deepq" + identifier
        )

        # Create the replay buffer
        #if self.prioritized:
        #    self.replay_buffer = PrioritizedReplayBuffer(size=self.buffer_size, alpha=0.6)
        #    self.beta_schedule = LinearSchedule(num_steps // 4, initial_p=0.4, final_p=1.0)
        #else:
        #    self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        self.exploration = LinearSchedule(schedule_timesteps=int(num_steps * 0.1), initial_p=1.0, final_p=0.01)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        self.update_target()

    def take_action(self, t):
        if self.postponed_action is None:
            # Take action and update exploration to the newest value
            action = self.act(np.array(self.obs)[None], update_eps=self.exploration.value(t))[0]
        else:
            # Take action postponed by yellow light transition
            action = self.postponed_action
            self.postponed_action = None

        return action

    def store(self, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(self.obs, self.current_action, rew, new_obs, float(done))

    def learn(self, t):
        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        if t > self.pre_train:
            if self.prioritized:
                experience = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, self.weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                self.weights = np.ones_like(rewards)

            # Minimize the error in Bellman's equation and compute TD-error
            self.td_errors = self.train(obses_t, actions, rewards, obses_tp1, dones, self.weights)

            # Update the priorities in the replay buffer
            if self.prioritized:
                new_priorities = np.abs(self.td_errors) + self.prioritized_eps
                self.replay_buffer.update_priorities(batch_idxes, new_priorities)

        self.update_target(t)

    def update_target(self, t):
        # Update target network periodically.
        if t % self.target_update == 0:
            self.update_target()

    def add_fingerprint_to_obs(self, obs, weights, identifier, td_errors):
        idx = 0

        for w in weights:
            obs[2, identifier, idx] = w
            idx += 1

        for td in td_errors:
            obs[2, identifier, idx] = td
            idx += 1

        return obs

    def add_fingerprint(self, weights, identifier, td_errors):
        self.obs = self.add_fingerprint_to_obs(self.obs, weights, identifier, td_errors)
