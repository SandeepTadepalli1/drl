import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.schedules import LinearSchedule
from traci_tls.trafficEnvironment import TrafficEnv
from baselines.deepq.utils import PlaceholderTfInput


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


def plot_rewards(path = "/Users/jeancarlo/PycharmProjects/thesis/"):
    import matplotlib.pyplot as plt
    import datetime

    plt.clf()
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.ylabel('Reward')
    plt.xlabel('Training Epochs')
    plt.savefig(path + 'images/rew/' + nameProc + str(datetime.datetime.now()).split('.')[0] + '.eps', format='eps', dpi=1000)

    with open(path + 'images/files/' + nameProc + 'REW.csv', 'a') as file:
        file.write(";".join(map(str, episode_rewards[-save_freq:])))

    plt.clf()
    plt.plot(np.arange(len(waiting_time_hist)), waiting_time_hist)
    plt.ylabel('Average Waiting Time')
    plt.xlabel('Training Epochs')
    plt.savefig(path + 'images/awt/' + nameProc + str(datetime.datetime.now()).split('.')[0] + '.eps', format='eps', dpi=1000)

    with open(path + 'images/files/' + nameProc + 'AWT.csv', 'a') as file:
        file.write(";".join(map(str, waiting_time_hist[-save_freq:])))

    plt.clf()
    plt.plot(np.arange(len(travel_time_hist)), travel_time_hist)
    plt.ylabel('Average Travel Time')
    plt.xlabel('Training Epochs')
    plt.savefig(path + 'images/att/' + nameProc + str(datetime.datetime.now()).split('.')[0] + '.eps', format='eps', dpi=1000)

    with open(path + 'images/files/' + nameProc + 'ATT.csv', 'a') as file:
        file.write(";".join(map(str, travel_time_hist[-save_freq:])))


if __name__ == '__main__':
    tf.set_random_seed(0)

    with U.make_session() as sess:
        save_freq = 25
        nameProc = "duelingdoubledqnPriori"
        simulation_time = 3600  # one simulated hour
        num_steps = 1000 * simulation_time
        pre_train = 2500
        prioritized = True
        prioritized_eps = 1e-4
        batch_size = 32
        buffer_size = 50000
        learning_freq = 500
        target_update = 5000

        # Create the environment
        env = TrafficEnv(simulation_time, nameProc)

        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: TrafficTfInput(env.observation.shape, name=name),
            q_func=dueling_model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4),
            gamma=0.99,
            double_q=True
        )

        # writer = tf.summary.FileWriter("/Users/jeancarlo/PycharmProjects/thesis/logs/", sess.graph)
        # writer.close()

        # Create the replay buffer
        if prioritized:
            replay_buffer = PrioritizedReplayBuffer(size=buffer_size, alpha=0.6)
            beta_schedule = LinearSchedule(num_steps // 4, initial_p=0.4, final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(buffer_size)

        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=int(num_steps * 0.1), initial_p=1.0, final_p=0.01)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        waiting_time_hist = []
        travel_time_hist = []
        obs = env.reset()
        for t in range(0, num_steps):
            # Take action and update exploration to the newest value
            action = act(np.array(obs)[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(False))
            obs = new_obs
            episode_rewards[-1] += rew

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if t > pre_train:
                if prioritized:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights = np.ones_like(rewards)

                # Minimize the error in Bellman's equation and compute TD-error
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)

                # Update the priorities in the replay buffer
                if prioritized:
                    new_priorities = np.abs(td_errors) + prioritized_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            # Update target network periodically.
            if t % target_update == 0:
                update_target()

            if done:
                print("Done Episode " + str(len(episode_rewards)))
                waiting_time_hist.append(env.get_average_waiting_time())
                travel_time_hist.append(env.get_average_travel_time())

                if len(episode_rewards) % save_freq == 0 and t > 0:
                    print("Done Episode " + str(len(episode_rewards)))
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", len(episode_rewards))
                    logger.record_tabular("episode reward", episode_rewards[-1])
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    if prioritized:
                        logger.record_tabular("max priority", replay_buffer._max_priority)
                    logger.dump_tabular()
                    plot_rewards()

                obs = env.reset()
                episode_rewards.append(0.0)
