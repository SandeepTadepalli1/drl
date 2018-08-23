import numpy as np
import tensorflow as tf
import gc

from baselines import logger
from traci_tls.trafficEnvironment import TrafficEnv
from baselines.deepq.DQNAgent import DQNAgent

import baselines.common.tf_util as U


def plot_rewards(path="/Users/jeancarlo/PycharmProjects/thesis/"):
    import matplotlib.pyplot as plt
    import datetime

    plt.clf()
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.ylabel('Reward')
    plt.xlabel('Training Epochs')
    plt.savefig(path + 'images/rew/' + name_process + str(datetime.datetime.now()).split('.')[0] + '.eps', format='eps',
                dpi=1000)

    with open(path + 'images/files/' + name_process + 'REW.csv', 'a') as file:
        file.write(";".join(map(str, episode_rewards[-save_freq:])))

    plt.clf()
    plt.plot(np.arange(len(waiting_time_hist)), waiting_time_hist)
    plt.ylabel('Average Waiting Time')
    plt.xlabel('Training Epochs')
    plt.savefig(path + 'images/awt/' + name_process + str(datetime.datetime.now()).split('.')[0] + '.eps', format='eps',
                dpi=1000)

    with open(path + 'images/files/' + name_process + 'AWT.csv', 'a') as file:
        file.write(";".join(map(str, waiting_time_hist[-save_freq:])))

    plt.clf()
    plt.plot(np.arange(len(travel_time_hist)), travel_time_hist)
    plt.ylabel('Average Travel Time')
    plt.xlabel('Training Epochs')
    plt.savefig(path + 'images/att/' + name_process + str(datetime.datetime.now()).split('.')[0] + '.eps', format='eps',
                dpi=1000)

    with open(path + 'images/files/' + name_process + 'ATT.csv', 'a') as file:
        file.write(";".join(map(str, travel_time_hist[-save_freq:])))


def reset():
    env.reset()
    for a in agents:
        a.obs = env.choose_next_observation(a.x, a.y)  # result_obs[indexObs]

        #index = 0
        #for na in agents:
        #    if na.id != a.id:
        #        a.add_fingerprint(na.weights, index, na.td_errors)
        #        index += 1


if __name__ == '__main__':
    tf.set_random_seed(0)

    with U.make_session() as sess:
        save_freq = 25
        name_process = "duelingdoubledqnPriori"  # Experience Replay Memory enabled
        simulation_time = 3600  # one simulated hour
        num_steps = 1000 * simulation_time

        # Create the environment
        env = TrafficEnv(simulation_time, name_process)

        agents = [
            DQNAgent("0", [0, 4], env.shape, x=256.0, y=256.0, num_steps=num_steps),  # right upper
            DQNAgent("5", [0, 4], env.shape, x=0.0, y=256.0, num_steps=num_steps),  # left upper
            DQNAgent("8", [0, 4], env.shape, x=256.0, y=0.0, num_steps=num_steps),  # right lower
            # DQNAgent("12", [0, 4], env.observationDim.shape, x=0.0, y=0.0, num_steps=num_steps)      # left lower
        ]

        # writer = tf.summary.FileWriter("/Users/jeancarlo/PycharmProjects/thesis/logs/", sess.graph)
        # writer.close()
        # saver = tf.train.Saver()

        episode_rewards = [0.0]
        waiting_time_hist = []
        travel_time_hist = []
        reset()

        for t in range(0, num_steps):
            for agent in agents:
                if agent.yellow_steps > 0:
                    # Still in yellow transition
                    agent.yellow_steps -= 1
                    continue

                agent.current_action = agent.take_action(t)
                should_postpone_action = env.set_phase(agent.current_action, agent.id, agent.actions)

                if should_postpone_action:
                    # Postpone action until yellow transition finishes
                    agent.yellow_steps = 3
                    agent.postponed_action = agent.current_action
                    continue

            env.make_step()
            reward = env.get_reward()  # Gets global reward r_{t}
            episode_rewards[-1] += reward
            done = env.is_done()

            for agent in agents:
                if agent.yellow_steps == 0:
                    new_obs = env.choose_next_observation(agent.x, agent.y)

                    #idx = 0
                    #for n in agents:
                    #    if n.id != agent.id:
                    #        new_obs = agent.add_fingerprint_to_obs(new_obs, n.weights, idx, n.td_errors)
                    #        idx += 1

                    #agent.store(reward, new_obs, done)
                    agent.obs = new_obs
                    #agent.learn(t)
                    agent.update_target_network(t)

            gc.collect()

            if done:
                env.close()
                print("Done Episode " + str(len(episode_rewards)))
                waiting_time_hist.append(env.get_average_waiting_time())
                travel_time_hist.append(env.get_average_travel_time())

                if len(episode_rewards) % save_freq == 0 and t > 0:
                    print("Done Episode " + str(len(episode_rewards)))
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", len(episode_rewards))
                    logger.record_tabular("episode reward", episode_rewards[-1])
                    logger.record_tabular("% time spent exploring", int(100 * agents[0].exploration.value(t)))
                    logger.dump_tabular()
                    plot_rewards()

                reset()
                episode_rewards.append(0.0)

        # Save the variables to disk.
        # save_path = saver.save(sess, "/tmp/model.ckpt")
        # print("Model saved in path: %s" % save_path)
