import numpy as np

from baselines import logger
from traci_tls.trafficEnvironment import TrafficEnv


def plot_rewards(path="/Users/jeancarlo/PycharmProjects/thesis/"):
    import matplotlib.pyplot as plt
    import datetime

    plt.clf()
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.ylabel('Reward')
    plt.xlabel('Training Epochs')
    plt.savefig(path + 'images/rew/' + name_process + str(datetime.datetime.now()).split('.')[0] + '.eps', format='eps', dpi=1000)

    with open(path + 'images/files/' + name_process + 'REW.csv', 'a') as file:
        file.write(";".join(map(str, episode_rewards[-save_freq:])))

    plt.clf()
    plt.plot(np.arange(len(waiting_time_hist)), waiting_time_hist)
    plt.ylabel('Average Waiting Time')
    plt.xlabel('Training Epochs')
    plt.savefig(path + 'images/awt/' + name_process + str(datetime.datetime.now()).split('.')[0] + '.eps', format='eps', dpi=1000)

    with open(path + 'images/files/' + name_process + 'AWT.csv', 'a') as file:
        file.write(";".join(map(str, waiting_time_hist[-save_freq:])))

    plt.clf()
    plt.plot(np.arange(len(travel_time_hist)), travel_time_hist)
    plt.ylabel('Average Travel Time')
    plt.xlabel('Training Epochs')
    plt.savefig(path + 'images/att/' + name_process + str(datetime.datetime.now()).split('.')[0] + '.eps', format='eps', dpi=1000)

    with open(path + 'images/files/' + name_process + 'ATT.csv', 'a') as file:
        file.write(";".join(map(str, travel_time_hist[-save_freq:])))


def reset():
    env.reset()


if __name__ == '__main__':

    save_freq = 25
    name_process = "dqn"
    simulation_time = 3600  # one simulated hour
    num_steps = 1000 * simulation_time

    # Create the environment
    env = TrafficEnv(simulation_time, name_process)

    episode_rewards = [0.0]
    waiting_time_hist = []
    travel_time_hist = []
    reset()

    for t in range(0, num_steps):
        env.make_step()
        reward = env.get_reward()  # Gets global reward
        episode_rewards[-1] += reward
        done = env.is_done()

        if done:
            print("Done Episode " + str(len(episode_rewards)))
            waiting_time_hist.append(env.get_average_waiting_time())
            travel_time_hist.append(env.get_average_travel_time())

            if len(episode_rewards) % save_freq == 0 and t > 0:
                print("Done Episode " + str(len(episode_rewards)))
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("episode reward", episode_rewards[-1])
                logger.dump_tabular()
                plot_rewards()

            reset()
            episode_rewards.append(0.0)
