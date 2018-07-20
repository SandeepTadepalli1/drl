from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from force.trafficSignalEnvironment import TrafficEnvironment
import numpy as np


def plot_rewards(episode_rewards):
    import matplotlib.pyplot as plt
    import datetime

    plt.clf()
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.ylabel('Reward')
    plt.xlabel('Training Epochs')
    plt.savefig(
        '/Users/jeancarlo/PycharmProjects/thesis/images/rew/' + nameProc + str(datetime.datetime.now()).split('.')[0]
        + '.eps', format='eps', dpi=1000)

    with open('/Users/jeancarlo/PycharmProjects/thesis/images/files/' + nameProc + 'REW.csv', 'a') as file:
        file.write(str(episode_rewards[-1]) + ";")

    waiting_time_hist.append(env.get_average_waiting_time())
    plt.clf()
    plt.plot(np.arange(len(waiting_time_hist)), waiting_time_hist)
    plt.ylabel('Average Waiting Time')
    plt.xlabel('Training Epochs')
    plt.savefig(
        '/Users/jeancarlo/PycharmProjects/thesis/images/awt/' + nameProc + str(datetime.datetime.now()).split('.')[
            0] + '.eps', format='eps', dpi=1000)

    with open('/Users/jeancarlo/PycharmProjects/thesis/images/files/' + nameProc + 'AWT.csv', 'a') as file:
        file.write(str(waiting_time_hist[-1]) + ";")

    with open('/Users/jeancarlo/PycharmProjects/thesis/images/files/' + nameProc + 'REW.csv', 'a') as file:
        file.write(str(episode_rewards[-1]) + ";")

    travel_time_hist.append(env.get_average_travel_time())
    plt.clf()
    plt.plot(np.arange(len(travel_time_hist)), travel_time_hist)
    plt.ylabel('Average Travel Time')
    plt.xlabel('Training Epochs')
    plt.savefig(
        '/Users/jeancarlo/PycharmProjects/thesis/images/att/' + nameProc + str(datetime.datetime.now()).split('.')[
            0] + '.eps', format='eps', dpi=1000)

    with open('/Users/jeancarlo/PycharmProjects/thesis/images/files/' + nameProc + 'ATT.csv', 'a') as file:
        file.write(str(travel_time_hist[-1]) + ";")


if __name__ == '__main__':
    waiting_time_hist = []
    travel_time_hist = []
    nameProc = "dqn"
    env = TrafficEnvironment(3600, nameProc)

    network_spec = [
        dict(type="conv2d", size=32, window=8, stride=4),
        dict(type="conv2d", size=64, window=4, stride=2),
        dict(type="conv2d", size=64, window=3, stride=1),
        dict(type="flatten"),
        dict(type='dueling', size=512),
        dict(type='dense', size=len(env.action_space))
    ]

    agent = DQNAgent(
        states=env.states,
        actions=env.actions,
        network=network_spec,
        update_mode=dict(
            unit="timesteps",
            batch_size=32,
            frequency=4
        ),
        memory=dict(
            type="prioritized_replay",
            capacity=50000,
            include_next_states=True,
            buffer_size=5000
        ),
        batching_capacity=50000,
        target_sync_frequency=5000,
        actions_exploration=dict(
            type='epsilon_decay',
            final_epsilon=0.01,
            timesteps=360000
        ),
        optimizer=dict(
            type='adam',
            learning_rate=1e-4
        ),
        double_q_model=False,
    )

    # Create the runner
    runner = Runner(agent=agent, environment=env)


    # Callback function printing episode statistics
    def episode_finished(r):
        print("Finished episode {ep} after {ts} timesteps (reward: {reward}) (Total timesteps {tts})"
              .format(ep=r.global_episode, ts=r.current_timestep, reward=r.episode_rewards[-1], tts=r.global_timestep))
        plot_rewards(r.episode_rewards)
        return True

    # Start learning
    runner.run(num_episodes=1000, episode_finished=episode_finished)
    runner.close()

    # Print statistics
    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        ep=runner.episode,
        ar=np.mean(runner.episode_rewards[-100:]))
    )
