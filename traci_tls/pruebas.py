import numpy as np

episode_rewards = []

for i in range(0, 100):
    episode_rewards.append(i)

print(";".join(map(str, episode_rewards[-25:])))
