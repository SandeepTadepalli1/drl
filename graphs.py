import matplotlib.pyplot as plt
import numpy as np


def read(file):
    import csv
    with open(file + '.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')

        n = []
        for row in spamreader:
            i = 0
            while len(n) < 1000:
                if float(row[i].rstrip()) < 1500:
                    n.append(float(row[i].rstrip()))
                i = i+1

        hist = np.asarray(n)
        return hist[0:1000]


def readNegative(file):
    import csv
    with open(file + '.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')

        n = []
        for row in spamreader:
            i = 0
            while len(n) < 350:
                n.append(float(row[i].rstrip()))
                i = i + 1

        hist = np.asarray(n)
        return hist[0:1000]


dqn = read("/Users/jeancarlo/PycharmProjects/thesis/images/files/dqnAWT")
ddqn = read("/Users/jeancarlo/PycharmProjects/thesis/images/files/doubledqnAWT")
ddqnPrior = read("/Users/jeancarlo/PycharmProjects/thesis/images/files/doubledqnPrioriAWT")
plt.plot(np.arange(len(dqn)), dqn, linewidth=.7, label="DQN")
plt.plot(np.arange(len(ddqn)), ddqn, linewidth=.7, label="DDQN", linestyle="dotted")
plt.plot(np.arange(len(ddqnPrior)), ddqnPrior, linewidth=.7, label="Prioritized DDQN", linestyle="dashed")

plt.ylabel('Average Waiting Time')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('average cumulative difference.eps', format='eps', dpi=1000, bbox_inches='tight')

plt.clf()

dqn = read("/Users/jeancarlo/PycharmProjects/thesis/images/files/dqnREW")
ddqn = read("/Users/jeancarlo/PycharmProjects/thesis/images/files/doubledqnREW")
ddqnPrior = read("/Users/jeancarlo/PycharmProjects/thesis/images/files/doubledqnPrioriREW")
plt.plot(np.arange(len(dqn)), dqn, linewidth=.7, label="DQN")
plt.plot(np.arange(len(ddqn)), ddqn, linewidth=.7, label="DDQN", linestyle="dotted")
plt.plot(np.arange(len(ddqnPrior)), ddqnPrior, linewidth=.7, label="Prioritized DDQN", linestyle="dashed")

plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('average cumulative difference rewards.eps', format='eps', dpi=1000, bbox_inches='tight')

plt.clf()

dqn = readNegative("/Users/jeancarlo/PycharmProjects/thesis/images 2/files/dqnAWT")
ddqn = readNegative("/Users/jeancarlo/PycharmProjects/thesis/images 2/files/doubledqnAWT")
ddqnPrior = readNegative("/Users/jeancarlo/PycharmProjects/thesis/images 2/files/doubledqnPrioriAWT")
duelingddqnPrior = readNegative("/Users/jeancarlo/PycharmProjects/thesis/images 2/files/duelingdoubledqnPrioriAWT")
plt.plot(np.arange(len(dqn)), dqn, linewidth=.7, label="DQN")
plt.plot(np.arange(len(ddqn)), ddqn, linewidth=.7, label="DDQN", linestyle="dotted")
plt.plot(np.arange(len(duelingddqnPrior)), ddqnPrior, linewidth=.7, label="Prioritized DDQN", linestyle="dashed")
plt.plot(np.arange(len(duelingddqnPrior)), ddqnPrior, linewidth=.7, label="Dueling DDQN", linestyle="dashdot")

plt.ylabel('Average Waiting Time')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('negative rewards.eps', format='eps', dpi=1000, bbox_inches='tight')
