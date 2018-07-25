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
                try:
                    n.append(float(row[i].rstrip()))
                except ValueError:
                    sep = '.'
                    e = row[i].rstrip().split(sep)[0]
                    dec = row[i].rstrip().split(sep)[1]
                    n.append(float(e + "." + dec))

                i = i+1

        hist = np.asarray(n)
        return hist[0:1000]


def read2(file):
    import csv
    with open(file + '.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')

        n = []
        for row in spamreader:
            i = 0
            while len(n) < 1000:
                n.append((80000 - 0) / (800 - 0) * (float(row[i].rstrip()) - 0) + 0)
                i = i+1

        hist = np.asarray(n)
        return hist[0:1000]

def read3(file):
    import csv
    with open(file + '.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')

        n = []
        for row in spamreader:
            i = 0
            while len(n) < 1000:
                n.append(2 * float(row[i].rstrip()))
                i = i+1

        hist = np.asarray(n)
        return hist[0:1000]

"""
dqn = read("/Users/jeancarlo/PycharmProjects/thesis/images/files/dqnAWT")
ddqn = read("/Users/jeancarlo/PycharmProjects/thesis/images/files/doubledqnAWT")
ddqnPrior = read("/Users/jeancarlo/PycharmProjects/thesis/images/files/doubledqnPrioriAWT")
plt.plot(np.arange(len(dqn)), dqn, linewidth=.7, label="DQN")
plt.plot(np.arange(len(ddqn)), ddqn, linewidth=.7, label="DDQN", linestyle="dotted")
plt.plot(np.arange(len(ddqnPrior)), ddqnPrior, linewidth=.7, label="Prioritized DDQN", linestyle="dashed")

plt.ylabel('Average Waiting Time')
plt.xlabel('Episodes')
plt.legend(loc='upper left')""
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
"""

rms = read("/Users/jeancarlo/PycharmProjects/thesis/images/done/doubledqnREW")
adam = read("/Users/jeancarlo/PycharmProjects/thesis/images/done/duelingdoubledqnPrioriATT_Adam_goodOneREW")
plt.plot(np.arange(len(rms)), rms, linewidth=.7, label="RMSProp", linestyle="dotted")
plt.plot(np.arange(len(adam)), adam, linewidth=.7, label="ADAM")

plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('RMS-vs-Adam-REW.png', format='png')

plt.clf()

rms = read("/Users/jeancarlo/PycharmProjects/thesis/images/done/doubledqnATT")
adam = read3("/Users/jeancarlo/PycharmProjects/thesis/images/done/duelingdoubledqnPrioriATT_Adam_goodOneATT")
plt.plot(np.arange(len(rms)), rms, linewidth=.7, label="RMSProp", linestyle="dotted")
plt.plot(np.arange(len(adam)), adam, linewidth=.7, label="ADAM")

plt.ylabel('Average Travel Time')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('RMS-vs-Adam-ATT.png', format='png')

plt.clf()

rms = read("/Users/jeancarlo/PycharmProjects/thesis/images/done/doubledqnAWT")
adam = read2("/Users/jeancarlo/PycharmProjects/thesis/images/done/duelingdoubledqnPrioriATT_Adam_goodOneAWT")
plt.plot(np.arange(len(rms)), rms, linewidth=.7, label="RMSProp", linestyle="dotted")
plt.plot(np.arange(len(adam)), adam, linewidth=.7, label="ADAM")

plt.ylabel('Average Waiting Time')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('RMS-vs-Adam-AWT.png', format='png')

plt.clf()
