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
                try:
                    n.append((800 - 0) / (80000 - 0) * (float(row[i].rstrip()) - 0) + 0)
                except ValueError:
                    sep = '.'
                    e = row[i].rstrip().split(sep)[0]
                    dec = row[i].rstrip().split(sep)[1]
                    val = float(e + "." + dec)
                    n.append((800 - 0) / (80000 - 0) * (val - 0) + 0)

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
                try:
                    n.append(float(row[i].rstrip()) / 10.0 + 90)
                except ValueError:
                    sep = '.'
                    e = row[i].rstrip().split(sep)[0]
                    dec = row[i].rstrip().split(sep)[1]
                    val = float(e + "." + dec)
                    n.append(val / 10.0 + 90)

                i = i+1

        hist = np.asarray(n)
        return hist[0:1000]


def read4(file, floor):
    import csv
    with open(file + '.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')

        n = []
        for row in spamreader:
            i = 0
            while len(n) < 1000:
                try:
                    if float(row[i].rstrip()) < floor:
                        n.append(float(row[i].rstrip()))
                except ValueError:
                    sep = '.'
                    e = row[i].rstrip().split(sep)[0]
                    dec = row[i].rstrip().split(sep)[1]
                    val = float(e + "." + dec)

                    if val < floor:
                        n.append(val)

                i = i+1

        hist = np.asarray(n)
        return hist[0:1000]

"""
Techniques
"""

dqn = read("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/dqnAWT")
plt.plot(np.arange(len(dqn)), dqn, linewidth=.7, label="DQN")
plt.ylabel('Average Waiting Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('DQN Techniques AWT.png', format='png')
plt.clf()

ddqn = read4("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/doubledqnAWT", 500)
plt.plot(np.arange(len(ddqn)), ddqn, linewidth=.7, label="DDQN")
plt.ylabel('Average Waiting Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('DDQN Techniques AWT.png', format='png')
plt.clf()

ddqnPrior = read("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/doubledqnPrioriAWT")
plt.plot(np.arange(len(ddqnPrior)), ddqnPrior, linewidth=.7, label="Prioritized DDQN")
plt.ylabel('Average Waiting Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('Priori DDQN Techniques AWT.png', format='png')
plt.clf()


ddqnPrior = read4("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/duelingdoubledqnPrioriATT_Adam_goodOneAWT", 400)
plt.plot(np.arange(len(ddqnPrior)), ddqnPrior, linewidth=.7, label="Prioritized Dueling DDQN")
plt.ylabel('Average Waiting Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('Dueling Priori DDQN Techniques AWT.png', format='png')
plt.clf()

dqn = read("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/dqnREW")
plt.plot(np.arange(len(dqn)), dqn, linewidth=.7, label="DQN")
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('DQN Techniques Rewards.png', format='png')
plt.clf()

ddqn = read("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/doubledqnREW")
plt.plot(np.arange(len(ddqn)), ddqn, linewidth=.7, label="DDQN")
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('DDQN Techniques Rewards.png', format='png')
plt.clf()

ddqnPrior = read("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/doubledqnPrioriREW")
plt.plot(np.arange(len(ddqnPrior)), ddqnPrior, linewidth=.7, label="Prioritized DDQN")
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('Priori DDQN Techniques Rewards.png', format='png')
plt.clf()

ddqnPrior = read("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/duelingdoubledqnPrioriATT_Adam_goodOneREW")
plt.plot(np.arange(len(ddqnPrior)), ddqnPrior, linewidth=.7, label="Dueling Prioritized DDQN")
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('Dueling Priori DDQN Techniques Rewards.png', format='png')
plt.clf()

dqn = read("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/dqnATT")
plt.plot(np.arange(len(dqn)), dqn, linewidth=.7, label="DQN")
plt.ylabel('Average Travel Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('DQN Techniques ATT.png', format='png')
plt.clf()

ddqn = read("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/doubledqnATT")
plt.plot(np.arange(len(ddqn)), ddqn, linewidth=.7, label="DDQN")
plt.ylabel('Average Travel Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('DDQN Techniques ATT.png', format='png')
plt.clf()

ddqnPrior = read("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/doubledqnPrioriATT")
plt.plot(np.arange(len(ddqnPrior)), ddqnPrior, linewidth=.7, label="Prioritized DDQN")
plt.ylabel('Average Travel Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('Priori DDQN Techniques ATT.png', format='png')
plt.clf()

ddqnPrior = read4("/Users/jeancarlo/PycharmProjects/thesis/images/techniques/duelingdoubledqnPrioriATT_Adam_goodOneATT", 120)
plt.plot(np.arange(len(ddqnPrior)), ddqnPrior, linewidth=.7, label="Dueling Prioritized DDQN")
plt.ylabel('Average Travel Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('Dueling Priori DDQN Techniques ATT.png', format='png')
plt.clf()

"""
RMS vs ADAM
"""

rms = read("/Users/jeancarlo/PycharmProjects/thesis/images/optimizer/doubledqnREW")
adam = read("/Users/jeancarlo/PycharmProjects/thesis/images/optimizer/duelingdoubledqnPrioriATT_Adam_goodOneREW")
plt.plot(np.arange(len(rms)), rms, linewidth=.7, label="RMSProp", linestyle="dotted")
plt.plot(np.arange(len(adam)), adam, linewidth=.7, label="ADAM")

plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('RMS-vs-Adam-REW.png', format='png')

plt.clf()

rms = read3("/Users/jeancarlo/PycharmProjects/thesis/images/optimizer/doubledqnATT")
adam = read("/Users/jeancarlo/PycharmProjects/thesis/images/optimizer/duelingdoubledqnPrioriATT_Adam_goodOneATT")
plt.plot(np.arange(len(rms)), rms, linewidth=.7, label="RMSProp", linestyle="dotted")
plt.plot(np.arange(len(adam)), adam, linewidth=.7, label="ADAM")

plt.ylabel('Average Travel Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('RMS-vs-Adam-ATT.png', format='png')

plt.clf()

rms = read2("/Users/jeancarlo/PycharmProjects/thesis/images/optimizer/doubledqnAWT")
adam = read("/Users/jeancarlo/PycharmProjects/thesis/images/optimizer/duelingdoubledqnPrioriATT_Adam_goodOneAWT")
plt.plot(np.arange(len(rms)), rms, linewidth=.7, label="RMSProp", linestyle="dotted")
plt.plot(np.arange(len(adam)), adam, linewidth=.7, label="ADAM")

plt.ylabel('Average Waiting Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('RMS-vs-Adam-AWT.png', format='png')

plt.clf()

"""
Rewards
"""

neg = read("/Users/jeancarlo/PycharmProjects/thesis/images/rewards/duelingdoubledqnPrioriATT")
good = read("/Users/jeancarlo/PycharmProjects/thesis/images/rewards/duelingdoubledqnPrioriATT_Adam_goodOneATT")
plt.plot(np.arange(len(neg)), neg, linewidth=.7, label="W_t-1 - W_t", linestyle="dotted")
plt.plot(np.arange(len(good)), good, linewidth=.7, label="1 / W_t")

plt.ylabel('Average Travel Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('Rewards-ATT.png', format='png')

plt.clf()

neg = read("/Users/jeancarlo/PycharmProjects/thesis/images/rewards/duelingdoubledqnPrioriAWT")
good = read("/Users/jeancarlo/PycharmProjects/thesis/images/rewards/duelingdoubledqnPrioriATT_Adam_goodOneAWT")
plt.plot(np.arange(len(neg)), neg, linewidth=.7, label="W_t-1 - W_t", linestyle="dotted")
plt.plot(np.arange(len(good)), good, linewidth=.7, label="1 / W_t")

plt.ylabel('Average Waiting Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('Rewards-AWT.png', format='png')