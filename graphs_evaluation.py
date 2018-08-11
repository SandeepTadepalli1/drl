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
                except IndexError:
                    break

                i = i+1

        hist = np.asarray(n)
        return hist[0:1000]


def read2(file, floor):
    import csv
    with open(file + '.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        prev = 0.0

        n = []
        for row in spamreader:
            i = 0
            while len(n) < 1000:
                try:
                    if float(row[i].rstrip()) < floor:
                        prev = float(row[i].rstrip())
                    n.append(prev)
                except ValueError:
                    sep = '.'
                    e = row[i].rstrip().split(sep)[0]
                    dec = row[i].rstrip().split(sep)[1]
                    val = float(e + "." + dec)

                    if val < floor:
                        prev = val
                    n.append(prev)
                except IndexError:
                    break

                i = i+1

        hist = np.asarray(n)
        return hist[0:1000]


"""
Evaluation - Heterogeneous Low
"""

fp = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FingerprintREW")
ft = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FixedTimeREW")
plt.plot(np.arange(len(fp)), fp, linewidth=.7, label="Fingerprint")
plt.plot(np.arange(len(ft)), ft, linewidth=.7, label="Fixed Time", linestyle="dotted")

plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('Low-Load-REW.png', format='png')

plt.clf()

fp = read2("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FingerprintATT", 120)
ft = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FixedTimeATT")
fig, ax = plt.subplots()
ax.plot(np.arange(len(fp)), fp, linewidth=.7, label="Fingerprint")
ax.plot(np.arange(len(ft)), ft, linewidth=.7, label="Fixed Time", linestyle="dotted")

plt.ylabel('Average Travel Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('Low-Load-ATT.png', format='png')

plt.clf()

fp = read2("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FingerprintAWT", 1250)
ft = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FixedTimeAWT")
fig, ax = plt.subplots()
ax.plot(np.arange(len(fp)), fp, linewidth=.7, label="Fingerprint")
ax.plot(np.arange(len(ft)), ft, linewidth=.7, label="Fixed Time", linestyle="dotted")
#start, end = ax.get_ylim()
#ax.yaxis.set_ticks(np.arange(start, end, 50.0))

plt.ylabel('Average Waiting Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('Low-Load-AWT.png', format='png')

plt.clf()

"""
Evaluation - Heterogeneous High
"""


erm = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/ERMREW")
ft = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/FixedTimeREW")
plt.plot(np.arange(len(erm)), erm, linewidth=.7, label="Fingerprint")
plt.plot(np.arange(len(ft)), ft, linewidth=.7, label="Fixed Time", linestyle="dotted")

plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('High-Load-REW.png', format='png')

plt.clf()

erm = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/ERMATT")
ft = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/FixedTimeATT")
fig, ax = plt.subplots()
ax.plot(np.arange(len(erm)), erm, linewidth=.7, label="Experience Replay", linestyle="dashed")
ax.plot(np.arange(len(ft)), ft, linewidth=.7, label="Fixed Time", linestyle="dotted")

plt.ylabel('Average Travel Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('High-Load-ATT.png', format='png')

plt.clf()

erm = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/ERMAWT")
ft = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/FixedTimeAWT")
fig, ax = plt.subplots()
ax.plot(np.arange(len(erm)), erm, linewidth=.7, label="Experience Replay", linestyle="dashed")
ax.plot(np.arange(len(ft)), ft, linewidth=.7, label="Fixed Time", linestyle="dotted")
#start, end = ax.get_ylim()
#ax.yaxis.set_ticks(np.arange(start, end, 50.0))

plt.ylabel('Average Waiting Time (ms)')
plt.xlabel('Episodes')
plt.legend(loc='upper left')
plt.savefig('High-Load-AWT.png', format='png')

plt.clf()
