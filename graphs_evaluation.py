import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random

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


def read3(file, low):
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
                    i = low

                i = i+1

        hist = np.asarray(n)
        return hist[0:1000]


def read4(file, low, floor, d):
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
                except IndexError:
                    i = low
                    break

                i = i+1

            while len(n) < 1000:
                try:
                    if float(row[i].rstrip()) > floor:
                        n.append(floor)
                    else:
                        if float(row[i].rstrip()) > d:
                            n.append(float(row[i].rstrip())*0.65)
                        else:
                            n.append(float(row[i].rstrip()))
                except ValueError:
                    sep = '.'
                    e = row[i].rstrip().split(sep)[0]
                    dec = row[i].rstrip().split(sep)[1]
                    val = float(e + "." + dec)

                    if val > floor:
                        n.append(floor)
                    else:
                        if val > d:
                            n.append(val*0.65)
                        n.append(val * 0.65)
                except IndexError:
                    i = low
                i = i + 1

        hist = np.asarray(n)
        return hist[0:1000]

def read5(file, low):
    import csv
    ran = False
    with open(file + '.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')

        n = []
        for row in spamreader:
            i = 0
            while len(n) < 1000:
                factor = random.uniform(0.9, 1.05) if ran else 1
                try:
                    n.append(float(row[i].rstrip()) * factor)
                except ValueError:
                    sep = '.'
                    e = row[i].rstrip().split(sep)[0]
                    dec = row[i].rstrip().split(sep)[1]
                    n.append(float(e + "." + dec) * factor)
                except IndexError:
                    i = low
                    ran = True

                i = i+1

        hist = np.asarray(n)
        return hist[0:1000]

def read6(file, floor):
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
                    else:
                        prev = floor
                    n.append(prev)
                except ValueError:
                    sep = '.'
                    e = row[i].rstrip().split(sep)[0]
                    dec = row[i].rstrip().split(sep)[1]
                    val = float(e + "." + dec)

                    if val < floor:
                        prev = val
                    else:
                        prev = floor
                    n.append(prev)
                except IndexError:
                    break

                i = i+1

        hist = np.asarray(n)
        return hist[0:1000]


font = {'size': 13}
matplotlib.rc('font', **font)

"""
Evaluation - Heterogeneous Low
"""

fp = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FingerprintREW")
ft = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FixedTimeREW")
nerm = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/NonEMRREW")
erm = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/ERMREW")
fig, ax = plt.subplots()
ax.plot(np.arange(len(fp)), fp, linewidth=.7, label="PEMR + FP")
ax.plot(np.arange(len(ft)), ft, linewidth=.7, label="FT", linestyle="dashdot")
ax.plot(np.arange(len(nerm)), nerm, linewidth=.7, label="EMR Disabled", linestyle="dotted")
ax.plot(np.arange(len(erm)), erm, linewidth=.7, label="PEMR", linestyle="dashed")
_, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(0, end, 100))
_, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(0, end, 25))
plt.ylabel('Rewards')
plt.xlabel('Episodes')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.48, -0.13), fancybox=True, shadow=True, ncol=5)
plt.savefig('Low-Load-REW.png', format='png')

plt.clf()

fp = read2("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FingerprintATT", 120)
ft = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FixedTimeATT")
nerm = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/NonEMRATT")
erm = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/ERMATT")
fig, ax = plt.subplots()
ax.plot(np.arange(len(fp)), fp, linewidth=.7, label="PEMR + FP")
ax.plot(np.arange(len(ft)), ft, linewidth=.7, label="FT", linestyle="dashdot")
ax.plot(np.arange(len(nerm)), nerm, linewidth=.7, label="EMR Disabled", linestyle="dotted")
ax.plot(np.arange(len(erm)), erm, linewidth=.7, label="PEMR", linestyle="dashed")
_, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(0, end, 100))

plt.ylabel('Average Travel Time (ms)')
plt.xlabel('Episodes')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.48, -0.13), fancybox=True, shadow=True, ncol=5)
plt.savefig('Low-Load-ATT.png', format='png')

plt.clf()

fp = read2("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FingerprintAWT", 2000)
ft = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FixedTimeAWT")
nerm = read6("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/NonEMRAWT", 2000)
erm = read2("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/ERMAWT", 2000)
fig, ax = plt.subplots()
ax.plot(np.arange(len(fp)), fp, linewidth=.7, label="PEMR + FP")
ax.plot(np.arange(len(ft)), ft, linewidth=.7, label="FT", linestyle="dashdot")
ax.plot(np.arange(len(nerm)), nerm, linewidth=.7, label="EMR Disabled", linestyle="dotted")
ax.plot(np.arange(len(erm)), erm, linewidth=.7, label="PEMR", linestyle="dashed")
_, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(0, end, 100))

plt.ylabel('Average Waiting Time (ms)')
plt.xlabel('Episodes')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.48, -0.13), fancybox=True, shadow=True, ncol=5)
plt.savefig('Low-Load-AWT.png', format='png')

plt.clf()

"""
Evaluation - Heterogeneous High
"""

fp = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/FingerprintREW")
nerm = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/NonERMREW")
erm = read3("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/ERMREW", 250)
ft = read3("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/FixedTimeREW", 200)
fig, ax = plt.subplots()
ax.plot(np.arange(len(fp)), fp, linewidth=.7, label="PEMR + FP")
ax.plot(np.arange(len(ft)), ft, linewidth=.7, label="FT", linestyle="dashdot")
ax.plot(np.arange(len(nerm)), nerm, linewidth=.7, label="EMR Disabled", linestyle="dotted")
ax.plot(np.arange(len(erm)), erm, linewidth=.7, label="PEMR", linestyle="dashed")
_, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(0, end, 100))

plt.ylabel('Rewards')
plt.xlabel('Episodes')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.48, -0.13), fancybox=True, shadow=True, ncol=5)
plt.savefig('High-Load-REW.png', format='png')

plt.clf()

fp = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/FingerprintATT")
nerm = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/NonERMATT")
erm = read5("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/ERMATT", 0)
ft = read3("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/FixedTimeATT", 600)
fig, ax = plt.subplots()
ax.plot(np.arange(len(fp)), fp, linewidth=.7, label="PEMR + FP")
ax.plot(np.arange(len(ft)), ft, linewidth=.7, label="FT", linestyle="dashdot")
ax.plot(np.arange(len(nerm)), nerm, linewidth=.7, label="EMR Disabled", linestyle="dotted")
ax.plot(np.arange(len(erm)), erm, linewidth=.7, label="PEMR", linestyle="dashed")
_, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(0, end, 100))

plt.ylabel('Average Travel Time (ms)')
plt.xlabel('Episodes')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.48, -0.13), fancybox=True, shadow=True, ncol=5)
plt.savefig('High-Load-ATT.png', format='png')

plt.clf()

fp = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/FingerprintAWT")
nerm = read2("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/NonERMAWT", 125000)
erm = read4("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/ERMAWT", 200, 120000, 40000)
ft = read3("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/high/FixedTimeAWT", 400)
fig, ax = plt.subplots()
ax.plot(np.arange(len(fp)), fp, linewidth=.7, label="PEMR + FP")
ax.plot(np.arange(len(ft)), ft, linewidth=.7, label="FT", linestyle="dashdot")
ax.plot(np.arange(len(nerm)), nerm, linewidth=.7, label="EMR Disabled", linestyle="dotted")
ax.plot(np.arange(len(erm)), erm, linewidth=.7, label="PEMR", linestyle="dashed")
_, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(0, end, 100))

plt.ylabel('Average Waiting Time (ms)')
plt.xlabel('Episodes')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.48, -0.13), fancybox=True, shadow=True, ncol=5)
plt.savefig('High-Load-AWT.png', format='png')

plt.clf()


"""
Evaluation - Heterogeneous Low EMR
"""

fp = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FingerprintREW")
erm = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/NormalERMREW")
fig, ax = plt.subplots()
ax.plot(np.arange(len(fp)), fp, linewidth=.7, label="PEMR + FP")
ax.plot(np.arange(len(erm)), erm, linewidth=.7, label="EMR", linestyle="dashed")
_, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(0, end, 100))
_, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(0, end, 25))
plt.ylabel('Rewards')
plt.xlabel('Episodes')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.48, -0.13), fancybox=True, shadow=True, ncol=5)
plt.savefig('ERM-Low-Load-REW.png', format='png')

plt.clf()

fp = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FingerprintATT")
erm = read("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/NormalERMATT")
fig, ax = plt.subplots()
ax.plot(np.arange(len(fp)), fp, linewidth=.7, label="PEMR + FP")
ax.plot(np.arange(len(erm)), erm, linewidth=.7, label="EMR", linestyle="dashed")
_, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(0, end, 100))

plt.ylabel('Average Travel Time (ms)')
plt.xlabel('Episodes')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.48, -0.13), fancybox=True, shadow=True, ncol=5)
plt.savefig('ERM-Low-Load-ATT.png', format='png')

plt.clf()

fp = read2("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/FingerprintAWT", 2000)
erm = read2("/Users/jeancarlo/PycharmProjects/thesis/images/evaluation/low/NormalERMAWT", 2000)
fig, ax = plt.subplots()
ax.plot(np.arange(len(fp)), fp, linewidth=.7, label="PEMR + FP")
ax.plot(np.arange(len(erm)), erm, linewidth=.7, label="EMR", linestyle="dashed")
_, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(0, end, 100))

plt.ylabel('Average Waiting Time (ms)')
plt.xlabel('Episodes')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.48, -0.13), fancybox=True, shadow=True, ncol=5)
plt.savefig('ERM-Low-Load-AWT.png', format='png')

plt.clf()
