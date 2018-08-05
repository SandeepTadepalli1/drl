#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2017 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26
# @version $Id$

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import numpy as np

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci


def generate_routefile():
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 10
    pNS = 1. / 10
    pSN = 1. / 10

    pt1 = 1. / 10
    pt2 = 1. / 10
    pt3 = 1. / 10
    pt4 = 1. / 10

    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
                                      <vType id="car" accel="1.0" decel="4.5" sigma="0.7" length="5" minGap="3" maxSpeed="15" guiShape="passenger"/>

                                      <route id="WETop" edges="5i 1i 2o" />
                                      <route id="EWTop" edges="2i 1o 5o" />
                                      <route id="NS" edges="4i 3o 8i" />
                                      <route id="SN" edges="8o 3i 4o" />
                                      <route id="WEBottom" edges="9o 7o 11o" />
                                      <route id="EWBottom" edges="11i 7i 9i" />

                                      <route id="t1" edges="6o 1i 3o 11o" />
                                      <route id="t2" edges="8o 7i 10i" />
                                      <route id="t3" edges="4i 1o 6i" />
                                      <route id="t4" edges="10o 7o 11o" />

                                      <route id="t5" edges="9o 7o 8i" />
                                      <route id="t6" edges="9o 10i" />
                                      <route id="t7" edges="6o 5o" />
                                      <route id="t8" edges="4i 3o 11o" />
                                      <route id="t9" edges="9o 7o 3i 1o 6i" />
                                      """, file=routes)

        vehNr = 0
        for i in range(3600):
            if random.uniform(0, 1) < pWE:
                print(
                    '    <vehicle id="WETop_%i" type="car" route="WETop" depart="%i" departSpeed="5"/>' % (
                        vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print(
                    '    <vehicle id="EWTop_%i" type="car" route="EWTop" depart="%i" departSpeed="5"/>' % (
                        vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWE:
                print(
                    '    <vehicle id="WEBottom_%i" type="car" route="WEBottom" depart="%i" departSpeed="5"/>' % (
                        vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print(
                    '    <vehicle id="EWBottom_%i" type="car" route="EWBottom" depart="%i" departSpeed="5"/>' % (
                        vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="NS_%i" type="car" route="NS" depart="%i" departSpeed="5"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSN:
                print('    <vehicle id="SN_%i" type="car" route="SN" depart="%i" departSpeed="5"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1

            if random.uniform(0, 1) < pt1:
                print(
                    '    <vehicle id="t1_%i" type="car" route="t1" depart="%i" departSpeed="5"/>' % (
                        vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pt2:
                print(
                    '    <vehicle id="t2_%i" type="car" route="t2" depart="%i" departSpeed="5"/>' % (
                        vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pt3:
                print('    <vehicle id="t3_%i" type="car" route="t3" depart="%i" departSpeed="5"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pt4:
                print('    <vehicle id="t4_%i" type="car" route="t4" depart="%i" departSpeed="5"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1

                print(
                    '    <vehicle id="t5_%i" type="car" route="t5" depart="%i" departSpeed="5"/>' % (
                        vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pt2:
                print(
                    '    <vehicle id="t6_%i" type="car" route="t6" depart="%i" departSpeed="5"/>' % (
                        vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pt3:
                print('    <vehicle id="t6_%i" type="car" route="t6" depart="%i" departSpeed="5"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pt4:
                print('    <vehicle id="t7_%i" type="car" route="t7" depart="%i" departSpeed="5"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pt3:
                print('    <vehicle id="t8_%i" type="car" route="t8" depart="%i" departSpeed="5"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pt4:
                print('    <vehicle id="t9_%i" type="car" route="t9" depart="%i" departSpeed="5"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1

        print("</routes>", file=routes)


def run():
    """execute the TraCI control loop"""
    step = 0

    LANE_LENGHT = 128
    CELL_SIZE = 4
    n = int((LANE_LENGHT * 2) / CELL_SIZE)
    x = 0.0
    y = 0.0

    state_matrix = np.zeros((2, n, n))

    minX = 99999999.9
    minY = 99999999.9
    maxX = -99999999.9
    maxY = -99999999.9

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        state_matrix.fill(0)

        # for veh in traci.vehicle.getIDList():
        #     position = traci.vehicle.getPosition(veh)
        #     position_zero = position[0] - x
        #     position_one = position[1] - y
        #     normalized_speed = traci.vehicle.getSpeed(veh) / traci.vehicle.getAllowedSpeed(veh)
        #
        #     minY = min(minY, position_zero)
        #     maxY = max(maxY, position_zero)
        #     minX = min(minX, position_one)
        #     maxX = max(maxX, position_one)
        #
        #     try:
        #         state_matrix[0, abs(int(position_one / CELL_SIZE) - n) - 1, int(position_zero / CELL_SIZE)] += 1.0
        #         state_matrix[1, abs(int(position_one / CELL_SIZE) - n) - 1, int(position_zero / CELL_SIZE)] += normalized_speed
        #     except IndexError:
        #         # vehicle is not in the agent's view
        #         continue

        step += 1
        traci.trafficlight.setPhase("12", 6)

    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--waiting-time-memory", "10000", "--time-to-teleport", "-1"])
    run()
