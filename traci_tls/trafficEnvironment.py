from __future__ import absolute_import
from __future__ import print_function

import optparse
import os
import random
import sys

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


class TrafficEnv:
    LANE_LENGHT = 128
    CELL_SIZE = 4

    path = "/Users/jeancarlo/PycharmProjects/thesis/"

    def __init__(self, time_steps, name):
        self.name = name
        self.time_steps = time_steps

        self.register_waiting_time = {}
        self.elapsed_steps = 0
        self.cumulative_waiting_time = 0.0

        self.register_loaded_time = {}
        self.register_travel_time = []

        self.n = int((self.LANE_LENGHT * 2) / self.CELL_SIZE)
        self.observationDim = np.zeros((2, self.n, self.n))

    def get_average_waiting_time(self):
        return self.cumulative_waiting_time / self.elapsed_steps

    def get_average_travel_time(self):
        total = 0.0
        for veh in self.register_travel_time:
            total += veh

        return total / len(self.register_travel_time)

    def generate_route_file(self):
        # demand per second from different directions
        pWE = 1. / 10
        pEW = 1. / 10
        pNS = 1. / 10
        pSN = 1. / 10

        pt2 = 1. / 10
        pt3 = 1. / 10
        pt4 = 1. / 10
        pt5 = 1. / 10
        pt6 = 1. / 10

        with open(self.path + "traci_tls/data/cross" + self.name + ".rou.xml", "w") as routes:
            print("""<routes>
                               <vType id="car" accel="1.0" decel="4.5" sigma="0.7" length="5" minGap="3" maxSpeed="15" guiShape="passenger"/>

                               <route id="WETop" edges="5i 1i 2o" />
                               <route id="EWTop" edges="2i 1o 5o" />
                               <route id="NS" edges="4i 3o 8i" />
                               <route id="SN" edges="8o 3i 4o" />
                               <route id="WEBottom" edges="9o 7o 11o" />
                               <route id="EWBottom" edges="11i 7i 9i" />

                               <route id="t2" edges="8o 7i 10i" />
                               <route id="t3" edges="4i 1o 6i" />
                               <route id="t4" edges="10o 7o 11o" />
                               <route id="t5" edges="9o 7o 8i" />
                               <route id="t6" edges="4i 3o 11o" />
                               """, file=routes)

            vehNr = 0
            for i in range(self.time_steps):
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
                if random.uniform(0, 1) < pt5:
                    print(
                        '    <vehicle id="t5_%i" type="car" route="t5" depart="%i" departSpeed="5"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pt6:
                    print(
                        '    <vehicle id="t6_%i" type="car" route="t6" depart="%i" departSpeed="5"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1

            print("</routes>", file=routes)

    @staticmethod
    def _get_options():
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=True, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options

    def choose_next_observation(self, x, y):
        observation = np.zeros((2, self.n, self.n))

        for veh in traci.vehicle.getIDList():
            position = traci.vehicle.getPosition(veh)
            position_zero = position[0] - x
            position_one = position[1] - y
            normalized_speed = traci.vehicle.getSpeed(veh) / traci.vehicle.getAllowedSpeed(veh)

            try:
                observation[0, abs(int(position_one / self.CELL_SIZE) - self.n) - 1, int(position_zero / self.CELL_SIZE)] += 1.0  # Position Matrix
                observation[1, abs(int(position_one / self.CELL_SIZE) - self.n) - 1, int(position_zero / self.CELL_SIZE)] += normalized_speed  # Speed Matrix
            except IndexError:
                # vehicle is not in the agent's view
                continue

        return observation

    def get_reward(self):
        vehicles = traci.vehicle
        current_total_waiting_time = 0.0

        for veh in vehicles.getIDList():
            if not (veh in self.register_waiting_time):
                self.register_waiting_time[veh] = vehicles.getAccumulatedWaitingTime(veh)
                current_total_waiting_time += vehicles.getAccumulatedWaitingTime(veh)

            if self.register_waiting_time.get(veh) != vehicles.getAccumulatedWaitingTime(veh):
                self.register_waiting_time[veh] = vehicles.getAccumulatedWaitingTime(veh)
                current_total_waiting_time += vehicles.getAccumulatedWaitingTime(veh)

        reward = 1.0 if current_total_waiting_time == 0.0 else 1.0 / current_total_waiting_time
        self.cumulative_waiting_time += current_total_waiting_time

        return reward

    def reset(self):
        options = self._get_options()

        # this script has been called from the command line. It will start sumo as a server, then connect and run
        if options.nogui:
            sumo_binary = checkBinary('sumo')
        else:
            sumo_binary = checkBinary('sumo-gui')

        # first, generate the route file for this simulation
        self.generate_route_file()

        # SUMO is started as a subprocess and then the python script connects and runs
        traci.start([sumo_binary, "-c", self.path + "traci_tls/data/cross" + self.name + ".sumocfg",
                     "--start", "--quit-on-end", "--waiting-time-memory", "10000", "--time-to-teleport", "-1"])

        self.cumulative_waiting_time = 0.0
        self.register_waiting_time.clear()
        self.register_travel_time.clear()
        self.register_loaded_time.clear()

        self.make_step()

    def set_phase(self, action, traffic_light_id, actions):
        if traci.trafficlight.getPhase(traffic_light_id) != actions[action]:
            traci.trafficlight.setPhase(traffic_light_id, traci.trafficlight.getPhase(traffic_light_id) + 1)  # Turn yellow
            return True

        traci.trafficlight.setPhase(traffic_light_id, actions[action])
        return False

    def is_done(self):
        if traci.simulation.getMinExpectedNumber() <= 0 or (traci.simulation.getCurrentTime() / 1000) > self.time_steps:
            self.elapsed_steps = (traci.simulation.getCurrentTime() / 1000)
            return True

        return False

    def close(self):
        traci.close(False)
        sys.stdout.flush()

    def calculate_metrics(self):
        for veh in traci.simulation.getLoadedIDList():
            self.register_loaded_time[veh] = traci.simulation.getCurrentTime()

        for veh in traci.simulation.getDepartedIDList():
            if veh in self.register_loaded_time:
                self.register_travel_time.append((traci.simulation.getCurrentTime() - self.register_loaded_time.get(veh)) / 1000.0)

    def make_step(self):
        traci.simulationStep()
        self.calculate_metrics()
