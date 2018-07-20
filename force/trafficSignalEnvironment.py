from __future__ import absolute_import
from __future__ import print_function

import optparse
import os
import random
import sys
import gym

import numpy as np

from tensorforce.environments import Environment

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


class TrafficEnvironment(Environment):
    LANE_LENGHT = 128
    CELL_SIZE = 4

    maxX = 0.0
    minX = 99999999.9
    maxY = 0.0
    minY = 99999999.9

    def __init__(self, time_steps, name):
        self.name = name
        self.time_steps = time_steps

        self.previous_total_waiting_time = 0.0
        self.register_waiting_time = {}
        self.elapsed_steps = 0
        self.cumulative_waiting_time = 0.0

        self.register_loaded_time = {}
        self.register_travel_time = []

        # 2 available actions
        # NUM | Action
        #  1  | Turn Green East<->West
        #  0  | Turn Green North<->South
        self.action_space = [0, 4]

        self.n = int((self.LANE_LENGHT * 2) / self.CELL_SIZE)
        self.observation = np.zeros((2, self.n, self.n))

    def __str__(self):
        return "Traffic Environment"

    @property
    def states(self):
        states = dict(shape=(2, self.n, self.n), type='float')
        return states

    @property
    def actions(self):
        actions = dict(type='int', num_actions=2)
        return actions

    def close(self):
        traci.close(False)
        sys.stdout.flush()

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

        with open("/Users/jeancarlo/PycharmProjects/thesis/traci_tls/data/cross" + self.name + ".rou.xml", "w") as routes:
            print("""<routes>
            <vType id="car" accel="1.0" decel="4.5" sigma="0.7" length="5" minGap="3" maxSpeed="15" guiShape="passenger"/>

            <route id="right" edges="1i 2o" />
            <route id="left" edges="2i 1o" />
            <route id="down" edges="4i 3o" />
            <route id="up" edges="3i 4o" />""", file=routes)

            vehNr = 0

            for i in range(self.time_steps):
                if random.uniform(0, 1) < pWE:
                    print('    <vehicle id="right_%i" type="car" route="right" depart="%i" departSpeed="5"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pEW:
                    print('    <vehicle id="left_%i" type="car" route="left" depart="%i" departSpeed="5"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pNS:
                    print('    <vehicle id="down_%i" type="car" route="down" depart="%i" departSpeed="5"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pSN:
                    print('    <vehicle id="up_%i" type="car" route="up" depart="%i" departSpeed="5"/>' % (
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

    def _choose_next_observation(self):
        self.make_step()
        self.observation.fill(0)

        for veh in traci.vehicle.getIDList():
            position = traci.vehicle.getPosition(veh)
            normalized_speed = traci.vehicle.getSpeed(veh) / traci.vehicle.getAllowedSpeed(veh)

            self.observation[
                0, abs(int((position[1]) / self.CELL_SIZE) - self.n) - 1, int((position[0]) / self.CELL_SIZE)] += 1.0
            self.observation[1, abs(int((position[1]) / self.CELL_SIZE) - self.n) - 1, int(
                (position[0]) / self.CELL_SIZE)] += normalized_speed

        # self._set_traffic_light_state()

    def _set_traffic_light_state(self):
        GREEN = 1.0
        RED = 2.0

        # N<->S GREEN
        # W<->E RED
        if traci.trafficlight.getPhase("0") == self.action_space[0]:
            self.observation[2, int(self.n / 2), int(self.n / 2)] = GREEN
            self.observation[2, int(self.n / 2) - 1, int(self.n / 2) - 1] = GREEN
            self.observation[2, int(self.n / 2), int(self.n / 2) - 1] = RED
            self.observation[2, int(self.n / 2) - 1, int(self.n / 2)] = RED

        # W<->E GREEN
        # N<->S RED
        if traci.trafficlight.getPhase("0") == self.action_space[1]:
            self.observation[2, int(self.n / 2), int(self.n / 2)] = RED
            self.observation[2, int(self.n / 2) - 1, int(self.n / 2) - 1] = RED
            self.observation[2, int(self.n / 2), int(self.n / 2) - 1] = GREEN
            self.observation[2, int(self.n / 2) - 1, int(self.n / 2)] = GREEN

    def _get_reward(self):
        """
        total_waiting_time = 0.0
        for veh in traci.vehicle.getIDList():
            total_waiting_time += traci.vehicle.getWaitingTime(veh)

        reward = 0.0 if len(traci.vehicle.getIDList()) == 0 else -(
                    (total_waiting_time ** 2) / len(traci.vehicle.getIDList()))
        self.cumulative_waiting_time += 0.0 if len(traci.vehicle.getIDList()) == 0 else total_waiting_time / len(
            traci.vehicle.getIDList())

        reward = 0.0
        total_waiting_time = 0.0
        vehicles = traci.vehicle
        for veh in vehicles.getIDList():
            if vehicles.isStopped(veh) or vehicles.getSpeed(veh) <= (vehicles.getAllowedSpeed(veh) * 0.25):  # less than 25% of the allowed speed
                reward -= ((vehicles.getAllowedSpeed(veh) * 0.25) - vehicles.getSpeed(veh)) * (1 + vehicles.getWaitingTime(veh))
                total_waiting_time += vehicles.getWaitingTime(veh)

        self.cumulative_waiting_time += 0.0 if len(vehicles.getIDList()) == 0 else total_waiting_time / len(vehicles.getIDList())
        """

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
        # self.previous_total_waiting_time = current_total_waiting_time

        return reward

    def reset(self):
        options = self._get_options()

        # this script has been called from the command line. It will start sumo as a
        # server, then connect and run
        if options.nogui:
            sumo_binary = checkBinary('sumo')
        else:
            sumo_binary = checkBinary('sumo-gui')

        # first, generate the route file for this simulation
        self.generate_route_file()

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([sumo_binary, "-c", "/Users/jeancarlo/PycharmProjects/thesis/traci_tls/data/cross" + self.name + ".sumocfg",
                     "--start", "--quit-on-end", "--waiting-time-memory", "10000", "--time-to-teleport", "-1"])

        self.previous_total_waiting_time = 0.0
        self.cumulative_waiting_time = 0.0
        self.register_waiting_time.clear()
        self.register_travel_time.clear()
        self.register_loaded_time.clear()

        self._choose_next_observation()
        return self.observation

    def execute(self, actions):
        done = False

        if traci.trafficlight.getPhase("0") != self.action_space[actions]:
            traci.trafficlight.setPhase("0", traci.trafficlight.getPhase("0") + 1)  # Turn yellow
            for s in range(0, 4):
                self.make_step()

        traci.trafficlight.setPhase("0", self.action_space[actions])
        self._choose_next_observation()

        reward = self._get_reward()

        if traci.simulation.getMinExpectedNumber() <= 0 or (traci.simulation.getCurrentTime() / 1000) > self.time_steps:
            self.elapsed_steps = (traci.simulation.getCurrentTime() / 1000)
            done = True
            self.close()

        return self.observation, done, reward

    def calculate_metrics(self):
        for veh in traci.simulation.getLoadedIDList():
            self.register_loaded_time[veh] = traci.simulation.getCurrentTime()

        for veh in traci.simulation.getDepartedIDList():
            if veh in self.register_loaded_time:
                self.register_travel_time.append((traci.simulation.getCurrentTime() - self.register_loaded_time.get(veh)) / 1000.0)

    def make_step(self):
        traci.simulationStep()
        self.calculate_metrics()
