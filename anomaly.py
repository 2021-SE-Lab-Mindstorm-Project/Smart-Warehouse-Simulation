import csv
import random

import config


def generate_anomaly(start=0, end=config.sim_total_ticks, name=None):
    anomalies = [[], [], []]

    for i in [0, 2]:
        recent_anomaly = BlockAccident(start - config.anomaly_duration - config.anomaly_after, 0)
        for j in range(start, end):
            if recent_anomaly.old(j):
                value = j - recent_anomaly.tick - config.anomaly_duration - config.anomaly_after
                prob = 1 - 2 ** (-value / config.anomaly_mtth)

                if random.random() < prob:
                    recent_anomaly = BlockAccident(j, i)
                    anomalies[i].append(recent_anomaly)

    if name is not None:
        with open('log/anomaly/' + name + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for i in [0, 2]:
                for anomaly in anomalies[i]:
                    csv_writer.writerow([anomaly.tick, anomaly.way])

    return anomalies


def get_anomaly(tick, anomalies):
    current_anomaly = [0, 0, 0]
    if anomalies is not None:
        for i in [0, 2]:
            for single_anomaly in anomalies[i]:
                if single_anomaly.valid(tick):
                    current_anomaly[i] = 1

    return [current_anomaly[0], current_anomaly[2]]


class Anomaly:
    def __init__(self, tick):
        self.tick = tick

    def valid(self, tick):
        if self.tick <= tick < self.tick + config.anomaly_duration:
            return True

        return False

    def old(self, tick):
        if self.tick + config.anomaly_duration + config.anomaly_after <= tick:
            return True

        return False


class BlockAccident(Anomaly):
    def __init__(self, tick, way):
        super().__init__(tick)
        self.tick = tick
        self.way = way
