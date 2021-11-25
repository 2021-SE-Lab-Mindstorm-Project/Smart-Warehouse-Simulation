import datetime

import numpy as np
from matplotlib import pyplot as plt

import config
import smart_warehouse


def run_simulation(name, order_list, anomalies):
    rewards = []
    ticks = []

    for target in config.sim_targets:
        reward, tick = smart_warehouse.run(name, target, order_list, anomalies)
        rewards.append(reward)
        ticks.append(tick)

    return np.array(rewards), np.array(ticks)


if __name__ == "__main__":
    rewards = [[] for _ in range(len(config.sim_targets))]
    ticks = [[] for _ in range(len(config.sim_targets))]

    for i in range(config.sim_count):
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        order_list = smart_warehouse.generate_order_list(experiment_name)
        anomalies = smart_warehouse.generate_anomaly(is_real=True, name=experiment_name)
        # anomalies = [[], [], []]

        single_reward, single_tick = run_simulation(experiment_name, order_list, anomalies)

        for j in range(len(config.sim_targets)):
            rewards[j].append(single_reward[j])
            ticks[j].append(single_tick[j])

    plt.title('Rewards')
    plt.boxplot(rewards)
    plt.xticks(list(range(1, len(config.sim_targets) + 1)), config.sim_targets)
    plt.show()
    plt.close()

    plt.title('Processed Time')
    plt.boxplot(ticks)
    plt.xticks(list(range(1, len(config.sim_targets) + 1)), config.sim_targets)
    plt.show()
    plt.close()

    for i, target in enumerate(config.sim_targets):
        print()
        print(target)
        print('average reward: ', rewards[i])
        print('average time: ', ticks[i])
