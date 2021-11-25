import datetime

from matplotlib import pyplot as plt

import config
import smart_warehouse
from main import run_simulation

if __name__ == "__main__":
    anomaly_mtbf_list = [1, 4, 25, 100]

    rewards = [[] for _ in range(len(config.sim_targets))]
    ticks = [[] for _ in range(len(config.sim_targets))]

    for i in anomaly_mtbf_list:
        config.anomaly_mtbf = i

        reward = [[] for _ in range(len(config.sim_targets))]
        tick = [[] for _ in range(len(config.sim_targets))]
        for j in range(config.sim_count):
            experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            order_list = smart_warehouse.generate_order_list(experiment_name)
            anomalies = smart_warehouse.generate_anomaly(is_real=True, name=experiment_name)
            # anomalies = [[], [], []]

            single_reward, single_tick = run_simulation(experiment_name, order_list, anomalies)

            for k in range(len(config.sim_targets)):
                reward[k].append(single_reward[k])
                tick[k].append(single_tick[k])

        plt.title('Rewards' + str(i))
        plt.boxplot(reward)
        plt.xticks(list(range(1, len(config.sim_targets) + 1)), config.sim_targets)
        plt.show()
        plt.close()

        plt.title('Processed Time' + str(i))
        plt.boxplot(tick)
        plt.xticks(list(range(1, len(config.sim_targets) + 1)), config.sim_targets)
        plt.show()
        plt.close()

        for j in range(len(config.sim_targets)):
            rewards[j].append(sum(reward[j]) / config.sim_count)
            ticks[j].append(sum(tick[j]) / config.sim_count)

    print('rewards: ', rewards)
    print('ticks: ', ticks)
