import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import config
import smart_warehouse


def show_values_on_bars(axs, scale):
    def _show_on_single_plot(ax):
        for i, p in enumerate(ax.patches):
            if i < len(ax.patches) / 2:
                value = int(p.get_height())
            else:
                value = int(p.get_height() / scale)

            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height() + 1

            value = '{:d}'.format(value)
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def run_simulation(name, order_list, anomalies):
    df = pd.DataFrame({'Approach': [],
                       'Reward': [],
                       'Processed': []})
    rewards = []
    ticks = []

    for target in config.sim_targets:
        reward, tick = smart_warehouse.run(name, target, order_list, anomalies)
        target_df = pd.DataFrame({'Approach': [target], 'Reward': [reward], 'Processed': [tick]})
        df = pd.concat([df, target_df])
        print(target)
        print('total: %d' % tick)
        print('reward: %.2f' % reward)
        print()
        rewards.append(reward)
        ticks.append(tick)

    df_melted = pd.melt(df, id_vars='Approach', var_name='Legend', value_name='value')

    mask = df_melted.Legend.isin(['Processed'])
    scale = df_melted[~mask].value.mean() / df_melted[mask].value.mean()
    df_melted.loc[mask, 'value'] = df_melted.loc[mask, 'value'] * scale

    fig, ax1 = plt.subplots()
    g = sns.barplot(x='Approach', y='value', hue='Legend', data=df_melted, ax=ax1)

    ax1.set_ylabel('Reward')
    ax2 = ax1.twinx()

    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticklabels(np.round(ax1.get_yticks() / scale, 0))
    ax2.set_ylabel('Tick')

    show_values_on_bars(ax1, scale)

    plt.title(name)
    # plt.savefig('figure/' + name + '.png', dpi=300)
    # plt.show()
    plt.close()

    return np.array(rewards), np.array(ticks)


if __name__ == "__main__":
    rewards = np.array([0.0] * len(config.sim_targets))
    ticks = np.array([0.0] * len(config.sim_targets))

    for i in range(config.sim_count):
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        order_list = smart_warehouse.generate_order_list(experiment_name)
        anomalies = smart_warehouse.generate_anomaly(is_real=True, name=experiment_name)
        # anomalies = [[], [], []]

        single_reward, single_tick = run_simulation(experiment_name, order_list, anomalies)

        rewards += single_reward
        ticks += single_tick

    rewards /= config.sim_count
    ticks /= config.sim_count

    for i, target in enumerate(config.sim_targets):
        print()
        print(target)
        print('average reward: ', rewards[i])
        print('average time: ', ticks[i])
