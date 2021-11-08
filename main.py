import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import anomaly
import config
import items
import warehouse


def show_values_on_bars(axs, scale):
    def _show_on_single_plot(ax):
        for i, p in enumerate(ax.patches):
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height() + 0.5
            if i < len(ax.patches) / 2:
                value = '{:d}'.format(int(p.get_height()))
            else:
                value = '{:.2f}'.format(p.get_height() / scale)
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


if __name__ == "__main__":
    df = pd.DataFrame({'Approach': [],
                       'Processed': [],
                       'Average Tick': []})

    for i in range(config.sim_count):
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        item_list = items.generate_item_list(False, experiment_name)
        order_list = items.generate_item_list(True, experiment_name)
        anomalies = anomaly.generate_anomaly(name=experiment_name)

        # for target in config.sim_targets:
        #     processed, average, ended = warehouse.run(target, item_list, order_list)
        #     print(target)
        #     print('ended: %d' % ended)
        #     print('total processed: %d' % processed)
        #     print('average processing: %.2f tick' % average)
        #     print()

        for target in config.sim_targets:
            processed, average, ended = warehouse.run(target, item_list, order_list, anomalies)
            target_df = pd.DataFrame({'Approach': [target], 'Processed': [processed], 'Average Tick': [average]})
            df = pd.concat([df, target_df])
            print(target)
            print('total processed: %d' % processed)
            print('average processing: %.2f tick' % average)
            print()

        df_melted = pd.melt(df, id_vars='Approach', var_name='Legend', value_name='value')

        mask = df_melted.Legend.isin(['Average Tick'])
        scale = df_melted[~mask].value.max() / len(item_list)
        df_melted.loc[mask, 'value'] = df_melted.loc[mask, 'value'] * scale

        fig, ax1 = plt.subplots()
        g = sns.barplot(x='Approach', y='value', hue='Legend', data=df_melted, ax=ax1)

        ax1.set_ylabel('Processed')
        ax1.set_ylim([0, len(item_list)])
        ax2 = ax1.twinx()

        ax2.set_ylim(ax1.get_ylim())
        ax2.set_yticklabels(np.round(ax1.get_yticks() / scale, 1))
        ax2.set_ylabel('Average Tick')

        show_values_on_bars(ax1, scale)

        plt.title('Processed items and average tick to process')
        plt.savefig('figure/' + experiment_name + '.png', dpi=300)
        plt.show()
        plt.close()
