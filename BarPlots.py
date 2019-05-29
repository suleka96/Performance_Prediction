import atexit
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import itertools

def bar_plot():

    filename = "Latency_MAPE.csv"

    df = pd.read_csv(filename)

    sns.set_context("paper", font_scale=2)

    ax =sns.barplot(x='ML_Model', y='MAPE', hue='Dataset', data=df)
    ax.set_yscale('log')


    num_locations = len(df.ML_Model.unique())
    hatches = itertools.cycle(['/', '//', '\\', '-'])

    for i, bar in enumerate(ax.patches):
        if i % num_locations == 0:
            hatch = next(hatches)
        bar.set_hatch(hatch)


    plt.xlabel("ML Model", fontdict={'size': 19})
    plt.ylabel("MAPE (log)", fontdict={'size': 19})

    plt.legend(loc=9, prop={'size': 17})
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=4, prop = {'size' : 17})
    plt.show()


bar_plot()




