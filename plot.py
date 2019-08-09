from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from run_benchmarks import *

DATA_DIRECTORY = Path("results/data")
PLOT_DIRECTORY = Path("results/plots")


def get_colors(columns):
    base_color_map = cm.get_cmap('tab10', 10).colors
    # only support specific setup, fall back to base colour map in other cases
    if columns.names != ["method", "grayscale"]:
        print("Using standard color map")
        return base_color_map

    methods = columns.unique(level='method').values
    method_to_color = {method: base_color_map[i] for i, method in enumerate(methods)}

    color_map = []
    for method, grayscale in columns.values:
        color = method_to_color[method]
        if grayscale:
            color = color.copy()
            color[-1] = 0.5  # make transparent
        color_map.append(color)

    return color_map


def barchart(means, stds, filename):
    means.plot.bar(yerr=stds, color=get_colors(means.columns))
    plt.savefig(filename)
    plt.close()


def plot(benchmark, x_axis):
    data_path = DATA_DIRECTORY / (benchmark.__name__ + ".json")
    plot_base_path = PLOT_DIRECTORY / benchmark.__name__
    runtime_plot_path = "{}_{}.png".format(plot_base_path, "runtimes")
    framerate_plot_path = "{}_{}.png".format(plot_base_path, "framerate")

    data = pd.read_json(data_path)

    # delete columns without variation
    non_runtime_columns = [col for col in data.columns if col != "runtime"]
    for col in non_runtime_columns:
        if len(data[col].unique()) == 1:
            del data[col]

    index_columns = [col for col in data.columns if col != "runtime"]
    assert x_axis in index_columns

    if "method" in index_columns:  # method should be first
        index_columns = ["method"] + [col for col in index_columns if col != "method"]
    runtimes = data.set_index(index_columns)["runtime"]
    runtimes = runtimes.apply(pd.Series)

    if len(index_columns) == 1:
        means = runtimes.mean(axis=1).transpose()
        stds = runtimes.std(axis=1).transpose()
        means.plot.bar(yerr=stds)
        plt.savefig(runtime_plot_path)
        plt.close()

        framerate = 1.0 / runtimes
        means = framerate.mean(axis=1).transpose()
        stds = framerate.std(axis=1).transpose()
        means.plot.bar(yerr=stds)
        plt.savefig(framerate_plot_path)
        plt.close()

    else:

        means = runtimes.mean(axis=1).unstack(level=x_axis).transpose()
        stds = runtimes.std(axis=1).unstack(level=x_axis).transpose()
        means.plot.bar(yerr=stds, color=get_colors(means.columns))
        plt.savefig(runtime_plot_path)
        plt.close()

        framerate = 1.0/runtimes
        means = framerate.mean(axis=1).unstack(level=x_axis).transpose()
        stds = framerate.std(axis=1).unstack(level=x_axis).transpose()
        means.plot.bar(yerr=stds, color=get_colors(means.columns))
        plt.savefig(framerate_plot_path)
        plt.close()


PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
plot(FaceDetectionPersonsBenchmark, x_axis="image")
plot(FaceDetectionScaleBenchmark, x_axis="scale")
plot(FaceCroppingBenchmark, x_axis="scale")
plot(FeatureExtractionBenchmark, x_axis="scale")
plot(FaceMatchingBenchmark, x_axis="num_registered_persons")
