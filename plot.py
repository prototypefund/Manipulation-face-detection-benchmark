import pandas as pd
import matplotlib.pyplot as plt

RUNTIMES_FILE = "results/runtimes.json"

runtimes = pd.read_json(RUNTIMES_FILE)
runtimes = runtimes.set_index(["method", "scale", "grayscale"])["runtime"]
runtimes = runtimes.apply(pd.Series)


means = runtimes.mean(axis=1).unstack(level="scale").transpose()
stds = runtimes.std(axis=1).unstack(level="scale").transpose()
means.plot.bar(yerr=stds)
plt.savefig("results/runtime.png")

framerate = 1.0/runtimes
means = framerate.mean(axis=1).unstack(level="scale").transpose()
stds = framerate.std(axis=1).unstack(level="scale").transpose()
means.plot.bar(yerr=stds)
plt.savefig("results/framerate.png")
