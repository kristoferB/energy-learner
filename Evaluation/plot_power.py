import numpy as np
from matplotlib import pyplot as plt
from Trainers.SampledDataSet import SampledDataSet

series = []

plot_titles = ['Series O', 'Series P', 'Series Q']

series.append(SampledDataSet(SampledDataSet.load_160406, 1))
series.append(SampledDataSet(SampledDataSet.load_171128, 1))
series.append(SampledDataSet(SampledDataSet.load_171201, 1))
plt_power = None
plt_pseudo = None

for i in range(0, len(series)):
    plt.subplot(3, 1, i+1)
    plt.ylabel('Power [kW]')
    pseudo_power = np.abs(np.sum(series[i].pseudo_power, axis=1))*50
    plt_pseudo, = plt.plot(series[i].timestamps, pseudo_power, label='Pseudo Power x 50')
    plt_power, = plt.plot(series[i].timestamps, series[i].power/1000, label='Measured Power')
    plt.title(plot_titles[i], x=0.2, y=0.7)

# Place a legend to the right of this smaller subplot.
plt.legend(handles=[plt_pseudo, plt_power], bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)
plt.xlabel('Time [s]')
plt.show()
