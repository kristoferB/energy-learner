import pandas as pd
from scipy import signal
from matplotlib import pyplot
import numpy as np

enable_plots = 1
# Downsampling factor 10 kHz * 12 ms = 120
downsampling_factor = 10*12

# Number difference between files i.e 1.txt, 2.txt, 3.txt, 4.txt and so on
for file_number in range(1, 10):
    folder_path = '../Data/Raw/160406- KR30-Sensitivity Analises/Results/jerk/'
    output_path = '../Data/Raw/160406- KR30-Sensitivity Analises/Results/jerk/'
    file_name_power = '160413 - E - jerk-' + str(file_number) + '.txt'
    file_name_trajectory = str(file_number) + '.txt'

    sampled_power_data = pd.read_table(folder_path + file_name_power, sep='\t', header=None,
                                       skipinitialspace=True, skiprows=9, skipfooter=1, usecols=range(7), engine='python')

    sampled_power_data.columns = ["Time", "Volt 1", "Volt 2", "Volt 3", "Amp 1", "Amp 2", "Amp 3"]

    # Power = Voltage * Current
    sampled_power_data["Power 1"] = sampled_power_data["Volt 1"].values * sampled_power_data["Amp 1"].values
    sampled_power_data["Power 2"] = sampled_power_data["Volt 2"].values * sampled_power_data["Amp 2"].values
    sampled_power_data["Power 3"] = sampled_power_data["Volt 3"].values * sampled_power_data["Amp 3"].values

    # Total power = Power phase 1 + Power phase 2 + Power phase 3
    sampled_power_data["Power"] = sampled_power_data["Power 1"].values + sampled_power_data["Power 2"].values + sampled_power_data["Power 3"].values

    # Drop everything but sampletimes and power
    sampled_power_data = sampled_power_data.drop('Volt 1', 1)
    sampled_power_data = sampled_power_data.drop('Volt 2', 1)
    sampled_power_data = sampled_power_data.drop('Volt 3', 1)

    sampled_power_data = sampled_power_data.drop('Amp 1', 1)
    sampled_power_data = sampled_power_data.drop('Amp 2', 1)
    sampled_power_data = sampled_power_data.drop('Amp 3', 1)

    sampled_power_data = sampled_power_data.drop('Power 1', 1)
    sampled_power_data = sampled_power_data.drop('Power 2', 1)
    sampled_power_data = sampled_power_data.drop('Power 3', 1)


    # Downsample
    result = signal.decimate(sampled_power_data.values, downsampling_factor, ftype='fir', axis=0)
    power_data = pd.DataFrame(data=result, columns=["Time", "Power"])

    if enable_plots:
        # Plot the downsampled signal
        pyplot.plot(sampled_power_data["Time"].values, sampled_power_data["Power"].values)
        pyplot.plot(power_data["Time"].values, power_data["Power"].values)
        pyplot.title("Downsampling", y=0.5, loc="right")
        pyplot.xlabel("Time [s]")
        pyplot.show()

    # Load trajectory
    data = pd.read_table(folder_path+file_name_trajectory, sep='\t', header=None,
                                       skipinitialspace=True, skiprows=6, skipfooter=1, usecols=range(7), engine='python')
    data.columns = ["Time", "Angle 1", "Angle 2", "Angle 3", "Angle 4", "Angle 5", "Angle 6"]

    # Derivative of angle to get angular speed
    omega = np.gradient(data.values[:, 1:], axis=0)

    # Square angular speed to remove negatives
    omega_sq = np.sum(np.square(omega), axis=1)

    # Cross-correlate the signals to find time lag
    corr = signal.correlate(power_data["Power"].values, omega_sq, "full")
    dt = np.mean(np.diff(data["Time"].values))
    shift = (np.argmax(corr) - len(omega_sq)) * dt

    start_index = (np.argmin(abs(data["Time"].values[0]+shift-power_data["Time"].values)))-1
    end_index = (np.argmin(abs(data["Time"].values[-1]+shift-power_data["Time"].values)))

    # Select power samples corresponding to angle measurements
    power = power_data["Power"].values[start_index:end_index]

    if enable_plots:
        pyplot.figure()
        pyplot.subplot(4, 1, 1)
        pyplot.plot(data["Time"].values, omega_sq[:])
        pyplot.title("Omega^2", y=0.5, loc="right")
        pyplot.subplot(4, 1, 2)
        pyplot.plot(power_data["Time"].values, power_data["Power"].values)
        pyplot.xlabel("Time [s]")
        pyplot.title("Power", y=0.5, loc="right")

        pyplot.subplot(4, 1, 3)
        pyplot.title("Correlation", y=0.5, loc="right")
        pyplot.plot(corr)

        pyplot.subplot(4, 1, 4)
        pyplot.plot(power)
        pyplot.plot(omega_sq*3000)
        pyplot.title("Selected power samples", y=0.5, loc="right")

        pyplot.show()


    data["Power"] = power

    # Write
    data.to_csv(output_path+"output_py_"+str(file_number)+".csv", sep=',', encoding='utf-8')