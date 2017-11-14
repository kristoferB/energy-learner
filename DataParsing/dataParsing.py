import pandas as pd

energy_dataset = pd.read_csv('Data\Original\R30\E-R30_original.csv', sep=',', header=None
                             ,names=['Time','Voltage','Current'], index_col=False, skiprows=9)
#print(energy_dataset)

traj_dataset= pd.read_table('Data\Original\R30\orig_traj.txt', sep=' ', header=None,
                            names=['Time', 'Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5', 'Axis 6'],
                            skipinitialspace=True, skiprows=4, skipfooter=1)
print(traj_dataset)
