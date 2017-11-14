import pandas as pd

# Loading original data
# R10
# Energy
orig_energy_R10 = pd.read_csv('..\Data\Original\R10\E-R10_orig.csv', sep=',', header=None,names=['Time','Voltage','Current'], index_col=False, skiprows=9)
# Trajectory
orig_traj_R10 = pd.read_table('..\Data\Original\R10\orig_traj.txt', delim_whitespace=True, header=None,
                           names=['Time', 'Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5', 'Axis 6'],
                           skipinitialspace=True, skiprows=4,skipfooter=1,engine='python')


# R30
#Energy
orig_energy_R30 = pd.read_csv('..\Data\Original\R30\E-R30_original.csv', sep=',', header=None,names=['Time','Voltage','Current'], index_col=False, skiprows=9)
#Trajectory
orig_traj_R30= pd.read_table('..\Data\Original\R30\orig_traj.txt', delim_whitespace=True, header=None,
                           names=['Time', 'Axis 1', 'Axis 2', 'Axis 3', 'Axis 4','Axis 5', 'Axis 6'],
                           skipinitialspace=True, skiprows=4,skipfooter=1,engine='python')


# Loading Optimized data
# R10
# Energy
opt_energy_R10 = pd.read_csv('..\Data\Optimized\R10\E-R10_opt_67s.csv', sep=',', header=None,names=['Time','Voltage','Current'], index_col=False, skiprows=9)

# Trajectory
opt_traj_R10 = pd.read_table('..\Data\Optimized\R10\opt_traj.txt', delim_whitespace=True, header=None,
                           names=['Time', 'Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5', 'Axis 6'],
                           index_col=False,skiprows=5,skipfooter=1,engine='python')


# R30
# Energy
opt_energy_R30 = pd.read_csv('..\Data\Optimized\R30\E-R30_opt.csv', sep=',', header=None,names=['Time','Voltage','Current'], index_col=False, skiprows=9)
# Trajectory
opt_traj_R30 = pd.read_table('..\Data\Optimized\R30\opt_traj.txt', delim_whitespace=True, header=None,
                           names=['Time', 'Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5', 'Axis 6'],
                           skipinitialspace=True, skiprows=5,skipfooter=1,engine='python')



# print(orig_energy_R10)
# print(orig_traj_R10)
#
# print(orig_energy_R30)
# print(orig_traj_R30)
#
# print(opt_energy_R30)
# print(opt_traj_R30)
#
# print(opt_energy_R10)
# print(opt_traj_R10)
