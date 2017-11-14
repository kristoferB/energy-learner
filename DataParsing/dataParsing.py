import pandas as pd

# Loading original data
# R10
# Energy
orig_energy_R10 = pd.read_csv('..\Data\Original\R10\E-R10_orig.csv', sep=',', header=None,names=['Time','Voltage','Current'], index_col=False, skiprows=9)
# Trajectory
orig_traj_R10 = pd.read_table('..\Data\Original\R10\orig_traj.txt', sep=' ', header=None,
                           names=['Time', 'Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5', 'Axis 6'],
                           skipinitialspace=True, skiprows=4,skipfooter=1,engine='python')

# R30
#Energy
orig_energy_R30 = pd.read_csv('..\Data\Original\R30\E-R30_original.csv', sep=',', header=None,names=['Time','Voltage','Current'], index_col=False, skiprows=9)
#Trajectory
orig_traj_R30= pd.read_table('..\Data\Original\R30\orig_traj.txt', sep=' ', header=None,
                           names=['Time', 'Axis 1', 'Axis 2', 'Axis 3', 'Axis 4','Axis 5', 'Axis 6'],
                           skipinitialspace=True, skiprows=4,skipfooter=1,engine='python')


# Loading Optimized data
# R30
# Energy
opt_energy_R30 = pd.read_csv('..\Data\Optimized\R30\E-R30_opt.csv', sep=',', header=None,names=['Time','Voltage','Current'], index_col=False, skiprows=9)
# Trajectory
opt_traj_R30 = pd.read_table('..\Data\Optimized\R30\opt_traj.txt', sep=' ', header=None,
                           names=['Time', 'Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5', 'Axis 6'],
                           skipinitialspace=True, skiprows=4,skipfooter=1,engine='python')
# R10
# Energy
opt_energy_R10_67s = pd.read_csv('..\Data\Optimized\R10\E-R10_opt_67s.csv', sep=',', header=None,names=['Time','Voltage','Current'], index_col=False, skiprows=9)
opt_energy_R10_firstTry = pd.read_csv('..\Data\Optimized\R10\E-R10_opt_firstTry.csv', sep=',', header=None,names=['Time','Voltage','Current'], index_col=False, skiprows=9)
opt_energy_R10_secondTry = pd.read_csv('..\Data\Optimized\R10\E-R10_opt_secondTry.csv', sep=',', header=None,names=['Time','Voltage','Current'], index_col=False, skiprows=9)


# Trajectory
opt_traj_R10 = pd.read_table('..\Data\Optimized\R10\opt_traj.txt', sep=' ', header=None,
                           names=['Time', 'Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5', 'Axis 6'],
                           skipinitialspace=True, skiprows=4,skipfooter=1,engine='python')
opt_traj_R10_first_second = pd.read_table('..\Data\Optimized\R10\R10_temp_first_second.txt', sep=' ', header=None,
                           names=['Time', 'Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5', 'Axis 6'],
                           skipinitialspace=True, skiprows=4,skipfooter=1,engine='python')
opt_traj_R10_temp_old = pd.read_table('..\Data\Optimized\R10\R10_temp_old.txt', sep=' ', header=None,
                           names=['Time', 'Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5', 'Axis 6'],
                           skipinitialspace=True, skiprows=4,skipfooter=1,engine='python')
