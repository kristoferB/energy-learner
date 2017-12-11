from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from Trainers.SeriesToSupervised import series_to_supervised
class SampledDataSet:
    """
        Use this data holder class to hold sampled data, parsed with a parsing script similar to the parse_KR40 script
        or provide another load_function. The loadfunction should return a pandas DataFrame with the following structure
        Timestamps | Angle 1 | Angle 2 | Angle 3 ... | Angle 6 | Power
          XX       |    XX   |   XX    |    XX   ... |   XX    |   XX
        And so on...

    """
    def __init__(self, load_function, file_number,time_steps=0):
        data = load_function(file_number)
        # Pick timestamps from loaded data
        self.timestamps = data.values[:, 0:1]
        # Pick measured angles from loaded data
        self.angles = data.values[:, 1:-1]
        # Pick power consumption from loaded data
        self.power = data.values[:, -1:]
        # pseudo power = velocity * acceleration
        self.omega = np.gradient(self.angles, axis=0)
        self.acc = np.gradient(self.omega, axis=0)
        self.pseudo_power = np.multiply(self.omega, self.acc)
        all_values = np.append(self.angles, self.omega, axis=1)  # Angles 1-6, Omega 1-6
        all_values = np.append(all_values, self.acc, axis=1)
        all_values = np.append(all_values, self.pseudo_power, axis=1)
        # ensure that all data is float and rename to features to match with machine learning conventions
        self.features = all_values.astype('float32')

        # normalize features
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_power = MinMaxScaler(feature_range=(0, 1))
        self.scaled_features = self.scaler_features.fit_transform(self.features)
        self.scaled_output = self.scaler_power.fit_transform(self.power)


        self.dT=self.timestamps[1]-self.timestamps[0]
        self.energy = sum(np.multiply(self.power,self.dT))  # [Joules]
        self.energy = np.divide(self.energy, 3600000)  # [kWh]
        # count number of features
        n_features = self.scaled_features.shape[1]
        if time_steps != 0:
            # reframe x
            self.LSTM_features = series_to_supervised(self.scaled_features, n_in=time_steps, n_out=1)
            # reframe y
            self.output = self.scaled_output[time_steps:]
            range_low = time_steps*n_features
            range_upper = self.LSTM_features.shape[1]
            self.LSTM_features.drop(self.LSTM_features.columns[[x for x in range(range_low, range_upper)]], axis=1,
                                  inplace=True)
            # reshape x
            self.LSTM_features = self.LSTM_features.values.reshape((self.LSTM_features.shape[0], time_steps, n_features))
            self.scaled_features=self.LSTM_features
            self.power=data.values[time_steps:, -1:]
            self.timestamps = data.values[time_steps:, 0:1]


    def __str__(self):
        return str(self.features)

    def load_160406(file_number):
        folder_path = "../Data/Formated/160406/"
        file_name = "fitted_data_simple" + str(file_number) + ".csv"
        data = pd.read_csv(folder_path + file_name, sep=",")
        return data

    def load_171128(file_number):
        folder_path = "../Data/Formated/171128/"
        file_name = "EL_data_fitted"+str(file_number)+".csv"
        data = pd.read_csv(folder_path + file_name, sep=",")
        return data

    def load_171201(file_number):
        folder_path = "../Data/Formated/171201/"
        file_name = "EL_data_veri_fitted" + str(file_number) + ".csv"
        data = pd.read_csv(folder_path + file_name, sep=",")
        return data

