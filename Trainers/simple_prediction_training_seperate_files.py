import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense

"""
This is the same simple network as used in the abs acceleration example but builds on training from different files.
"""

def load_160406(file_number):
    folder_path = "../Data/Formated/160406/"
    file_name = "output_py_"+str(file_number)+".csv"
    data = pd.read_csv(folder_path + file_name, sep=",")
    return data

def plot_matrix(values):
    print(values)
    (_, categories) = values.shape
    # plot each column
    pyplot.figure()
    for i in range(0, categories):
        pyplot.subplot(categories, 1, i+1)
        pyplot.plot(values[:, i])
        i += 1
    pyplot.show()


class SampledDataSet:
    """
        Use this data holder class to hold sampled data, parsed with a parsing script similar to the parse_KR40 script
        or provide another load_function. The loadfunction should return a pandas DataFrame with the following structure
        Timestamps | Angle 1 | Angle 2 | Angle 3 ... | Angle 6 | Power
          XX       |    XX   |   XX    |    XX   ... |   XX    |   XX
        And so on...

    """
    def __init__(self,load_function, file_number):
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
        # ensure that all data is float and rename to features to match with machine learning conventions
        self.features = self.pseudo_power.astype('float32')

        # normalize features
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_power = MinMaxScaler(feature_range=(0, 1))
        self.scaled_features = self.scaler_features.fit_transform(self.features)
        self.scaled_output = self.scaler_power.fit_transform(self.power)

    def __str__(self):
        return str(self.features)


training_data = []
for i in range(1, 10):
    training_data.append(SampledDataSet(load_160406, i))

# design network
model = Sequential()
model.add(Dense(6, input_dim=6, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="mean_squared_error")
for i in range(0, 7):
    history = model.fit(training_data[i].scaled_features, training_data[i].scaled_output, batch_size=26, epochs=60,
                        validation_data=(training_data[8].scaled_features, training_data[8].scaled_output), verbose=2, shuffle=False)

y_hat = model.predict(training_data[7].scaled_features)
y_inv = training_data[7].scaler_power.inverse_transform(y_hat)
pyplot.plot(y_inv)
pyplot.plot(training_data[7].power)
pyplot.show()