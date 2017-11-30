from math import sqrt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from Trainers.SampledDataSet import SampledDataSet
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

training_data = []
for i in range(1, 10):
    training_data.append(SampledDataSet(load_160406, i))

(_, input_dimension) = training_data[0].features.shape

# design network
model = Sequential()
model.add(Dense(input_dimension, input_dim=input_dimension, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(6, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="mean_squared_error")

val_loss = []

for i in range(8):
    history = model.fit(training_data[i].scaled_features, training_data[i].scaled_output, batch_size=26, epochs=160,
                        validation_data=(training_data[8].scaled_features, training_data[8].scaled_output), verbose=2,
                        shuffle=False)
    val_loss += history.history['val_loss']
    print('Fitting finished for index %f' % i)

pyplot.plot(val_loss)
pyplot.xlabel("Epochs")
pyplot.ylabel("Label")
pyplot.show()

y_hat = model.predict(training_data[8].scaled_features)
y_inv = training_data[8].scaler_power.inverse_transform(y_hat)
plt_pred, = pyplot.plot(training_data[8].timestamps, y_inv, label='Prediction')
plt_meas, = pyplot.plot(training_data[8].timestamps, training_data[8].power, label='Measurement 8')

# calculate RMSE
rmse = sqrt(mean_squared_error(y_inv, training_data[8].power))
print('Test RMSE: %.3f' % rmse)
rmse = sqrt(mean_squared_error(training_data[0].power, training_data[8].power))
print('Seq diff between run 1 and 8 RMSE: %.3f' % rmse)
pyplot.legend(handles=[plt_pred, plt_meas])
pyplot.ylabel("Power [Watt]")
pyplot.xlabel("Time [s]")
pyplot.show()

#
# Prediction on another datafile
#
"""
def load_R30_original(filenumber):
    folder_path = "../Data/Formated/"
    file_name = "R30_original_fitted.csv"
    data = pd.read_csv(folder_path + file_name, sep=",")
    data.drop('Power [Watt]', axis=1, inplace=True)
    return data


other_data = SampledDataSet(load_R30_original, 1)
y_hat = model.predict(other_data.scaled_features)
y_inv = other_data.scaler_power.inverse_transform(y_hat)
plt_pred, = pyplot.plot(other_data.timestamps, y_inv, label='Prediction')
plt_meas, = pyplot.plot(other_data.timestamps, other_data.power, label='Measurement')

# calculate RMSE
rmse = sqrt(mean_squared_error(y_inv, other_data.power))
print('Test RMSE: %.3f' % rmse)
pyplot.legend(handles=[plt_pred, plt_meas])
pyplot.ylabel("Power [Watt]")
pyplot.xlabel("Time [s]")
pyplot.show()
"""