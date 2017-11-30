from math import sqrt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from Trainers.SampledDataSet import SampledDataSet
import h5py
from datetime import datetime

"""
This is the same simple network as used in the abs acceleration example but builds on training from different files.
"""


def load_160406(file_number):
    folder_path = "../Data/Formated/160406/"
    file_name = "fitted_data_simple"+str(file_number)+".csv"
    data = pd.read_csv(folder_path + file_name, sep=",")
    return data


def load_171128(file_number):
    folder_path = "../Data/Formated/171128/"
    file_name = "EL_data_fitted"+str(file_number)+".csv"
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
# for i in range(0, 10):
#    training_data.append(SampledDataSet(load_160406, i))

for i in range(1, 6):
    training_data.append(SampledDataSet(load_171128, i))

(_, input_dimension) = training_data[0].features.shape

training_indexes = [0, 1, 2, 3]  # , 4, 5, 6, 7, 8, 9, 11, 12, 13]
validation_index = 4
print(len(training_data))

# Examples of what the arrays might look like
# training_data = [dataset_0, dataset_1, dataset_2, dataset_3 ...]
# training_indexes = [0,1,2,3,4, ... 10]
# validation_indexes = [11,12]


# design network
model = Sequential()
model.add(Dense(input_dimension, input_dim=input_dimension, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="sigmoid"))
model.add(Dense(12, activation="relu"))
model.add(Dense(6, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="mean_squared_error")

val_loss = []

for i in training_indexes:
    history = model.fit(training_data[i].scaled_features, training_data[i].scaled_output, batch_size=26, epochs=60,
                        validation_data=(training_data[validation_index].scaled_features, training_data[validation_index].scaled_output),
                        verbose=2, shuffle=False)
    val_loss += history.history['val_loss']
    print('Fitting finished for index %f' % i)


pyplot.plot(val_loss)
pyplot.xlabel("Epochs")
pyplot.ylabel("Loss")
pyplot.show()


y_hat = model.predict(training_data[validation_index].scaled_features)
y_inv = training_data[validation_index].scaler_power.inverse_transform(y_hat)
plt_pred, = pyplot.plot(training_data[validation_index].timestamps, y_inv, label='Prediction')
plt_meas, = pyplot.plot(training_data[validation_index].timestamps, training_data[validation_index].power, label='Measurement ' + str(validation_index+1))

# calculate RMSE
rmse = sqrt(mean_squared_error(y_inv, training_data[validation_index].power))
print('RMSE: %.3f' % rmse)
pyplot.legend(handles=[plt_pred, plt_meas])
pyplot.ylabel("Power [Watt]")
pyplot.xlabel("Time [s]")
pyplot.show()

should_save = input("Do you want to save the mode? (y/n)")

if should_save == 'y' or should_save == 'Y':
    model.save('../Models/' + datetime.now().strftime('%y%m%d-%H%M%S') + '_Kim_simple.h5')



# Save model
#

# Load Model
# model=load_model('LSTM_alpha.h5')


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