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
from keras.utils import plot_model

"""
This is the same simple network as used in the abs acceleration example but builds on training from different files.
"""

"""
y_true should contain [Power | Pseudo Power       ] y_pred should contain [w1, w2, w3, w4, w5, w6]
                      [  x   |  a1,a2,a3,a4,a5,a6 ]
                      [  x   |  a1,a2,a3,a4,a5,a6 ]
                      [  x   |  a1,a2,a3,a4,a5,a6 ]
"""
def custom_loss_function(y_true, y_pred):
    power = y_true[0]
    pseudo_power = y_true[1, :]
    pseudo_power = np.multiply(pseudo_power, y_pred)
    return np.sum(np.abs(np.power(pseudo_power, 2)-np.power(power, 2)))


training_data = []
for i in range(0, 10):
    training_data.append(SampledDataSet(SampledDataSet.load_160406, i))

(_, input_dimension) = training_data[0].features.shape

training_indexes = [0, 1, 2, 3, 4, 5, 6, 7] # , 8, 9, 10, 11, 12, 13, 14, 15, 16]
validation_index = 8
print(len(training_data))

# Examples of what the arrays might look like
# training_data = [dataset_0, dataset_1, dataset_2, dataset_3 ...]
# training_indexes = [0,1,2,3,4, ... 10]
# validation_indexes = [11,12]


# design network
model = Sequential()
model.add(Dense(input_dimension, input_dim=input_dimension, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(6, activation="relu"))
model.add(Dense(6, activation="sigmoid"))

model.compile(optimizer="adam", loss=custom_loss_function)

val_loss = []

for i in training_indexes:
    y_true = (training_data[i].scaled_output, training_data[i].scaled_features[:, -6:])
    history = model.fit(training_data[i].scaled_features, y_true, batch_size=26, epochs=10,
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