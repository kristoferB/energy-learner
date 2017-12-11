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

training_data = []
for i in range(0, 10):
    training_data.append(SampledDataSet(SampledDataSet.load_160406, i))

for i in range(1, 6):
    training_data.append(SampledDataSet(SampledDataSet.load_171128, i))

for i in range(1, 4):
    training_data.append(SampledDataSet(SampledDataSet.load_171201, i))

training_indexes = [8, 9, 11, 12, 13, 14, 15, 16]
validation_index = 17
(_, input_dimension) = training_data[0].features.shape

# design network
model = Sequential()
model.add(Dense(input_dimension, input_dim=input_dimension, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="mean_squared_error")

for i in training_indexes:
    history = model.fit(training_data[i].scaled_features, training_data[i].scaled_output, batch_size=26, epochs=60,
                        validation_data=(training_data[validation_index].scaled_features, training_data[validation_index].scaled_output),
                        verbose=2, shuffle=False)


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
