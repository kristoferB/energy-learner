from math import sqrt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from Trainers.SampledDataSet import SampledDataSet
from Trainers.SampledDataSet import load_171128
import h5py
from datetime import datetime


# Load Model
model = load_model('../Models/' + '171130-151527' + '_Kim_simple.h5')
test_data = SampledDataSet(load_171128, 2)


y_hat = model.predict(test_data.scaled_features)
y_inv = test_data.scaler_power.inverse_transform(y_hat)
plt_pred, = pyplot.plot(test_data.timestamps, y_inv, label='Prediction')
plt_meas, = pyplot.plot(test_data.timestamps, test_data.power, label='Measurement')

# calculate RMSE
rmse = sqrt(mean_squared_error(y_inv,test_data.power))
print('RMSE: %.3f' % rmse)
pyplot.legend(handles=[plt_pred, plt_meas])
pyplot.ylabel("Power [Watt]")
pyplot.xlabel("Time [s]")
pyplot.show()

# yolo