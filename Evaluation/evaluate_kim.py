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
from keras.utils import plot_model
import h5py

# Load Model
# model = load_model('../Models/' + '171205-0932' + '_Kim_simple.h5')
model = load_model('../Models/171128-1to5_LSTM5timesteps_100epochs.h5')

test_data = SampledDataSet(SampledDataSet.load_171201,1,5)
print("True energy")
print(test_data.energy)


y_hat = model.predict(test_data.scaled_features)
y_inv = test_data.scaler_power.inverse_transform(y_hat)
plt_pred, = pyplot.plot(test_data.timestamps, y_inv, label='Prediction')
plt_meas, = pyplot.plot(test_data.timestamps, test_data.power, label='Measurement')

dT=test_data.dT
y_energy = sum(abs(np.multiply(y_inv,dT)))  # [Joules]
y_energy= np.divide(y_energy, 3600000)
print("Predicted energy")
print(y_energy)


print("Relative energy error")
print((test_data.energy-y_energy)/test_data.energy)
# calculate RMSE
rmse = sqrt(mean_squared_error(y_inv, test_data.power))
print('RMSE: %.3f' % rmse)
pyplot.legend(handles=[plt_pred, plt_meas],fontsize=16)
pyplot.ylabel("Power [Watt]",fontsize=16)
pyplot.xlabel("Time [s]",fontsize=16)
pyplot.show()