from keras.models import load_model
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from matplotlib import pyplot
from Trainers.SampledDataSet import SampledDataSet
from Trainers.SeriesToSupervised import series_to_supervised
import numpy as np
import h5py



# Define data structure
training_data = []
test_data = []
training_data.append(SampledDataSet(SampledDataSet.load_171128, 1))
training_data.append(SampledDataSet(SampledDataSet.load_171128, 2))
training_data.append(SampledDataSet(SampledDataSet.load_171128, 3))
training_data.append(SampledDataSet(SampledDataSet.load_171128, 4))
training_data.append(SampledDataSet(SampledDataSet.load_171128, 5))
test_data.append(SampledDataSet(SampledDataSet.load_171201, 3))

print('train data')
print(training_data[0].scaled_features.shape)

# convert training time-series to supervised learning
n_time_steps=5
n_features=24
reframed_train_x = series_to_supervised(training_data[0].scaled_features,n_in=n_time_steps,n_out=1)
reframed_test_x = series_to_supervised(test_data[0].scaled_features,n_in=n_time_steps,n_out= 1)

print('reframed')
print(reframed_train_x.shape)

# Drop the features that are not predicted
range_low=n_time_steps*(n_features)
range_upper_train = reframed_train_x.shape[1]
range_upper_test = reframed_test_x.shape[1]
print('Ranges')
print(range_low)
print(range_upper_train)
print(range_upper_test)
reframed_train_x.drop(reframed_train_x.columns[[x for x in range(range_low,range_upper_train)]], axis=1, inplace=True)
reframed_test_x.drop(reframed_test_x.columns[[x for x in range(range_low,range_upper_test)]], axis=1, inplace=True)


print('reframed,after drop')
print(reframed_train_x.shape)
print(reframed_test_x.shape)

# Transforming input-data to supervised introduces a time-lag, thus the output data  must be cut correspondingly
train_y =  training_data[0].scaled_output
train_y = train_y[n_time_steps:]
test_y =  test_data[0].scaled_output
test_y = test_y[n_time_steps:]

# Final input
train_X = reframed_train_x.values
test_X = reframed_test_x.values

print(train_X.shape)


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_time_steps, n_features))
test_X = test_X.reshape((test_X.shape[0], n_time_steps, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# design network
model = Sequential()
model.add(LSTM(50,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=20, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# Save model
#model.save('..\Models\LSTM_EL_data_fitted1_50timesteps_50epochs_tanh_forgetBias.h5')



# Load Model
#model=load_model('...\Models\LSTM_EL_data_fitted1_50timesteps_50epochs.h5')



# make a prediction

yhat = model.predict(test_X)
print('----')
print(yhat.shape)

#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
test_X = test_X.reshape([test_X.shape[0], n_features*n_time_steps])
print(test_X.shape)

# invert scaling for forecast
inv_yhat = test_data[0].scaler_power.inverse_transform(yhat)
print('inv_yhat shape')
print(inv_yhat.shape)


# invert scaling for actual
test_y = test_y.reshape([len(test_y), 1])
inv_y = test_data[0].scaler_power.inverse_transform(test_y)
print('inv_y shape')
print(inv_y.shape)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

Esum=sum(inv_y-inv_yhat)
print("Energy")
print(test_data[0].energy)
print(Esum)

pyplot.figure(2)
pyplot.plot(inv_y)
pyplot.plot(inv_yhat)
pyplot.show()

should_save = input("Do you want to save the mode? (y/n)")

if should_save == 'y' or should_save == 'Y':
    name= input("Maodel specifications")
    model.save('../Models/' + '171128-1to5_LSTM'+name+'.h5')
