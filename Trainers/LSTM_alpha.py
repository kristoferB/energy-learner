from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from pandas import DataFrame
from pandas import concat
from numpy import concatenate
from numpy import sqrt
import numpy
import math
from matplotlib import pyplot
import h5py

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



data_set = pd.read_csv('..\Data\Formated\R30_original_fitted.csv', sep=',', header=None,dtype='a',
                       names=['t','ax1','ax2','ax3','ax4','ax5','ax6','Power','ConsumedPower' ],index_col=False)

values=data_set.values[1:,:]
values=values.astype('float32')

#data_set[].astype('float32')
#print(data_set.ax1[1:].astype('float32'))
#data_set.astype('float32')
# ax1 angle as numeric value
#ax1_ang=pd.to_numeric(data_set.ax1[1:])
# ax1 velocity

# ax1 angle
ax1_ang = values[:,1]
# ax1 velocity
ax1_vel = numpy.gradient(ax1_ang)
# ax1 Acceleration
ax1_acc = numpy.gradient(ax1_vel)

ax2_ang = values[:,2]
ax2_vel = numpy.gradient(ax2_ang)
ax2_acc = numpy.gradient(ax2_vel)

ax3_ang =values[:,3]
ax3_vel = numpy.gradient(ax3_ang)
ax3_acc = numpy.gradient(ax3_vel)

ax4_ang =values[:,4]
ax4_vel = numpy.gradient(ax4_ang)
ax4_acc = numpy.gradient(ax4_vel)

ax5_ang =values[:,5]
ax5_vel = numpy.gradient(ax5_ang)
ax5_acc = numpy.gradient(ax5_vel)

ax6_ang =values[:,6]
ax6_vel = numpy.gradient(ax6_ang)
ax6_acc = numpy.gradient(ax6_vel)

P = values[:,7]
Consumed_P = values[:,8]
pyplot.figure(1)
pyplot.plot(Consumed_P)
pyplot.show()

# Matrix with [Angle, Ang.Vel, Ang.Acc, PseudoPower]
augmented_data=numpy.array([ax1_ang,ax2_ang,ax3_ang,ax4_ang,ax5_ang,ax6_ang,
                ax1_vel,ax2_vel,ax3_vel,ax4_vel,ax5_vel,ax6_vel,
                ax1_acc,ax2_acc,ax3_acc,ax4_acc,ax5_acc,ax6_acc,
                ax1_vel*ax1_acc,ax2_vel*ax2_acc,ax3_vel*ax3_acc,ax4_vel*ax4_acc,ax5_vel*ax5_acc,ax6_vel*ax6_acc,
                Consumed_P])
augmented_data = numpy.transpose(augmented_data)

# Normalizing data
print('Augmented data shape')
print(augmented_data.shape)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(augmented_data)

# Transform to supervised learning structure
n_time_steps=1
n_features=25
reframed = series_to_supervised(scaled,n_in=n_time_steps,n_out=1)
print('reframed')
print(reframed.shape)
# drop n_in*26 to n_in

range_low=n_time_steps*n_features
range_upper = range_low+n_features-1 # -2 corresponds to the number of outputs
print('Ranges')
print(range_low)
print(range_upper)
reframed.drop(reframed.columns[[x for x in range(range_low,range_upper)]], axis=1, inplace=True)
print('reframed,after drop')
print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_samples = 2500
train = values[:n_train_samples, :]
print('train')
print(train.shape)
test = values[n_train_samples:, :]
# split into input and outputs
# using the last feature as output, Consumed_P
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

print(train_X.shape)
print(train_y.shape)

# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], n_time_steps, n_features))
test_X = test_X.reshape((test_X.shape[0], n_time_steps, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

'''
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# Save model
model.save('LSTM_alpha.h5')
'''
# Load Model
model=load_model('LSTM_alpha.h5')

# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

# make a prediction
yhat = model.predict(test_X)
print('----')
print(yhat.shape)

#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
test_X = test_X.reshape((test_X.shape[0], n_features*n_time_steps))

# invert scaling for forecast
print(test_X.shape)
print(yhat.shape)


inv_yhat = concatenate((yhat, test_X[:, 1:]),axis=1)


inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]


# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

pyplot.figure(2)
pyplot.plot(inv_y)
pyplot.plot(inv_yhat)
pyplot.show()
