import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot
from math import floor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def load_r30():
    """
    Loads the R30_original_fitted.csv data.
    :return: (Column names, values matrix)
    """
    dataset = pd.read_csv('../Data/Formated/R30_original_fitted.csv')
    dataset.drop('Power [Watt]', axis=1, inplace=True)
    values = dataset.values


    return (dataset.columns.values, values)



def plot_axis_and_power(xvalues, values, titles=['Axis 1','Axis 2','Axis 3','Axis 4','Axis 5','Axis 6','Power Consumption'], fig=1):
    (_,categories) = values.shape
    # plot each column
    pyplot.figure()
    for i in range(0,categories):
        pyplot.subplot(categories, 1, i+1)
        pyplot.plot(xvalues[:, 0], values[:, i])
        pyplot.title(titles[i], y=0.5, loc='right')
        pyplot.xlabel("Time [s]")
        i += 1



(columns, values) = load_r30()

# Pick timestamps from loaded data
timestamps = values[:, 0:1]
# Pick measured angles from loaded data
angles = values[:, 1:-1]
# Pick power consumption from loaded data
power = values[:, -1:]

plot_axis_and_power(timestamps, angles)
# Derivate angles to get angular rate
omega = np.gradient(angles, axis=0)
# Derivate angular rate to get angular acceleration
acc = np.gradient(omega, axis=0)

# Take absolute value of acceleration since orientation does not effect power consumption
acc_abs = np.absolute(acc)

"""
#Plot the results
plot_axis_and_power(timestamps, omega, titles=['Omega 1','Omega 2','Omega 3','Omega 4','Omega 5','Omega 6','Power Consumption'], fig=2)
plot_axis_and_power(timestamps, acc_abs, titles=['Acc 1','Acc 2','Acc 3','Acc 4','Acc 5','Acc 6','Power Consumption'], fig=3)
pyplot.figure(4)
pyplot.plot(timestamps, power)
pyplot.show()
"""

# ensure that all data is float and rename to features to match with machine learning conventions
features = acc_abs.astype('float32')

# normalize features
scaler_input = MinMaxScaler(feature_range=(0, 1))
scaler_output = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_input.fit_transform(features)
scaled_output = scaler_output.fit_transform(power)

# Use 70% as training data
(sample_length, _) = scaled_features.shape
n_train_samples = floor(sample_length*0.7)

train_X = scaled_features[:n_train_samples, :]
train_Y = scaled_output[:n_train_samples, :]
test_X = scaled_features[n_train_samples:, :]
test_Y = scaled_output[n_train_samples:, :]

# Plot testing and training data
pyplot.figure(1)
# Train plot
pyplot.subplot(7,2,1)
pyplot.plot(train_X[:, 0])
pyplot.title("Acc 1 Train", y=0.5, loc="right")
pyplot.subplot(7,2,3)
pyplot.plot(train_X[:, 1])
pyplot.title("Acc 2 Train", y=0.5, loc="right")
pyplot.subplot(7,2,5)
pyplot.plot(train_X[:, 2])
pyplot.title("Acc 3 Train", y=0.5, loc="right")
pyplot.subplot(7,2,7)
pyplot.plot(train_X[:, 3])
pyplot.title("Acc 4 Train", y=0.5, loc="right")
pyplot.subplot(7,2,9)
pyplot.plot(train_X[:, 4])
pyplot.title("Acc 4 Train", y=0.5, loc="right")
pyplot.subplot(7,2,11)
pyplot.plot(train_X[:, 5])
pyplot.title("Acc 5 Train", y=0.5, loc="right")


#Test plot
pyplot.subplot(7,2,2)
pyplot.plot(test_X[:, 0])
pyplot.title("Acc 1 Test", y=0.5, loc="right")
pyplot.subplot(7,2,4)
pyplot.plot(test_X[:, 1])
pyplot.title("Acc 2 Test", y=0.5, loc="right")
pyplot.subplot(7,2,6)
pyplot.plot(test_X[:, 2])
pyplot.title("Acc 3 Test", y=0.5, loc="right")
pyplot.subplot(7,2,8)
pyplot.plot(test_X[:, 3])
pyplot.title("Acc 4 Test", y=0.5, loc="right")
pyplot.subplot(7,2,10)
pyplot.plot(test_X[:, 4])
pyplot.title("Acc 5 Test", y=0.5, loc="right")
pyplot.subplot(7,2,12)
pyplot.plot(test_X[:, 5])
pyplot.title("Acc 6 Test", y=0.5, loc="right")

#Output train
pyplot.subplot(7,2,13)
pyplot.plot(train_Y[:, 0])
pyplot.title("Power", y=0.5, loc="right")

#Output test
pyplot.subplot(7,2,14)
pyplot.plot(test_Y[:, 0])
pyplot.title("Power", y=0.5, loc="right")


pyplot.show()

# design network
model = Sequential()
model.add(Dense(6, input_dim=6, activation="sigmoid"))
model.add(Dense(12, activation="sigmoid"))
model.add(Dense(12, activation="sigmoid"))
model.add(Dense(12, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="mean_squared_error")
history = model.fit(train_X, train_Y, batch_size=16, epochs=60,
                    validation_data=(test_X, test_Y), verbose=2, shuffle=False)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


y_hat = model.predict(test_X)

pyplot.plot(y_hat)
pyplot.plot(test_Y)
pyplot.show()
