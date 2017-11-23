from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy
import math
from matplotlib import pyplot
import h5py


data_set = pd.read_csv('..\Data\Formated\R30_original_fitted.csv', sep=',', header=None,
                       names=['t','ax1','ax2','ax3','ax4','ax5','ax6','Power','ConsumedPower' ],index_col=False)
# ax1 angle as numeric value
ax1_ang=pd.to_numeric(data_set.ax1[1:])
# ax1 velocity
ax1_vel = numpy.gradient(ax1_ang)
# ax1 Acceleration
ax1_acc = numpy.gradient(ax1_vel)

ax2_ang=pd.to_numeric(data_set.ax2[1:])
ax2_vel = numpy.gradient(ax2_ang)
ax2_acc = numpy.gradient(ax2_vel)

ax3_ang=pd.to_numeric(data_set.ax3[1:])
ax3_vel = numpy.gradient(ax3_ang)
ax3_acc = numpy.gradient(ax3_vel)

ax4_ang=pd.to_numeric(data_set.ax4[1:])
ax4_vel = numpy.gradient(ax4_ang)
ax4_acc = numpy.gradient(ax4_vel)

ax5_ang=pd.to_numeric(data_set.ax5[1:])
ax5_vel = numpy.gradient(ax5_ang)
ax5_acc = numpy.gradient(ax5_vel)

ax6_ang=pd.to_numeric(data_set.ax6[1:])
ax6_vel = numpy.gradient(ax6_ang)
ax6_acc = numpy.gradient(ax6_vel)

P=pd.to_numeric(data_set.Power[1:])
Consumed_P=pd.to_numeric(data_set.ConsumedPower[1:])
# input vectors
pseudo_power = [ax1_vel*ax1_acc,ax2_vel*ax2_acc,ax3_vel*ax3_acc,ax4_vel*ax4_acc,ax5_vel*ax5_acc,ax6_vel*ax6_acc]

feature_matrix = [ax1_ang,ax2_ang,ax3_ang,ax4_ang,ax5_ang,ax6_ang,ax1_vel,ax2_vel,ax3_vel,ax4_vel,ax5_vel,ax6_vel,
                  ax1_acc,ax2_acc,ax3_acc,ax4_acc,ax5_acc,ax6_acc]
feature_matrix=numpy.absolute(feature_matrix)
feature_matrix =numpy.transpose(feature_matrix)
pseudo_power = numpy.transpose(pseudo_power)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(pseudo_power)
X_train = X[:3000,:]
X_test= X[3000:,:]
# Supposed output of the model
Y=[Consumed_P]

#print(P)
Y=numpy.transpose(Y)
Y=scaler.fit_transform(Y)

Yp_train=Y[:3000]
Yp_test=Y[3000:]
Ycp=Consumed_P

#print(Y)
# create model

model = Sequential()
model.add(Dense(6, input_dim=6, activation='sigmoid'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# save model
model.save('vanilla_1.h5')
# Load Model

#model=load_model('vanilla_1.h5')
# evaluate the model


#scores = model.evaluate(X_test[:,12:], Yp_test)
pred = model.predict(X)

inv_pred=scaler.inverse_transform(pred)
inv_Y_test=scaler.inverse_transform(Y)

E=math.sqrt(mean_squared_error(inv_Y_test,inv_pred))
print(E)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



#print(pred)
pyplot.plot(inv_Y_test)
pyplot.plot(inv_pred)
#pyplot.plot(Consumed_P-inv_pred[:,0])

'''
pyplot.figure(2)
pyplot.title('Velocity')
pyplot.subplot(2,3,1)
pyplot.plot(feature_matrix[:,6])
pyplot.subplot(2,3,2)
pyplot.plot(feature_matrix[:,7])
pyplot.subplot(2,3,3)
pyplot.plot(feature_matrix[:,8])
pyplot.subplot(2,3,4)
pyplot.plot(feature_matrix[:,9])
pyplot.subplot(2,3,5)
pyplot.plot(feature_matrix[:,10])
pyplot.subplot(2,3,6)
pyplot.plot(feature_matrix[:,11])


pyplot.figure(3)
pyplot.title('Acceleration')
pyplot.subplot(2,3,1)
pyplot.plot(feature_matrix[:,12])
pyplot.subplot(2,3,2)
pyplot.plot(feature_matrix[:,13])
pyplot.subplot(2,3,3)
pyplot.plot(feature_matrix[:,14])
pyplot.subplot(2,3,4)
pyplot.plot(feature_matrix[:,15])
pyplot.subplot(2,3,5)
pyplot.plot(feature_matrix[:,16])
pyplot.subplot(2,3,6)
pyplot.plot(feature_matrix[:,17])
'''
pyplot.show()
