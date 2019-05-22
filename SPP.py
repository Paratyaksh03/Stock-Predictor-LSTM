# IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import preprocessing 
from matplotlib.widgets import CheckButtons
from datetime import datetime
import mpl_finance
# FOR REPRODUCIBILITY
np.random.seed(7)

# IMPORTING DATASET 
dataset = pd.read_csv('AAPL_2.csv', usecols=[0,1,2,3,4],error_bad_lines=False)
dataset = dataset.reindex(index = dataset.index[::-1])
print(dataset)

# CODE FOR 1 MIN AND 5 MIN DATA
"""
dataset['Date_Time'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'])
dataset.set_index(['Date_Time'])
# YOU CAN DROP THE IRRELEVANT COLUMNS AFTER THIS CODE AND MAKE SURE TO ACCOMODATE THESE CHANGES TO THE PLOTTING BLOCK
"""

# CREATING OWN INDEX FOR FLEXIBILITY
obs = np.arange(1, len(dataset) + 1, 1)

# TAKING DIFFERENT INDICATORS FOR PREDICTION
OHLC_avg = dataset[['Open','High', 'Low', 'Close']].mean(axis = 1)
HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
close_val = dataset[['Close']]

fig, ax = plt.subplots()

l1,=ax.plot(dataset['Date'], OHLC_avg,visible=False,lw=1.5, color='r', label = 'OHLC avg')
l2,=ax.plot(dataset['Date'], HLC_avg, visible=False,lw=1.5, color='b', label = 'HLC avg')
l3,=ax.plot(dataset['Date'], close_val,lw=1, color='g', label = 'Closing price')

plt.subplots_adjust(left=0.21)


# PLOTTING ALL INDICATORS IN ONE PLOT
#l1,=plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
#l2,=plt.plot(obs, HLC_avg, 'b', label = 'HLC avg')
#l3,=plt.plot(obs, close_val, 'g', label = 'Closing price')
plt.legend(loc = 'best')

lines=[l1,l2,l3]

rax = plt.axes([0.01, 0.85, 0.2, 0.15])
labels = [str(line.get_label()) for line in lines]
visibility = [line.get_visible() for line in lines]
check = CheckButtons(rax, labels, visibility)

def func(label):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    plt.draw()

check.on_clicked(func)

plt.show()


# PREPARATION OF TIME SERIES DATAS
OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)

# TRAIN-TEST SPLIT
train_OHLC = int(len(OHLC_avg) * 0.75)
test_OHLC = len(OHLC_avg) - train_OHLC
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = preprocessing.new_dataset(train_OHLC, 1)
testX, testY = preprocessing.new_dataset(test_OHLC, 1)

# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1

# LSTM MODEL
model = Sequential()
model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
model.add(LSTM(16))
model.add(Dense(1))
model.add(Activation('linear'))

# MODEL COMPILING AND TRAINING
model.compile(loss='mean_squared_error', optimizer='adam') # Try SGD, adam, adagrad and compare!!!
model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
print(type(trainY))
print(type(trainX))


# TRAINING RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train RMSE: %.2f' % (trainScore))

# TEST RMSE
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE: %.2f' % (testScore))

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(OHLC_avg)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(OHLC_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict

# DE-NORMALIZING MAIN DATASET 
OHLC_avg = scaler.inverse_transform(OHLC_avg)

fig1, ax = plt.subplots()

#buttons 
ax5min = plt.axes([0.09, 0.9, 0.05, 0.071])
ax15min = plt.axes([0.14, 0.9, 0.05, 0.071])
ax30min = plt.axes([0.19, 0.9, 0.05, 0.071])
ax1hr = plt.axes([0.24, 0.9, 0.05, 0.071])
ax4hr = plt.axes([0.29, 0.9, 0.05, 0.071])
ax1w = plt.axes([0.34, 0.9, 0.05, 0.071])
ax1m = plt.axes([0.38, 0.9, 0.05, 0.071])

# PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
l4,=ax.plot(OHLC_avg, 'g', label = 'original dataset')
l5,=ax.plot(trainPredictPlot, 'r', label = 'training set')
l6,=ax.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
ax.legend(loc = 'best')
ax.title('ACTUAL STOCK PRICE')
ax.set_xlabel('Time in Days')
ax.set_ylabel('OHLC Value of Apple Stocks')

lines1=[l4,l5,l6]

rax = ax.axes([0.01, 0.85, 0.2, 0.15])
labels = [str(line.get_label()) for line in lines1]
visibility = [line.get_visible() for line in lines1]
check = CheckButtons(rax, labels, visibility)

def func(label):
    index = labels.index(label)
    lines1[index].set_visible(not lines1[index].get_visible())
    plt.draw()

check.on_clicked(func)




# PREDICT FUTURE VALUES
last_val = testPredict[-1]
last_val_scaled = last_val/last_val
next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
print ("Last Day Value:", np.asscalar(last_val))
print ("Next Day Value:", np.asscalar(last_val*next_val))
print (np.append(last_val, next_val))
ax.show()
