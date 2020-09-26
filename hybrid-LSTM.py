#%% Fin 794 - Group Project: Enhancing Time Series Momentum Strategies Using Deep Neural Networks
""" Created for Backtesting TSMOM-LSTM set up in paper - Han (Aaron) Xiao"""
# LSTM Part
# Part 1 - Preparation
# Module Load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
# load strategy setting from TSMOM
import TSMOM
import empyrical

#%%
# import raw daily return for all single asset/future
ast = TSMOM.ast
# import raw cum return for all single asset/future
cum_return = TSMOM.cum_return
# import raw daily signal for all single asset/future
X_t = TSMOM.X_t
normalized_daily = pd.DataFrame(index=TSMOM.daily_index)
df = pd.DataFrame(index=TSMOM.daily_index)
day_index_vol = pd.DataFrame(index=TSMOM.daily_index)
ind_return = pd.DataFrame(index=TSMOM.daily_index)

# Based on section V.B Backtest Description in paper,
# we need to rescale daily return to Normalised Returns
#for i in ast:
    #day_vol = ast[i].rolling(1).std()
    #day_index_vol = pd.concat([day_index_vol, day_vol], axis=1)

#for i in ast:
    #day_vol = ast[i].ewm(ignore_na=False,
                          #adjust=True,
                          #span=60,
                          #min_periods=0).std(bias=False)
    #day_index_vol = pd.concat([day_index_vol, day_vol], axis=1)
#del df

#for i in cum_return:
    #df[i + '-Ndaily'] = cum_return[i] / day_index_vol[i]
    #normalized_daily = pd.concat([normalized_daily, df[i + '-Ndaily']], axis=1)

# For now, I take MACD signal as input temporarily
#%% Train&Test read
train_url = '/Users/aaronx-mac/PycharmProjects/Learning/Fin794 Group Project/train.csv'
test_url = '/Users/aaronx-mac/PycharmProjects/Learning/Fin794 Group Project/test.csv'
train = pd.read_csv(train_url)
test = pd.read_csv(test_url)
train.set_index('Date', inplace=True)
test.set_index('Date', inplace=True)

#%%
train_array = np.array(train)
test_array = np.array(test)
#%% Creating a data structure with 63 timesteps and 1 output
X_train = []
y_train = []

for i in range(63, 9355):
    X_train.append(train_array[i - 63:i, 0])
    y_train.append(train_array[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#%% Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# from keras.losses import

#%% Initialising the RNN
regressor = Sequential()

#%% Add layer

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],60)))
regressor.add(Dropout(0.3))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

# Adding the output layer
regressor.add(Dense(units = 60))

#%%
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 256)


#%% Part 3 - Making the predictions and visualising the results
X_test = []
for i in range(63, 1043):
    X_test.append(test_array[i-63:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_X_t = regressor.predict(X_test)

#%%
predicted_X_t = pd.DataFrame(predicted_X_t)
predicted_X_t.info()

#%% put signal back into equation 1 & calculate daily return in portfolio
target_vol = 0.15

for i in ast:
    # 进行水平方向的合并
    df = pd.concat([ast[i], predicted_X_t[i + "-X_t"]], axis=1)

    day_vol = df[i].ewm(ignore_na=False,
                          adjust=True,
                          span=60,
                          min_periods=0).std(bias=False)

    # daily return based on equation (1) for individual asset
    df[i + '-ind_return'] = df[i] * df[i + "-X_t"] * target_vol / day_vol

    # convert to daily return for different asset in t
    ind_return = pd.concat([ind_return, df[i + '-ind_return']], axis=1)

# daily return in portfolio
ind_return['port_avg'] = ind_return.mean(skipna=1, axis=1)

#%% Print Results
print("Annualized Sharpe Ratio = ", empyrical.sharpe_ratio(ind_return['port_avg'], period='daily'))
print("Annualized Mean Returns = ", empyrical.annual_return(ind_return['port_avg'], period='daily'))
print("Annualized Standard Deviations = ", empyrical.annual_volatility(ind_return['port_avg'], period='daily'))
print("Max Drawdown (MDD) = ", empyrical.max_drawdown(ind_return['port_avg']))
print("Sortino ratio = ", empyrical.sortino_ratio(ind_return['port_avg'], period='daily'))
print("Calmar ratio = ", empyrical.calmar_ratio(ind_return['port_avg'], period='daily'))
