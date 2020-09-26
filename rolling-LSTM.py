#%% 1. Import Libraries
#Standard libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time

# library for sampling
from scipy.stats import uniform

# libraries for Data Download
import datetime
from pandas_datareader import data as pdr

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Keras
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier

#%% 2. Create Classes
# Define a callback class
# Resets the states after each epoch (after going through a full time series)
class ModelStateReset(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.reset_states()
reset=ModelStateReset()

# Different Approach
#class modLSTM(LSTM):
#    def call(self, x, mask=None):
#        if self.stateful:
#             self.reset_states()
#        return super(modLSTM, self).call(x, mask)

#%% 3. Write Functions¶
# Function to create an LSTM model, required for KerasClassifier
def create_shallow_LSTM(epochs=1,
                        LSTM_units=1,
                        num_samples=1,
                        look_back=1,
                        num_features=None,
                        dropout_rate=0.3,
                        recurrent_dropout=0,
                        verbose=0):
    model = Sequential()

    model.add(LSTM(units=LSTM_units,
                   batch_input_shape=(num_samples, look_back, num_features),
                   stateful=True,
                   recurrent_dropout=recurrent_dropout))

    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.he_normal(seed=1)))

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model

#%% 4. Data
# 4.1 Import Raw Data
import TSMOM
cum_return = TSMOM.cum_return
signal = TSMOM.Signal
ast = TSMOM.ast

#%% 4.2 Create Features
AD1 = pd.DataFrame(index=TSMOM.daily_index)
AD1 = pd.concat([AD1, ast['AD1']], axis=1)

# Compute daily returns using the pandas rolling sum function
AD1['Ret_1w'] = AD1['AD1'].rolling(window=21).sum()
AD1['Ret_2w'] = AD1['AD1'].rolling(window=63).sum()
AD1['Ret_3w'] = AD1['AD1'].rolling(window=126).sum()
AD1['Ret_4w'] = AD1['AD1'].rolling(window=252).sum()

# Compute Volatility using the pandas rolling standard deviation function
AD1['Vol_w'] = AD1['AD1'].rolling(window=2).std()*np.sqrt(1)
AD1['Vol_1w'] = AD1['AD1'].rolling(window=21).std()*np.sqrt(21)
AD1['Vol_2w'] = AD1['AD1'].rolling(window=63).std()*np.sqrt(63)
AD1['Vol_3w'] = AD1['AD1'].rolling(window=126).std()*np.sqrt(126)
AD1['Vol_4w'] = AD1['AD1'].rolling(window=252).std()*np.sqrt(252)

# Compute Normalized Return using above
AD1['NR_1w'] = AD1['AD1']/AD1['Vol_w']
AD1['NR_2w'] = AD1['Ret_1w']/AD1['Vol_1w']
AD1['NR_3w'] = AD1['Ret_2w']/AD1['Vol_2w']
AD1['NR_4w'] = AD1['Ret_3w']/AD1['Vol_3w']
AD1['NR_5w'] = AD1['Ret_4w']/AD1['Vol_4w']

# MACD Indicators
AD1['AD1-MACD'] = np.where(signal['AD1_Mean'] > 0, 1, -1)

# Drop
AD1 = AD1.dropna("index")
AD1 = AD1.drop(['Ret_1w', 'Ret_2w', 'Ret_3w', 'Ret_4w',
                'Vol_w', 'Vol_1w', "Vol_2w", 'Vol_3w',
                'Vol_4w'], axis=1)

#%% 5. Separate Test Data & Generate Model Sets for Baseline and LSTM Models¶
# Model Set
training = AD1.iloc[:,1:6]
targeting = AD1.iloc[:,6]
X_train, X_test, y_train, y_test = train_test_split(training,targeting, train_size=0.9 ,shuffle=False, stratify=None)
#%% LSTM
# Input arrays should be shaped as (samples or batch, time_steps or look_back, num_features):
X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])

#%% Time Series Split
dev_size = 0.1
n_splits = int((1//dev_size)-1)   # using // for integer division
tscv = TimeSeriesSplit(n_splits=n_splits)

#%% 6. Models
# Standardized Data
steps_b=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
           ('logistic', linear_model.SGDClassifier(loss="log", shuffle=False, early_stopping=False, tol=1e-3, random_state=1))]

#Normalized Data
#steps_b=[('scaler', MinMaxScaler(feature_range=(0, 1), copy=True)),
#         ('logistic', linear_model.SGDClassifier(loss="log", shuffle=False, early_stopping=False, tol=1e-3, random_state=1))]

pipeline_b=Pipeline(steps_b) # Using a pipeline we glue together the Scaler & the Classifier
# This ensure that during cross validation the Scaler is fitted to only the training folds

# Penalties
penalty_b=['l1', 'l2', 'elasticnet']

# Evaluation Metric
scoring_b={'AUC': 'roc_auc', 'accuracy': make_scorer(accuracy_score)} #multiple evaluation metrics
metric_b='accuracy' #scorer is used to find the best parameters for refitting the estimator at the end

#%%
# Batch_input_shape=[1, 1, Z]  -> (batch size, time steps, number of features)
# Data set inputs(trainX)=[X, 1, Z]  -> (samples, time steps, number of features)

# number of samples
num_samples=1
# time_steps
look_back=1

# Evaluation Metric
scoring_lstm='accuracy'

#%% Model specific Parameter
# Number of iterations
iterations_4_b=[50]

# Grid Search
# Regularization
alpha_g_4_b=[0.0011, 0.0012, 0.0013]
l1_ratio_g_4_b=[0, 0.2, 0.4, 0.6, 0.8, 1]

#%% Create hyperparameter options
hyperparameters_g_4_b={'logistic__alpha':alpha_g_4_b,
                       'logistic__l1_ratio':l1_ratio_g_4_b,
                       'logistic__penalty':penalty_b,
                       'logistic__max_iter':iterations_4_b}

#%% Create grid search
search_g_4_b = GridSearchCV(estimator=pipeline_b,
                          param_grid=hyperparameters_g_4_b,
                          cv=tscv,
                          verbose=0,
                          n_jobs=-1,
                          scoring=scoring_b,
                          refit=metric_b,
                          return_train_score=False)
# Setting refit='Accuracy', refits an estimator on the whole dataset with the parameter setting that has the best cross-validated mean Accuracy score.
# For multiple metric evaluation, this needs to be a string denoting the scorer is used to find the best parameters for refitting the estimator at the end
# If return_train_score=True training results of CV will be saved as well

#%% Fit grid search
tuned_model_4_b=search_g_4_b.fit(X_train, y_train)
#search_g_4_b.cv_results_


#%% Random Search

# Create regularization hyperparameter distribution using uniform distribution
#alpha_r_4_b=uniform(loc=0.00006, scale=0.002) #loc=0.00006, scale=0.002
#l1_ratio_r_4_b=uniform(loc=0, scale=1)

# Create hyperparameter options
#hyperparameters_r_4_b={'logistic__alpha':alpha_r_4_b, 'logistic__l1_ratio':l1_ratio_r_4_b, 'logistic__penalty':penalty_b,'logistic__max_iter':iterations_4_b}

# Create randomized search
#search_r_4_b=RandomizedSearchCV(pipeline_b, hyperparameters_r_4_b, n_iter=10, random_state=1, cv=tscv, verbose=0, n_jobs=-1, scoring=scoring_b, refit=metric_b, return_train_score=False)
# Setting refit='Accuracy', refits an estimator on the whole dataset with the parameter setting that has the best cross-validated Accuracy score.

# Fit randomized search
#tuned_model_4_b=search_r_4_b.fit(X_train_4, y_train_4)

# View Cost function
print('Loss function:', tuned_model_4_b.best_estimator_.get_params()['logistic__loss'])

# View Accuracy
print(metric_b +' of the best model: ', tuned_model_4_b.best_score_);print("\n")
# best_score_ Mean cross-validated score of the best_estimator

# View best hyperparameters
print("Best hyperparameters:")
print('Number of iterations:', tuned_model_4_b.best_estimator_.get_params()['logistic__max_iter'])
print('Penalty:', tuned_model_4_b.best_estimator_.get_params()['logistic__penalty'])
print('Alpha:', tuned_model_4_b.best_estimator_.get_params()['logistic__alpha'])
print('l1_ratio:', tuned_model_4_b.best_estimator_.get_params()['logistic__l1_ratio'])

# Find the number of nonzero coefficients (selected features)
print("Total number of features:", len(tuned_model_4_b.best_estimator_.steps[1][1].coef_[0][:]))
print("Number of selected features:", np.count_nonzero(tuned_model_4_b.best_estimator_.steps[1][1].coef_[0][:]))

# Gridsearch table
plt.title('Gridsearch')
pvt_4_b=pd.pivot_table(pd.DataFrame(tuned_model_4_b.cv_results_), values='mean_test_accuracy', index='param_logistic__l1_ratio', columns='param_logistic__alpha')
ax_4_b=sns.heatmap(pvt_4_b, cmap="Blues")
plt.show()

#%%
# Make predictions
y_pred_4_b=tuned_model_4_b.predict(X_test)

# create confustion matrix
fig, ax=plt.subplots()
sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_4_b)), annot=True, cmap="Blues" ,fmt='g')
plt.title('Confusion matrix'); plt.ylabel('Actual label'); plt.xlabel('Predicted label')
ax.xaxis.set_ticklabels(['Down', 'Up']); ax.yaxis.set_ticklabels(['Down', 'Up'])

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_4_b))
print("Precision:",metrics.precision_score(y_test, y_pred_4_b))
print("Recall:",metrics.recall_score(y_test, y_pred_4_b))

#%%
y_proba_4_b=tuned_model_4_b.predict_proba(X_test)[:, 1]
fpr, tpr, _=metrics.roc_curve(y_test,  y_proba_4_b)
auc=metrics.roc_auc_score(y_test, y_proba_4_b)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--') # plot no skill
plt.title('ROC-Curve')
plt.show()

#%% LSTM
start=time.time()
# number of epochs
epochs=1
# number of units
LSTM_units_4_lstm = 100
# number of samples
num_samples = 1
# time_steps
look_back=1
# numer of features
num_features_4_lstm=X_train.shape[1]
# Regularization
dropout_rate=0.3
recurrent_dropout=0.4
# print
verbose=0

#hyperparameter
batch_size=[1]

# hyperparameter
hyperparameter_4_lstm={'batch_size':batch_size}


#%% create Classifier
clf_4_lstm=KerasClassifier(build_fn=create_shallow_LSTM,
                           epochs=epochs,
                           LSTM_units=LSTM_units_4_lstm,
                           num_samples=num_samples,
                           look_back=look_back,
                           num_features=num_features_4_lstm,
                           dropout_rate=dropout_rate,
                           recurrent_dropout=recurrent_dropout,
                           verbose=verbose)
#%% Gridsearch
search_lstm=GridSearchCV(estimator=clf_4_lstm,
                           param_grid=hyperparameter_4_lstm,
                           n_jobs=-1,
                           cv=tscv,
                           scoring=scoring_lstm, # accuracy
                           refit=True,
                           return_train_score=False,
                         )

#%% Fit model
tuned_model_4_lstm = search_lstm.fit(X_train_lstm, y_train, shuffle=False, callbacks=[reset])


#%%
print("\n")

# View Accuracy
print(scoring_lstm +' of the best model: ', tuned_model_4_lstm.best_score_)
# best_score_ Mean cross-validated score of the best_estimator

print("\n")

# View best hyperparameters
print("Best hyperparameters:")
print('epochs:', tuned_model_4_lstm.best_estimator_.get_params()['epochs'])
print('batch_size:', tuned_model_4_lstm.best_estimator_.get_params()['batch_size'])
print('dropout_rate:', tuned_model_4_lstm.best_estimator_.get_params()['dropout_rate'])
print('recurrent_dropout:', tuned_model_4_lstm.best_estimator_.get_params()['recurrent_dropout'])

end=time.time()
print("\n")
print("Running Time:", end - start)


#%%
model = create_shallow_LSTM(epochs=10, LSTM_units=1,num_samples=1,look_back=1,
                            num_features=5,dropout_rate=0.3, )

model.fit(X_train_lstm, y_train, epochs = 30, batch_size = 1)
#%%
y_pred_lstm = model.predict(X_test_lstm, batch_size=1)

#%%
# create confustion matrix
y = np.where(y_pred_lstm > 0, 1, -1)
#%%
fig, ax=plt.subplots()
sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_test, y)), annot=True, cmap="Blues" ,fmt='g')
plt.title('Confusion matrix'); plt.ylabel('Actual label'); plt.xlabel('Predicted label')
ax.xaxis.set_ticklabels(['Down', 'Up']); ax.yaxis.set_ticklabels(['Down', 'Up'])
plt.show()

print("Accuracy:",metrics.accuracy_score(y_test, y))
print("Precision:",metrics.precision_score(y_test, y))
print("Recall:",metrics.recall_score(y_test, y))
#%%
df = pd.DataFrame()
day_vol = AD1.iloc[7479:,0].ewm(ignore_na=False,
                          adjust=True,
                          span=60,
                          min_periods=0).std(bias=False)
df['return'] = AD1.iloc[7479:,0] * y_pred_lstm * 0.15 / day_vol

#%%
import empyrical
print("Annualized Sharpe Ratio = ", empyrical.sharpe_ratio(df['return'], period='daily'))
print("Annualized Mean Returns = ", empyrical.annual_return(df['return'], period='daily'))
print("Annualized Standard Deviations = ", empyrical.annual_volatility(df['return'], period='daily'))
print("Max Drawdown (MDD) = ", empyrical.max_drawdown(df['return']))
print("Sortino ratio = ", empyrical.sortino_ratio(df['return'], period='daily'))
print("Calmar ratio = ", empyrical.calmar_ratio(df['return'], period='daily'))

#%%
a = pd.DataFrame()
a = pd.concat([a, TSMOM.ind_return['port_avg']], axis=1)
a = a.tail(831)
b = empyrical.cum_returns(a)
c = empyrical.cum_returns(df['return'])
plt.plot(b, color = 'red', label = 'R')
plt.plot(c, color = 'blue', label = 'R')
plt.title('Cumulative return in daily basis')
plt.xlabel('Time')
plt.ylabel('Cumulative return')
plt.legend()
plt.show()