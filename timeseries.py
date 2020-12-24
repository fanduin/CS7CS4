#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math, sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True


# In[2]:


# df = pd.read_csv('bike_data/dublinbikes_20190401_20190701.csv', usecols = [1,6], parse_dates=[1])
df = pd.read_csv('dublinbikes_20190401_20190701.csv', usecols = [0,1,6], parse_dates=[1])

print(df.head())


# In[3]:


start = pd.to_datetime('2019-04-01',format='%Y-%m-%d')
end = pd.to_datetime('2019-07-01',format='%Y-%m-%d')

print(f'start: {start}')
print(f'end: {end}')


# In[4]:


s_full = pd.array(df.iloc[:,0])
t_full = pd.array(pd.DatetimeIndex(df.iloc[:,1]).astype(np.int64))/1000000000

t_full = np.extract([s_full == 2], t_full)

dt = t_full[1]-t_full[0]
print(f'data sampling is {dt:.2f} secs')


# In[5]:


t_start = pd.DatetimeIndex([start]).astype(np.int64)/1000000000
t_end = pd.DatetimeIndex([end]).astype(np.int64)/1000000000

t = np.extract([(t_full>=t_start[0]) & (t_full<=t_end[0])], t_full)

t = (t-t[0])/60/60/24

y = np.extract([(t_full>=t_start[0]) & (t_full<=t_end[0])], df.iloc[:,2]).astype(np.int64)

print(len(t))
print(len(y))

fig = plt.figure()

plt.scatter(t, y, color='red', marker='.')
plt.xlabel('Day')
plt.ylabel('No. of available bikes')


# In[59]:


def test_preds(q, dd, lag, plot, tw='50', tu='minutes'):
    stride = 1
    XX = y[0:y.size - q - lag * dd:stride]
    
    for i in range(1,lag):
        X = y[i * dd:y.size - q - (lag - i)*dd:stride]
        XX = np.column_stack((XX,X))
        
    print(lag * dd + q)
    
    yy = y[lag * dd + q::stride]
    tt = t[lag * dd + q::stride]
    
    print(yy)
    
    train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)
    
    model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
    print(model.intercept_, model.coef_)
    
    if plot:
        y_pred = model.predict(XX)
        fig = plt.figure()
        
        plt.scatter(t, y, color='black')
        plt.scatter(tt, y_pred, color='blue')
        
        plt.xlabel('Time (days)')
        plt.ylabel('No. of bikes available')
        plt.legend(['Training data', 'Time-series predictions'], loc='upper right')
        plt.title(f'Time series predictions: +-{tw} {tu}')
        
        day = math.floor(24*60*60/dt)
        plt.xlim(((lag * dd + q)/day, (lag * dd + q)/day + 2))
        
        plt.savefig(f'time_series_predictions_{tw}.png')
        
        print(np.sum(np.subtract(yy, y_pred) ** 2) / yy.size)
#         print(accuracy_score(yy, y_pred, normalize=False))


# In[60]:


plot = True
test_preds(q=10, dd=1, lag=3, plot=plot)


# In[61]:


d=math.floor(24*60*60/dt) # number of samples per day
test_preds(q=d,dd=d,lag=3,plot=plot,tw='1',tu='day')


# In[62]:


w=math.floor(7*24*60*60/dt) # number of samples per day
test_preds(q=w,dd=w,lag=3,plot=plot,tw='7',tu='days')

plt.show()

# In[10]:


#putting it together
# q=10
# lag=3; stride=1
# w=math.floor(7*24*60*60/dt) # number of samples per week
# len = y.size-w-lag*w-q
# XX=y[q:q+len:stride]
# for i in range(1,lag):
#     X=y[i*w+q:i*w+q+len:stride]
#     XX=np.column_stack((XX,X))
# d=math.floor(24*60*60/dt) # number of samples per day
# for i in range(0,lag):
#     X=y[i*d+q:i*d+q+len:stride]
#     XX=np.column_stack((XX,X))
# for i in range(0,lag):
#     X=y[i:i+len:stride]
#     XX=np.column_stack((XX,X))
# yy=y[lag*w+w+q:lag*w+w+q+len:stride]
# tt=t[lag*w+w+q:lag*w+w+q+len:stride]

# train, test = train_test_split(np.arange(0,yy.size),test_size=0.2)
# #train = np.arange(0,yy.size)

# model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
# print(model.intercept_, model.coef_)
# if plot:
#     y_pred = model.predict(XX)
#     plt.scatter(t, y, color='black')
#     plt.scatter(tt, y_pred, color='blue')
#     plt.xlabel('time (days)')
#     plt.ylabel("#bikes")
#     plt.legend(["training data","predictions"],loc='upper right')
#     day=math.floor(24*60*60/dt) # number of samples per day
#     plt.xlim((4*7,4*7+4))
#     plt.show()


# In[44]:


# def convert2matrix(data_arr, look_back):
#     X, Y =[], []
#     for i in range(len(data_arr)-look_back):
#         d=i+look_back  
#         X.append(data_arr[i:d,])
#         Y.append(data_arr[d,])
#     return np.array(X), np.array(Y)


# # In[51]:


# t_train, t_test, y_train, y_test = train_test_split(t, y, test_size=0.2)

# t_train = t_train.reshape(-1,1)

# scaler = MinMaxScaler(feature_range=(0, 1))
# t_train = scaler.fit_transform(t_train)

# look_back = 10
# t_train = 


# # In[38]:


# model = Sequential()
# model.add(LSTM(4, input_shape=t_train.shape))
# model.add(Dense(1), input_shap=t_train.shape)
# model.compile(loss='mean_squared_error', optimizer='adam')

# model.fit(t_train, y_train, epochs=20, batch_size=1, verbose=2)

