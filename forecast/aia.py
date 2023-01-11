import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

##### read
f=open('forecast/Dow_Jones_10year_stock.txt', 'r')
tickers_more_than_10years = f.readline()
f.close()
data = yf.download(tickers_more_than_10years, period = "max", group_by = 'ticker', end='2023-01-09')  ### 2023-01-06
time = data.index
time_list = time.to_list()

aia = yf.download('1299.HK', period = "max", group_by = 'ticker', end='2023-01-09')

# time1 = aia.index
# i = time1[99]
# # for i in time1:
# a = time_list.index(i-relativedelta(days=1))
# b = time[a-99:a+1]
# c = data.loc[b]
# tickers= tickers_more_than_10years.split(' ')
# for ticker in tickers:        ### all vaild
#     idx = c[ticker].index[c[ticker].isnull().all(1)]
#     print(ticker, idx)


time1 = aia.index
aia1 = aia.to_numpy()
d1={} ### in the lsit
d2={} ### out of the list
for i in range(99,len(time1)):
    day=time1[i].strftime('%Y/%m/%d')
    year1=day[:4]
    month1=day[5:7]
    day1=day[-2:]
    week_no = datetime.date(int(year1),int(month1),int(day1)).isoweekday() 
    if week_no==1:
        previous_day = time1[i]-relativedelta(days=3)      ### Last Friday
    elif week_no==6 or week_no==7:                  
        print("wrong")
    else:
        previous_day = time1[i]-relativedelta(days=1)      ### previous night
        
    if previous_day in time_list:
        a = time_list.index(previous_day)  
        d1[i]=a
    else:
        d2[i]=previous_day

samples=len(d1)
stock =np.zeros((samples,30,100))
tickers= tickers_more_than_10years.split(' ')
j=0
for key in d1:
    a = d1[key]
    b = time[a-99:a+1]             ### 100 day data
    c = data.loc[b]
    feature=np.zeros((30,100))
    index=0
    for ticker in tickers:
        data100 = c[ticker].to_numpy()
        high_low = data100[:,1]-data100[:,2]     ### INTC 4918, 5195 ZERO
        feature[index] = high_low
        index=index+1

    feature[index] = aia1[key-99:key+1,1]-aia1[key-99:key+1,2] 
    stock[j]=feature
    # print(j)
    j=j+1

np.save('forecast/raw/aia.npy', stock)  #### (2822, 30, 100)

# #####  normalize
# stock_norm = stock.copy()
# for i in range(len(tickers)):
#     ticker=tickers[i]
#     high_price = data[ticker].to_numpy()[:,1]
#     x_max = np.nanmax(high_price)
#     stock_norm[:,i,:]=stock_norm[:,i,:]/x_max

# aia_max= np.max(aia1[:,1])       ### 109.3
# stock_norm[:,-1,:]=stock_norm[:,-1,:]/aia_max

# np.save('forecast/aia_normalize.npy', stock_norm)  #### (2822, 30, 100)


stock_norm = np.load('forecast/raw/aia.npy')
# stock_norm = np.load('forecast/aia_normalize.npy')

### split train 0.8 /test 0.2
np.random.seed(0)
per = np.random.permutation(np.arange(stock_norm.shape[0]))
train_index = per[:int(stock_norm.shape[0]*0.8)+1]
test_index = per[int(stock_norm.shape[0]*0.8)+1:]
x_train = stock_norm[train_index,:,:]  ### (2258, 30, 100)
x_test = stock_norm[test_index,:,:]    ### (564, 30, 100)
np.save('forecast/raw/aia_train.npy', x_train.transpose(0,2,1))
np.save('forecast/raw/aia_test.npy', x_test.transpose(0,2,1))
# np.save('forecast/aia_train.npy', x_train.transpose(0,2,1))
# np.save('forecast/aia_test.npy', x_test.transpose(0,2,1))
