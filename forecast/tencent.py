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
# gs=data['GS'].to_numpy()
tencent = yf.download('0700.HK', period = "max", group_by = 'ticker', end='2023-01-09')  ### 2023-01-06
# aia = yf.download('1299.HK', period = "max", group_by = 'ticker')
# hsbc = yf.download('0005.HK', period = "max", group_by = 'ticker')

time1 = tencent.index
i = time1[99]
# for i in time1:
a = time_list.index(i-relativedelta(days=1))
b = time[a-99:a+1]
c = data.loc[b]
tickers= tickers_more_than_10years.split(' ')
for ticker in tickers:        ### V and CRM have nan data
    idx = c[ticker].index[c[ticker].isnull().all(1)]
    print(ticker, idx)

ticker27=[]
for ticker in tickers:
    if ticker!='V' and ticker!='CRM':
        ticker27.append(ticker)

ticker27_ = ' '.join(ticker27)

##### save 
f=open('forecast/ticker for tencent.txt', 'w')
f.write(ticker27_)
f.close()

##### read
f=open('forecast/ticker for tencent.txt', 'r')
ticker27_ = f.readline()
f.close()
data = yf.download(ticker27_, period = "max", group_by = 'ticker', end='2023-01-09')  ### 2023-01-06
tencent = yf.download('0700.HK', period = "max", group_by = 'ticker', end='2023-01-09')  ### 2023-01-06
time = data.index
time_list = time.to_list()
time1 = tencent.index
tencent1 = tencent.to_numpy()
# i = time1[3]
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
stock =np.zeros((samples,28,100))
ticker27=ticker27_ .split(' ')
j=0
for key in d1:
    a = d1[key]
    b = time[a-99:a+1]             ### 100 day data
    c = data.loc[b]
    feature=np.zeros((28,100))
    index=0
    for ticker in ticker27:
        data100 = c[ticker].to_numpy()
        high_low = data100[:,1]-data100[:,2]
        feature[index] = high_low
        index=index+1

    feature[index] = tencent1[key-99:key+1,1]-tencent1[key-99:key+1,2] ### 1377, 1387 negative
    stock[j]=feature
    print(j)
    j=j+1

np.save('forecast/raw/tencent.npy', stock)  #### (4359, 28, 100)


# #####  normalize
# stock_norm = stock.copy()
# for i in range(len(ticker27)):
#     ticker=ticker27[i]
#     high_price = data[ticker].to_numpy()[:,1]
#     x_max = np.nanmax(high_price)
#     stock_norm[:,i,:]=stock_norm[:,i,:]/x_max

# tencent_max= np.max(tencent1[:,1])    ### 775.5
# stock_norm[:,-1,:]=stock_norm[:,-1,:]/tencent_max

# np.save('forecast/tencent_normalize.npy', stock_norm)  #### (4359, 28, 100)

#### load normalized data
stock_norm = np.load('forecast/raw/tencent.npy')
# stock_norm = np.load('forecast/tencent_normalize.npy')

### split train 0.8 /test 0.2
np.random.seed(0)
per = np.random.permutation(np.arange(stock_norm.shape[0]))
train_index = per[:int(stock_norm.shape[0]*0.8)+1]
test_index = per[int(stock_norm.shape[0]*0.8)+1:]
x_train = stock_norm[train_index,:,:]  ### (3488, 28, 100)
x_test = stock_norm[test_index,:,:]    ### (871, 28, 100)
np.save('forecast/raw/tencent_train.npy', x_train.transpose(0,2,1))
np.save('forecast/raw/tencent_test.npy', x_test.transpose(0,2,1))
# np.save('forecast/tecent_train.npy', x_train.transpose(0,2,1))
# np.save('forecast/tecent_test.npy', x_test.transpose(0,2,1))
