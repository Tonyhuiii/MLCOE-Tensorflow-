import yfinance as yf
import datetime
import pandas as pd
import numpy as np

# read_file = pd.read_excel ("sotck_data/Hang Seng.xlsx")
# read_file['Ticker']=read_file['Ticker'].apply(lambda x: str(x).zfill(4)+'.HK')
# read_file.to_csv ("sotck_data/Hang Seng.csv", 
#                   index = None,
#                   header=True)
    
#### check stock with year
file = pd.read_csv ("stock_data/EuroStoxx/EuroStoxx_50.csv")
ticker_list=[]
for ticker in file['Ticker']:
    ticker_list.append(ticker)
tickers=' '.join(ticker_list)
data = yf.download(tickers, period = "max", group_by = 'ticker')

stock_days = {}
for ticker in ticker_list:
    day = 0
    for i in data[ticker]['Open'].isnull():
        if i==False:
            day=day+1
    stock_days[ticker]=day

stock_more_than_10year = []
for i in stock_days:
    if stock_days[i] >= 250*10:
        stock_more_than_10year.append(i)
        # print(i, stock_days[i])

stock_no = len(stock_more_than_10year)
tickers_more_than_10years = ' '.join(stock_more_than_10year)

##### save 
f=open('stock_data/EuroStoxx/EuroStoxx_10year_stock.txt', 'w')
f.write(tickers_more_than_10years)
f.close()

##### read
f=open('stock_data/EuroStoxx/EuroStoxx_10year_stock.txt', 'r')
tickers_more_than_10years = f.readline()
f.close()
data = yf.download(tickers_more_than_10years, period = "max", group_by = 'ticker', end="2022-12-1")

### iterate all weekdays
start_date = data['ABI.BR'].index[0].strftime('%Y/%m/%d') ###'1989/01/16'
start_year = int(start_date[:4])
start_month = int(start_date[5:7])
start_day = int(start_date[-2:])
# print(start_year, start_month, start_day)
end_date = data['ABI.BR'].index[-1].strftime('%Y/%m/%d')  ###'2022/11/29'
end_year = int(end_date[:4])
end_month = int(end_date[5:7])
end_day = int(end_date[-2:])
start=datetime.date(start_year, start_month, start_day)
end=datetime.date(end_year, end_month, end_day)
weekdays=[]
for i in range((end-start).days+1):
    day= str(start+ datetime.timedelta(days=i))
    year1=day[:4]
    month1=day[5:7]
    day1=day[-2:]
    week_no = datetime.date(int(year1),int(month1),int(day1)).isoweekday() 
    if week_no !=6 and week_no !=7:
        weekdays.append(day)
        # print(week_no, day)


tickers= tickers_more_than_10years.split(' ')
datas = np.zeros((47, len(weekdays), 6))
labels = np.zeros((47,len(weekdays)))
j=0
### mask/label;  -1:holiday; 0:nan; 1:valid; 
for ticker in tickers:
    mask = {}
    for i in weekdays:
        mask[i] = -1

    count_nan=0
    count_valid=0
    for day in data.index.strftime('%Y/%m/%d'):
        year1=day[:4]
        month1=day[5:7]
        day1=day[-2:]
        day2=str(datetime.date(int(year1),int(month1),int(day1)))
        if not pd.isnull(data[ticker]['Open'][day]):
            count_valid=count_valid+1        
            mask[day2]=1
        else:
            count_nan=count_nan+1
            mask[day2]=0
            # print(day)

    count_holiday= len(weekdays) - len(data.index)
    partial_data = data[ticker].to_numpy()
    total_data = np.zeros((len(mask), 6))
    a = np.empty(6)
    a[:] = np.nan
    b = 0
    for i in range (len(total_data)):
        if mask[weekdays[i]]==-1:
            total_data[i]= a
        else:
            total_data[i]=partial_data[b]
            b = b + 1

    label = np.empty(len(mask))
    label[:] = -1
    for i in range(len(label)):
        label[i] = mask[weekdays[i]]

    print(j)
    datas[j]=total_data
    labels[j]=label
    j=j+1
    # label.shape

### save raw data
np.save('stock_data/EuroStoxx/EuroStoxx_47_data.npy', datas)
np.save('stock_data/EuroStoxx/EuroStoxx_47_label.npy', labels)

#### load raw data
x = np.load('stock_data/EuroStoxx/EuroStoxx_47_data.npy')
x_max = np.nanmax(x,axis=1)
x_min = np.nanmin(x,axis=1)
x_max= np.tile(x_max[:,None,:], (1, 8837, 1))
x_min= np.tile(x_min[:,None,:], (1, 8837, 1))

### normalize [0,1]
x_normalize = (x - x_min)/(x_max-x_min)
np.save('stock_data/EuroStoxx/EuroStoxx_47_data_normalize.npy', x_normalize)

#### load normalized data
x_normalize = np.load('stock_data/EuroStoxx/EuroStoxx_47_data_normalize.npy')
y = np.load('stock_data/EuroStoxx/EuroStoxx_47_label.npy')
# masks = np.tile(y[:,:, None], (1, 1, 6))

#### reshape [batch, length, channel]
x = x_normalize[:, :-1].reshape(-1,94,6) ### (4418, 94, 6)
masks = y[:, :-1].reshape(-1,94)  ### (4418, 94)

#### filter the batch with many nan data
count=[]
for index, batch in enumerate(masks):
    nan_mask = np.where(batch==0)[0]
    if len(nan_mask) < 9:
        count.append(index)

x_filter=x[count]   ### (3096, 94, 6)
masks_filter=masks[count]   ### (3096, 94)

### split train 0.8 /test 0.2
x=x_filter
masks=masks_filter
np.random.seed(0)
per = np.random.permutation(np.arange(x.shape[0]))
train_index = per[:int(x.shape[0]*0.8)+1]
test_index = per[int(x.shape[0]*0.8)+1:]
x_train = x[train_index,:,:]  ### (2477, 94, 6)
y_train = masks[train_index,:]
np.save('stock_data/EuroStoxx/train_data_94.npy', x_train)
np.save('stock_data/EuroStoxx/train_mask_94.npy', y_train)
x_test = x[test_index,:,:]  ### (619, 94, 6)
y_test = masks[test_index,:]
np.save('stock_data/EuroStoxx/test_data_94.npy', x_test)
np.save('stock_data/EuroStoxx/test_mask_94.npy', y_test)



# #### fiter the nan data before the stock has been on the market
# count=[]
# for index, batch in enumerate(masks):
#     nan_mask = np.where(batch==0)[0]
#     if len(nan_mask)>40:
#         count.append(index)


# x_filter = np.zeros((x.shape[0]-len(count), 94, 6))
# masks_filter = np.zeros((x.shape[0]-len(count), 94))

# k=0
# for index, batch in enumerate(x):
#     if index not in count:
#         x_filter[k] = x[index]
#         masks_filter[k] = masks[index]
#         k=k+1
