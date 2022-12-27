from imputers.CSDI_stock import CSDIImputer
import numpy as np 
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


### Hang Seng
train_data = np.load('stock_data/Hang_Seng/train_data_103.npy')  ### Hang Seng (2190, 103, 6)
train_data = np.nan_to_num(train_data)
train_data = tf.constant(train_data,dtype=tf.float32)
test_data = np.load('stock_data/Hang_Seng/test_data_103.npy')   ### Hang Seng (547, 103, 6)
test_data = np.nan_to_num(test_data)
test_data = tf.constant(test_data, dtype=tf.float32)
train_mask = np.load('stock_data/Hang_Seng/train_mask_103.npy')  
test_mask = np.load('stock_data/Hang_Seng/test_mask_103.npy')   
train_split = 0.6669
valid_split = 0.1666

### EuroStoxx
# train_data = np.load('stock_data/EuroStoxx/train_data_94.npy')  ### (2477, 94, 6)
# train_data = np.nan_to_num(train_data)
# train_data = tf.constant(train_data,dtype=tf.float32)
# test_data = np.load('stock_data/EuroStoxx/test_data_94.npy')   ### (619, 94, 6)
# test_data = np.nan_to_num(test_data)
# test_data = tf.constant(test_data, dtype=tf.float32)
# train_mask = np.load('stock_data/EuroStoxx/train_mask_94.npy')  
# test_mask = np.load('stock_data/EuroStoxx/test_mask_94.npy')   
# train_split = 0.6668
# valid_split = 0.1666

### Dow Jones
# train_data = np.load('stock_data/Dow_Jones/train_data.npy')  ### (2083, 137, 6)
# train_data = np.nan_to_num(train_data)
# train_data = tf.constant(train_data,dtype=tf.float32)
# test_data = np.load('stock_data/Dow_Jones/test_data.npy')   ### (521, 137, 6)
# test_data = np.nan_to_num(test_data)
# test_data = tf.constant(test_data, dtype=tf.float32)
# train_mask = np.load('stock_data/Dow_Jones/train_mask.npy')  
# test_mask = np.load('stock_data/Dow_Jones/test_mask.npy')   
# train_split = 0.6666
# valid_split = 0.1667

#### merge the dataset
data = tf.concat((train_data, test_data, test_data), 0)  ### 
mask = np.concatenate((train_mask, test_mask, test_mask), 0)  ### 
print(data.shape, mask.shape)

batch= 32
imputer = CSDIImputer()
target_strategy = "random missing with length"
k_segments_or_k_misssing = 21
# target_strategy = "blackout missing with length"
# k_segments_or_k_misssing = 5
imputer.train(data, mask, k_segments_or_k_misssing, epochs = 200, samples_generate=3, train_split = train_split, valid_split = valid_split, batch_size=batch, target_strategy = target_strategy, path_save='csdi_stock/hang_seng_rm/') # for training

# #### inference and impute
# test_data = np.load('stock_data/Hang_Seng/Hang_Seng_test_data.npy')   ### Hang Seng (253, 239, 6)
# test_data = np.nan_to_num(test_data)
# test_data = tf.constant(test_data, dtype=tf.float32)
# test_mask = np.load('stock_data/Hang_Seng/Hang_Seng_test_mask.npy')   ### Hang Seng (253, 239)

# imputer = CSDIImputer()
# imputer.load_weights('csdi_stock_test/0/0', 'csdi_stock_test/config_csdi_training.json') 
# # Note the input data shape should be changed in line 810-812 in CSDI_stock.py for different dataset.

# imputations = imputer.impute(test_data, test_mask, 10) # sampling