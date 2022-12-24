from imputers.CSDI_stock import CSDIImputer
import numpy as np 
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

train_data = np.load('stock_data/Hang_Seng/Hang_Seng_train_data.npy')  ### Hang Seng (1014, 239, 6)
train_data = np.nan_to_num(train_data)
train_data = tf.constant(train_data,dtype=tf.float32)
test_data = np.load('stock_data/Hang_Seng/Hang_Seng_test_data.npy')   ### Hang Seng (253, 239, 6)
test_data = np.nan_to_num(test_data)
test_data = tf.constant(test_data, dtype=tf.float32)

train_mask = np.load('stock_data/Hang_Seng/Hang_Seng_train_mask.npy')  ### Hang Seng (1014, 239)
# train_mask = np.tile(train_mask[:, :, None], (1, 1, 6))
test_mask = np.load('stock_data/Hang_Seng/Hang_Seng_test_mask.npy')   ### Hang Seng (253, 239)
# test_mask = np.tile(train_mask[:, :, None], (1, 1, 6))

data = tf.concat((train_data, test_data, test_data), 0)  ### 
mask = np.concatenate((train_mask, test_mask, test_mask), 0)  ### 
print(data.shape, mask.shape)

batch= 32
imputer = CSDIImputer()
# target_strategy = "random missing with length"
target_strategy = "blackout missing with length"
imputer.train(data, mask, epochs = 1, train_split = 0.6672, valid_split = 0.1664, batch_size=batch, target_strategy = target_strategy, path_save='csdi_stock_test/') # for training

##### inference and impute
# test_data = np.load('stock_data/Hang_Seng/Hang_Seng_test_data.npy')   ### Hang Seng (253, 239, 6)
# test_data = np.nan_to_num(test_data)
# test_data = tf.constant(test_data, dtype=tf.float32)
# test_mask = np.load('stock_data/Hang_Seng/Hang_Seng_test_mask.npy')   ### Hang Seng (253, 239)

# imputer = CSDIImputer()
# imputer.load_weights('csdi_stock_test/1/1', 'csdi_stock_test/config_csdi_training.json') # after training

# imputations = imputer.impute(test_data, test_mask, 10) # sampling