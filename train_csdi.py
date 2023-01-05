from imputers.CSDI import CSDIImputer
import numpy as np 
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


imputer = CSDIImputer()

train_data = np.load('datasets/train_ptbxl_1000.npy')
train_data = train_data.reshape([-1,12,4,250])
train_data = train_data.transpose(0,2,3,1)
train_data = train_data.reshape([-1,250,12])
train_data = tf.constant(train_data)
print(train_data.shape)
val_data =  np.load('datasets/val_ptbxl_1000.npy')
val_data = val_data.reshape([-1,12,4,250])
val_data = val_data.transpose(0,2,3,1)
val_data = val_data.reshape([-1,250,12])
val_data = tf.constant(val_data)
print(val_data.shape)
test_data = np.load('datasets/test_ptbxl_1000.npy')
test_data = test_data.reshape([-1,12,4,250])
test_data = test_data.transpose(0,2,3,1)
test_data = test_data.reshape([-1,250,12])
test_data = tf.constant(test_data)
print(test_data.shape)

data = tf.concat((train_data, val_data, test_data), 0)
print(data.shape)

masking='rm'
missing_ratio=0.0
batch= 32
imputer.train(data, masking, missing_ratio, epochs = 200, train_split = 0.7987, valid_split = 0.10042, batch_size=batch, path_save='csdi_ptbxl_0103/bm/') # for training


##### inference and impute
# from imputers.CSDI import CSDIImputer
# import numpy as np 
# import tensorflow as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

# # test_data = tf.constant(np.load('datasets/test_ptbxl_1000.npy'))
# # test_data = tf.reshape(test_data, [-1,12,250,4])
# # test_data = tf.reshape(test_data, [-1,250,12]) 
# # test_data = test_data[:100]

# test_data = tf.constant(np.load('results/ptbxl/bm/inference/original0.npy'))
# test_data = test_data[:10]
# test_data = tf.transpose(test_data, [0,2,1])
# # test_data = tf.reshape(test_data, [10,250,12]) 

# imputer = CSDIImputer()
# imputer.load_weights('csdi_bm_test/199/199', 'csdi_bm_test/config_csdi_training.json') # after training
# mask = tf.ones(test_data[0].shape)
# imputations = imputer.impute(test_data, mask, 10) # sampling