from imputers.CSDI import CSDIImputer
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


imputer = CSDIImputer()

train_data = tf.constant(np.load('datasets/train_ptbxl_1000.npy'))
train_data = tf.reshape(train_data, [-1,12,250,4])
train_data = tf.reshape(train_data, [-1,250,12])
val_data =  tf.constant(np.load('datasets/val_ptbxl_1000.npy'))
val_data = tf.reshape(val_data, [-1,12,250,4])
val_data = tf.reshape(val_data, [-1,250,12])  
test_data = tf.constant(np.load('datasets/test_ptbxl_1000.npy'))
test_data = tf.reshape(test_data, [-1,12,250,4])
test_data = tf.reshape(test_data, [-1,250,12]) 
data = tf.concat((train_data, val_data, test_data), 0)
print(data.shape)

masking='rm'
missing_ratio=0.0
batch= 32
imputer.train(data, masking, missing_ratio, epochs = 200, train_split = 0.7987, valid_split = 0.10042, batch_size=batch, path_save='csdi_bm_test/') # for training


##### inference and impute
# from imputers.CSDI import CSDIImputer
# import numpy as np 
# import tensorflow as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

# test_data = tf.constant(np.load('datasets/test_ptbxl_1000.npy'))
# test_data = tf.reshape(test_data, [-1,12,250,4])
# test_data = tf.reshape(test_data, [-1,250,12]) 
# test_data = test_data[:100]

# imputer = CSDIImputer()
# imputer.load_weights('csdi_rm_test/1/1', 'csdi_rm_test/config_csdi_training.json') # after training
# mask = tf.ones(test_data[0].shape)
# imputations = imputer.impute(test_data, mask, 10) # sampling