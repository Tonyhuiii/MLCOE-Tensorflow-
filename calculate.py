import pickle
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

# res = open('csdi_stock/dow_bm/generated_outputs_nsample3.pk','rb')
# res = open('csdi_stock/hang_seng_bm/generated_outputs_nsample3.pk','rb')
res = open('csdi_stock/euro_bm/generated_outputs_nsample3.pk','rb')
# res = open('csdi_rm_test/generated_outputs_nsample10.pk','rb')
a= pickle.load(res)
all_generated_samples=tf.cast(a[0],tf.float32)
all_target=a[1]
all_evalpoint=a[2]
all_observed_point=a[3]
all_observed_time=a[4]


mse_total = 0
mae_total = 0
evalpoints_total = 0
n_sample = 3
for i in range(n_sample):
    mse_current = ((all_generated_samples[:,i,:,:] - all_target) * all_evalpoint) ** 2
    mae_current = tf.math.abs((all_generated_samples[:,i,:,:] - all_target) * all_evalpoint) 
    # print(mse_current, mae_current)
    mse_total += tf.reduce_sum(mse_current)
    mae_total += tf.reduce_sum(mae_current)
    evalpoints_total += tf.reduce_sum(all_evalpoint)

mae = (mae_total/ evalpoints_total).numpy()
mse = (mse_total / evalpoints_total).numpy()
rmse = np.sqrt(mse_total / evalpoints_total)

print('mae:{}, mse:{}, rmse:{}'.format(mae, mse, rmse))

# def quantile_loss(target, forecast, q: float, eval_points) -> float:

#     return 2 * tf.reduce_sum(tf.math.abs((forecast - target) * eval_points * (tf.cast(target <= forecast, tf.float32) * 1.0 - q)))


# def calc_denominator(target, eval_points):
#     return tf.reduce_sum(tf.math.abs(target * eval_points))


# def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
#     target = target * scaler + mean_scaler
#     forecast = forecast * scaler + mean_scaler

#     quantiles = np.arange(0.05, 1.0, 0.05)
#     denom = calc_denominator(target, eval_points)
#     CRPS = 0
#     for i in range(len(quantiles)):
#         q_pred = []
#         for j in range(len(forecast)):
#             q_pred.append(tf.constant(np.quantile(forecast[j: j + 1], quantiles[i], axis=1), tf.float32))
#         q_pred = tf.concat(q_pred, 0)
#         q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
#         CRPS += q_loss / denom
        
#     return CRPS.numpy() / len(quantiles)

# CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, 0, 1)
# print('CRPS:{}'.format(CRPS))