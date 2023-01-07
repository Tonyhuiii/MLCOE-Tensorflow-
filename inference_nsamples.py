import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams

from imputers.SSSDS4Imputer import SSSDS4Imputer
import pickle
import time
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


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * tf.reduce_sum(tf.math.abs((forecast - target) * eval_points * (tf.cast(target <= forecast, tf.float32) * 1.0 - q)))


def calc_denominator(target, eval_points):
    return tf.reduce_sum(tf.math.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(tf.constant(np.quantile(forecast[j: j + 1], quantiles[i], axis=1), tf.float32))
        q_pred = tf.concat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
        
    return CRPS.numpy() / len(quantiles)

def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             use_model,
             masking,
             missing_k,
             only_generate_missing,
             nsamples=10):
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])
    local_path1 = 'inference_nsamples'
    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path1)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    # for key in diffusion_hyperparams:
    #     if key != "T":
    #         diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

            
    # predefine model
    if use_model == 0:
        net = DiffWaveImputer(**model_config).cuda()
    elif use_model == 1:
        net = SSSDSAImputer(**model_config).cuda()
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config)
    else:
        print('Model chosen not available.')
    # print_size(net)

    net.compile(optimizer=keras.optimizers.Adam(2e-4), loss='mse')

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}/{}'.format(ckpt_iter,ckpt_iter))
    try:
        print(model_path)
        net.load_weights(model_path).expect_partial()
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

        
        
    ### Custom data loading and reshaping ###
    testing_data = np.load(trainset_config['test_data_path'])
    # testing_data = np.split(testing_data, 4, 0)         ### mujoco (4, 500, 100, 14)
    testing_data = testing_data[:-3].reshape(-1,12,4,250) ### ptbxl (2200, 12, 4, 250)
    testing_data = testing_data.transpose(0,2,3,1)        ### ptbxl (2200, 4, 250, 12)
    testing_data = testing_data.reshape((-1,250,12))      ### ptbxl (8800, 250, 12)
    testing_data = np.split(testing_data, 16, 0)          ### ptbxl (16, 550, 250, 12)    
    # testing_data = testing_data[:100]
    # testing_data = np.split(testing_data, 2, 0)           
    testing_data = np.array(testing_data)
    testing_data = tf.constant(testing_data, dtype=tf.float32)
    print('Data loaded', testing_data.shape)

    mse_total = 0
    mae_total = 0
    evalpoints_total = 0
    all_target = []
    all_evalpoint = []
    all_generated_samples = []
    for i, batch in enumerate(testing_data):

        if masking == 'mnr':
            mask_T = get_mask_mnr(batch[0], missing_k)

        elif masking == 'bm':
            mask_T = get_mask_bm(batch[0], missing_k)

        elif masking == 'rm':
            mask_T = get_mask_rm(batch[0], missing_k)


        mask = tf.transpose(mask_T, perm=[1,0])
        mask = tf.expand_dims(mask,0)
        mask = tf.tile(mask,[batch.shape[0], 1, 1])            
        loss_mask = 1-mask            
        batch = tf.transpose(batch, perm=[0, 2, 1])
    
        B, K, L = batch.shape
        imputed_samples = np.zeros([B, nsamples, K, L])
        start = time.time()
        for j in range(nsamples):
            generated_audio = sampling(net, batch.shape,
                                    diffusion_hyperparams,
                                    cond=batch,
                                    mask=mask,
                                    only_generate_missing=only_generate_missing)

            generated_audio = generated_audio.numpy()                        
            imputed_samples[:, j]= generated_audio  # (B,nsample,K,L)

        end=time.time()

        print('generated {} nsamples for {} utterances of random_digit at iteration {} in {} mins'.format(nsamples,
                                                                                                         batch.shape[0],
                                                                                                         ckpt_iter,
                                                                                                         (end-start)/60))
        samples = tf.constant(imputed_samples)
        samples = tf.transpose(samples, [0, 1, 3, 2]) # (B,nsample,L,K)
        c_target = tf.transpose(batch, [0, 2, 1])  # (B,L,K)
        eval_points = tf.transpose(loss_mask, [0, 2, 1])

        samples_median = tf.constant(np.median(samples.numpy(), 1), tf.float32)
        all_target.append(c_target)
        all_evalpoint.append(eval_points)
        all_generated_samples.append(samples)

        mse_current = ((samples_median - c_target) * eval_points) ** 2
        mae_current = tf.math.abs((samples_median - c_target) * eval_points)

        mse_total += tf.reduce_sum(mse_current)
        mae_total += tf.reduce_sum(mae_current)
        evalpoints_total +=  tf.reduce_sum(eval_points) 

    with open(f"{output_directory}/generated_outputs_nsample"+str(nsamples)+".pk","wb") as f:
        all_target = tf.concat(all_target, 0)
        all_evalpoint = tf.concat(all_evalpoint, 0)
        all_generated_samples = tf.concat(all_generated_samples, 0)

        pickle.dump(
            [
                all_generated_samples,
                all_target,
                all_evalpoint,
            ],
            f,
        )

    CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, 0, 1)

    with open(f"{output_directory}/result_nsample" + str(nsamples) + ".pk", "wb") as f:
        pickle.dump(
            [
                np.sqrt(mse_total.numpy()/ evalpoints_total.numpy()),
                mae_total.numpy() / evalpoints_total.numpy(), 
                CRPS
            ], 
            f)

    print("MAE:", mae_total.numpy() / evalpoints_total.numpy())
    print("RMSE:", np.sqrt(mse_total.numpy()/ evalpoints_total.numpy()))    
    print("CRPS:", CRPS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=500,
                        help='Number of utterances to be generated')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']

    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=args.num_samples,
             use_model=train_config["use_model"],
             data_path=trainset_config["test_data_path"],
             masking=train_config["masking"],
             missing_k=train_config["missing_k"],
             only_generate_missing=train_config["only_generate_missing"])
