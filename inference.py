import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams

from imputers.SSSDS4Imputer import SSSDS4Imputer

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statistics import mean
import math
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

def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             use_model,
             masking,
             missing_k,
             only_generate_missing):
    
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
    local_path1 = 'inference'
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
    testing_data = np.split(testing_data, 4, 0)  ### mujoco (4, 500, 100, 14)
    # testing_data = testing_data[:-3].reshape(-1,12,250,4) ### ptbxl (2200, 12, 250, 4)
    # testing_data = testing_data.reshape((-1,250,12))   ### ptbxl (8800, 12, 250)
    # testing_data = np.split(testing_data, 8, 0)  ### ptbxl (8, 1100, 12, 250)
    testing_data = np.array(testing_data)
    testing_data = tf.constant(testing_data, dtype=tf.float32)
    print('Data loaded', testing_data.shape)

    all_mse = []
    all_mae = []
    
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
            
        batch = tf.transpose(batch, perm=[0, 2, 1])
        
        start = time.time()

        sample_length = batch.shape[2]
        sample_channels = batch.shape[1]
        generated_audio = sampling(net, batch.shape,
                                    # (num_samples, sample_channels, sample_length),
                                   diffusion_hyperparams,
                                   cond=batch,
                                   mask=mask,
                                   only_generate_missing=only_generate_missing)

        end=time.time()

        print('generated {} utterances of random_digit at iteration {} in {} mins'.format(batch.shape[0],
                                                                                            # num_samples,
                                                                                             ckpt_iter,
                                                                                             (end-start)/60))

        generated_audio = generated_audio.numpy()
        batch = batch.numpy()
        mask = mask.numpy() 
        
        outfile = f'imputation{i}.npy'
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, generated_audio)

        outfile = f'original{i}.npy'
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, batch)

        outfile = f'mask{i}.npy'
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, mask)

        print('saved generated samples at iteration %s' % ckpt_iter)

        loss_mask = (1-mask)>0
        mse = mean_squared_error(generated_audio[loss_mask], batch[loss_mask])
        mae = mean_absolute_error(generated_audio[loss_mask], batch[loss_mask])

        all_mse.append(mse)
        all_mae.append(mae)   

    print('Total MAE:', mean(all_mae))
    print('Total MSE:', mean(all_mse))
    print('RMSE:', math.sqrt(mean(all_mse)))

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
