import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.util import find_max_epoch, print_size, training_stock_loss, calc_diffusion_hyperparams
from tqdm import tqdm
from imputers.SSSDS4Imputer_stock import SSSDS4Imputer
import random
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

def my_loss(y_true, y_pred):
    residual = (y_true - y_pred)
    num_eval = tf.cast(tf.math.count_nonzero(y_true), tf.float32)
    # assert num_eval== y_true.shape[0]*train_config['missing_k']*6
    # print(num_eval)
    loss = tf.reduce_sum(residual ** 2)/ num_eval 
    return loss

def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          k_segments_or_k_misssing):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   "random missing with length" or "blackout missing with length"
    k_segments_or_k_misssing(int):  k_segments for "blackout missing with length", e.g., 5; k_misssing for "random missing with length", e.g., 50..
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
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

    # define optimizer
    # net.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mse')
    net.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=my_loss)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            net.load_weights(model_path).expect_partial()
            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')


    ### Custom data loading and reshaping ###
    training_data = np.load(trainset_config['train_data_path'])
    training_data = np.split(training_data, 73, 0)  ### Hang Seng (2190, 103, 6) -> (73, 30, 103, 6)
    # training_data = np.split(training_data[:-3], 52, 0)  ### Dow Jones (2080, 137, 6) -> (52, 40, 137, 6)
    # training_data = np.split(training_data[:-2], 55, 0)  ### Euro (2475, 94, 6) -> (55, 45, 94, 6)
    training_data = np.array(training_data)
    training_data = np.nan_to_num(training_data)
    training_data = tf.constant(training_data,dtype=tf.float32)
    print('Data loaded', training_data.shape)

    training_mask = np.load(trainset_config['train_mask_path'])
    training_mask = np.split(training_mask, 73, 0)   ### Hang Seng
    # training_mask = np.split(training_mask[:-3], 52, 0)  ### Dow Jones 
    # training_mask = np.split(training_mask[:-2], 55, 0)  ### Euro
    training_mask = np.array(training_mask) 
    print('Mask loaded',  training_mask.shape)   
    
    # training
    pbar = tqdm(total=n_iters + 1)
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for index, batch in enumerate(training_data):

            if masking == 'random missing with length':
                #### generate random mask for each batch
                mask_batch = training_mask[index] ### (B, L)
                masks=[]
                for j in range(len(mask_batch)):
                    mask = mask_batch[j] ###(L)
                    valid_mask=np.where(mask==1)[0]
                    # nan_mask=np.where(mask==0)[0]
                    # holiday_mask=np.where(mask==-1)[0]
                    # length = len(valid_mask)
                    perm = np.random.permutation(valid_mask)
                    idx = perm[0:k_segments_or_k_misssing]
                    mask[idx] = 2.0  #### label missing value as 2
                    masks.append(mask)
                masks = np.array(masks) ### (B, L)

            elif masking == 'blackout missing with length':
                observed_mask = training_mask[index]### (B, L)
                masks = []
                for i in range(len(observed_mask)):
                    mask = observed_mask[i] ###(L)
                    valid_mask = np.where(mask==1)[0]
                    list_of_segments_index = np.array_split(valid_mask, k_segments_or_k_misssing) ### only valid mask will be selected
                    s_nan = random.choice(list_of_segments_index)
                    mask[s_nan] = 2 #### label missing value as 2 
                    masks.append(mask)                 
                masks = np.array(masks) ### (B, L)

            masks = np.tile(masks[:, :, None], (1, 1, 6)) ###(B, L, C)
            train_mask = masks.copy()
            train_mask[np.where(masks!=1)]=0
            train_mask = tf.constant(train_mask, dtype=tf.float32)
            loss_mask = masks.copy()
            loss_mask[np.where(masks==2)]=1
            loss_mask[np.where(masks!=2)]=0
            loss_mask = tf.constant(loss_mask, dtype=tf.float32)
            train_mask = tf.transpose(train_mask, perm=[0,2,1]) ###(B, C, L)
            loss_mask = tf.transpose(loss_mask, perm=[0,2,1]) ###(B, C, L)

            batch = tf.transpose(batch, perm=[0, 2, 1])

            assert batch.shape == train_mask.shape == loss_mask.shape

            # back-propagation
            X = batch, batch, train_mask, loss_mask
            
            loss = training_stock_loss(net, X, diffusion_hyperparams,
                                 only_generate_missing=only_generate_missing)

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss))

            if n_iter > 0 and n_iter % iters_per_logging == 0:
                pbar.update(iters_per_logging)

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}/{}'.format(n_iter, n_iter)
                net.save_weights(os.path.join(output_directory, checkpoint_name))
                # torch.save({'model_state_dict': net.state_dict(),
                #             'optimizer_state_dict': optimizer.state_dict()},
                #            os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1
            
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSDS4.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

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

    train(**train_config)
