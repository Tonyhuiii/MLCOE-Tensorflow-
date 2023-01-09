import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
from tqdm import tqdm
from imputers.SSSDS4Imputer import SSSDS4Imputer
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


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k):
    
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
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
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
    net.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mse')

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
    # training_data = np.split(training_data, 109, 0)           ### tecent (109, 32, 100, 28)
    training_data = np.split(training_data[:-1], 61, 0)     ### aia    (61, 37, 100, 30)
    training_data = np.array(training_data)
    training_data = tf.constant(training_data,dtype=tf.float32)
    training_data = tf.math.abs(training_data)
    print('Data loaded', training_data.shape)
    
    # training
    pbar = tqdm(total=n_iters + 1)
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for batch in training_data:
            if masking == 'tf':
                transposed_mask = np.ones(batch[0].shape)
                transposed_mask[-1][-1] = 0
                transposed_mask = tf.constant(transposed_mask, dtype=tf.float32)

            mask = tf.transpose(transposed_mask, perm=[1,0])
            mask = tf.expand_dims(mask,0)
            mask = tf.tile(mask,[batch.shape[0], 1, 1])

            loss_mask = (1-mask)>0
            batch = tf.transpose(batch, perm=[0, 2, 1])

            assert batch.shape == mask.shape == loss_mask.shape

            # back-propagation
            X = batch, batch, mask, loss_mask
            
            loss = training_loss(net, X, diffusion_hyperparams,
                                 only_generate_missing=only_generate_missing)

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss))

            if n_iter > 0 and n_iter % iters_per_logging == 0:
                pbar.update(iters_per_logging)

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}/{}'.format(n_iter, n_iter)
                net.save_weights(os.path.join(output_directory, checkpoint_name))
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
