import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import tensorflow_models as tfm
from tqdm import tqdm
import pickle
import math
import json


def train(model, config, train_loader, valid_loader=None, valid_epoch_interval=50, path_save=""):

    
    p1 = int(0.75 * config["epochs"]*len(train_loader))
    p2 = int(0.9 * config["epochs"]*len(train_loader))
    step = tf.Variable(0, trainable=False)
    boundaries = [p1-1, p2-1]
    lr_rate=config["lr"]
    values = [lr_rate, lr_rate*0.1, lr_rate*0.01]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    learning_rate = learning_rate_fn(step)
    optimizer = keras.optimizers.Adam(learning_rate, weight_decay=1e-6)
    best_valid_loss = 1e10

    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        with tqdm(train_loader, mininterval=20, maxinterval=20) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                with tf.GradientTape() as tape:
                    loss = model(train_batch, training=True)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                avg_loss += loss.numpy()

                it.set_postfix(ordered_dict={"avg_epoch_loss": avg_loss / batch_no,"epoch": epoch_no + 1},refresh=False)
    
        if (epoch_no + 1) % valid_epoch_interval == 0:
            output_path = path_save+'/{}/{}'.format(epoch_no, epoch_no)
            model.save_weights(output_path)

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            avg_loss_valid = 0
            with tqdm(valid_loader, mininterval=20, maxinterval=20) as it:
                for batch_no, valid_batch in enumerate(it, start=1):
                    loss = model(valid_batch, is_train=0)
                    avg_loss_valid += loss
                    it.set_postfix(ordered_dict={"valid_avg_epoch_loss":avg_loss_valid/batch_no,"epoch":epoch_no},refresh=False)
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print("\n best loss is updated to ",avg_loss_valid/batch_no,"at",epoch_no+1)
        try:
          wandb.log({"loss_valid": avg_loss_valid / batch_no})
        except:
          pass

    
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


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, path_save=""):

    mse_total = 0
    mae_total = 0
    evalpoints_total = 0

    all_target = []
    all_observed_point = []
    all_observed_time = []
    all_evalpoint = []
    all_generated_samples = []
    with tqdm(test_loader, mininterval=5.0, maxinterval=5.0) as it:
        for batch_no, test_batch in enumerate(it, start=1):
            output = model.evaluate(test_batch, nsample)

            samples, c_target, eval_points, observed_points, observed_time = output
            samples = tf.transpose(samples, [0, 1, 3, 2]) # (B,nsample,L,K)
            c_target = tf.transpose(c_target, [0, 2, 1])  # (B,L,K)
            eval_points = tf.transpose(eval_points, [0, 2, 1])
            # observed_points = tf.transpose(observed_points, [0, 2, 1])  # observed_points (B,L)

            samples_median = tf.constant(np.median(samples.numpy(), 1), tf.float32)
            all_target.append(c_target)
            all_evalpoint.append(eval_points)
            all_observed_point.append(observed_points)
            all_observed_time.append(observed_time)
            all_generated_samples.append(samples)

            mse_current = (((samples_median - c_target) * eval_points) ** 2) * (scaler ** 2)
            mae_current = (tf.math.abs((samples_median - c_target) * eval_points)) * scaler

            mse_total += tf.reduce_sum(mse_current)
            mae_total += tf.reduce_sum(mae_current)
            evalpoints_total +=  tf.reduce_sum(eval_points)

            it.set_postfix(ordered_dict={
                    "rmse_total": np.sqrt(mse_total.numpy() / evalpoints_total.numpy()),
                    "mae_total": mae_total.numpy() / evalpoints_total.numpy(),
                    "batch_no": batch_no}, refresh=False)
            
        with open(f"{path_save}generated_outputs_nsample"+str(nsample)+".pk","wb") as f:
            all_target = tf.concat(all_target, 0)
            all_evalpoint = tf.concat(all_evalpoint, 0)
            all_observed_point = tf.concat(all_observed_point, 0)
            all_observed_time = tf.concat(all_observed_time, 0)
            all_generated_samples = tf.concat(all_generated_samples, 0)

            pickle.dump(
                [
                    all_generated_samples,
                    all_target,
                    all_evalpoint,
                    all_observed_point,
                    all_observed_time,
                    scaler,
                    mean_scaler,
                ],
                f,
            )

        CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

        with open(f"{path_save}result_nsample" + str(nsample) + ".pk", "wb") as f:
            pickle.dump(
                [
                    np.sqrt(mse_total.numpy()/ evalpoints_total.numpy()),
                    mae_total.numpy() / evalpoints_total.numpy(), 
                    CRPS
                ], 
                f)
            print("RMSE:", np.sqrt(mse_total.numpy()/ evalpoints_total.numpy()))
            print("MAE:", mae_total.numpy() / evalpoints_total.numpy())
            print("CRPS:", CRPS)


    return all_generated_samples.numpy()


def get_tf_trans(heads=8, layers=1, channels=64):
    encoder_layer = tfm.nlp.layers.TransformerEncoderBlock(heads, inner_dim=64, inner_activation='gelu',)
    #  inner_dropout=0.1, output_dropout=0.1)

    return encoder_layer

def Conv1d_with_init(out_channels, kernel_size):
    layer = keras.layers.Conv1D(out_channels, kernel_size, padding='same', data_format='channels_first',
    kernel_initializer= 'he_normal', bias_initializer= 'he_uniform')
    return layer

class DiffusionEmbedding(keras.Model):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super(DiffusionEmbedding, self).__init__()
        if projection_dim is None:
            projection_dim = embedding_dim

        self.embedding = tf.Variable(self._build_embedding(num_steps, embedding_dim / 2), trainable=False)
        # self.embedding = tf.constant(self._build_embedding(num_steps, embedding_dim / 2), tf.float32)
        self.projection1 = keras.layers.Dense(projection_dim,
        kernel_initializer= 'he_uniform', bias_initializer= 'he_uniform')
        self.projection2 = keras.layers.Dense(projection_dim,
        kernel_initializer= 'he_uniform', bias_initializer= 'he_uniform')

    def call(self, diffusion_step):

        x = tf.gather(self.embedding, diffusion_step)
        x = self.projection1(x)
        x = keras.activations.swish(x)
        x = self.projection2(x)
        x = keras.activations.swish(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = tf.expand_dims(tf.constant(np.arange(num_steps), tf.float32),1)  # (T,1)
        frequencies = tf.expand_dims(10.0 ** (tf.constant(np.arange(dim), tf.float32)/ (dim - 1) * 4.0), 0) # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis=1)  # (T,dim*2)
        return table

    
class diff_CSDI(keras.Model):
    def __init__(self, config, inputdim=2):
        super(diff_CSDI, self).__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"])

        self.input_projection = Conv1d_with_init(self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, 1)
        ### note: bias_initial in original PyTorch Conv layer
        bound = 1 / math.sqrt(self.channels)
        self.output_projection2 = keras.layers.Conv1D(1, kernel_size=1, padding='valid', data_format='channels_first',
        kernel_initializer='zeros', bias_initializer=tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound))
        # bias_initializer= 'he_uniform')

        self.residual_layers = []
        for i in range(config["layers"]):
            self.residual_layers.append(ResidualBlock(side_dim=config["side_dim"],
                    channels=self.channels, diffusion_embedding_dim=config["diffusion_embedding_dim"], nheads=config["nheads"]))


    def call(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = tf.reshape(x,[B, inputdim, K * L])
        x = self.input_projection(x)
        x = keras.activations.relu(x)
        x = tf.reshape(x,[B, self.channels, K, L])

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for n in range(len(self.residual_layers)):
            x, skip_connection = self.residual_layers[n](x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x1 = tf.reduce_sum(tf.stack(skip), axis=0) / math.sqrt(len(self.residual_layers))
        x1 = tf.reshape(x1,[B, self.channels, K * L])
        x1 = self.output_projection1(x1)  # (B,channel,K*L)
        x1 = keras.activations.relu(x1)
        x1 = self.output_projection2(x1)  # (B,1,K*L)   
        x1 = tf.reshape(x1,[B, K, L])

        return x1

    
class ResidualBlock(keras.Model):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super(ResidualBlock, self).__init__()
        self.diffusion_projection = keras.layers.Dense(channels, kernel_initializer= 'he_uniform', bias_initializer= 'he_uniform')
        self.cond_projection = Conv1d_with_init(2 * channels, 1)
        self.mid_projection = Conv1d_with_init( 2 * channels, 1)
        self.output_projection = Conv1d_with_init(2 * channels, 1)

        self.time_layer = get_tf_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_tf_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, channel, K, L]), perm=[0, 2, 1, 3]),[B * K, channel, L])
        y = tf.transpose(self.time_layer(tf.transpose(y, [0, 2, 1])), [0, 2, 1]) ### B*K, L, channel

        y = tf.reshape(tf.transpose(tf.reshape(y, [B, K, channel, L]), [0, 2, 1, 3]), [B, channel, K * L])

        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, channel, K, L]), perm=[0, 3, 1, 2]),[B * L, channel, K])
        y = tf.transpose(self.feature_layer(tf.transpose(y, [0, 2, 1])), [0, 2, 1]) ### B*L, K, channel
        y = tf.reshape(tf.transpose(tf.reshape(y, [B, L, channel, K]), [0, 2, 3, 1]), [B, channel, K * L])

        return y

    def call(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = tf.reshape(x, [B, channel, K * L])

        diffusion_emb = tf.expand_dims(self.diffusion_projection(diffusion_emb), -1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)
        
        
        _, cond_dim, _, _ = cond_info.shape
        cond_info = tf.reshape(cond_info, [B, cond_dim, K * L]) 
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = tf.split(y, num_or_size_splits=2, axis=1)
        y = tf.math.sigmoid(gate) * tf.math.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = tf.split(y, num_or_size_splits=2, axis=1)
        x = tf.reshape(x, [B, channel, K, L])
        residual = tf.reshape(residual, [B, channel, K, L])
        skip = tf.reshape(skip, [B, channel, K, L])

        return (x + residual) / math.sqrt(2.0), skip


class CSDI_base(keras.Model):
    def __init__(self, target_dim, config):
        super(CSDI_base, self).__init__()

        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.k_segments_or_k_misssing = config["model"]["k_segments_or_k_misssing"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = keras.layers.Embedding(self.target_dim, self.emb_feature_dim)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_tf = tf.expand_dims(tf.expand_dims(tf.constant(self.alpha, tf.float32), 1), 1)

    def time_embedding(self, pos, d_model=128):
        pe = np.zeros((pos.shape[0], pos.shape[1], d_model))
        position = tf.expand_dims(pos, 2)
        div_term = 1 / tf.math.pow(10000.0, np.arange(0, d_model, 2) / d_model)
        pe[:, :, 0::2] = (tf.math.sin(position * div_term)).numpy()
        pe[:, :, 1::2] = (tf.math.cos(position * div_term)).numpy()

        return tf.constant(pe, tf.float32)

    def get_rm_in_length_mask(self, observed_mask, k_segments_or_k_misssing=50):

        #### generate random mask for each batch ### (B, L)
        masks=[]
        for j in range(len(observed_mask)):
            mask = observed_mask[j].numpy()   ###(L)
            valid_mask=np.where(mask==1)[0]
            perm = np.random.permutation(valid_mask)
            idx = perm[0:k_segments_or_k_misssing]
            mask[idx] = 2.0  #### label missing value as 2
            masks.append(mask)
        masks = np.array(masks) ### (B, L)
        masks = np.tile(masks[:, :, None], (1, 1, 6)) ###(B, L, K)
        # print(masks)
        train_mask = masks.copy()
        train_mask[np.where(masks!=1)]=0
        train_mask = tf.constant(train_mask, dtype=tf.float32)
        # print('111', train_mask)
        loss_mask = masks.copy()
        loss_mask[np.where(masks==2)]=1
        loss_mask[np.where(masks!=2)]=0
        loss_mask = tf.constant(loss_mask, dtype=tf.float32)
        # print('222', loss_mask)  
        train_mask = tf.transpose(train_mask, perm=[0,2,1]) ###(B, K, L)
        loss_mask = tf.transpose(loss_mask, perm=[0,2,1]) ###(B, K, L)
        
        return train_mask, loss_mask

    def get_bm_in_length_mask(self, observed_mask, k_segments_or_k_misssing=5):

        masks = []
        for i in range(len(observed_mask)):
            mask = observed_mask[i].numpy() ###(L)
            valid_mask = np.where(mask==1)[0]
            list_of_segments_index = np.array_split(valid_mask, k_segments_or_k_misssing) ### only valid mask will be selected
            s_nan = random.choice(list_of_segments_index)
            mask[s_nan] = 2 #### label missing value as 2 
            masks.append(mask)              
        masks = np.array(masks) ### (B, L)
        masks = np.tile(masks[:, :, None], (1, 1, 6)) ###(B, L, K)
        train_mask = masks.copy()
        train_mask[np.where(masks!=1)]=0
        train_mask = tf.constant(train_mask, dtype=tf.float32)
        loss_mask = masks.copy()
        loss_mask[np.where(masks==2)]=1
        loss_mask[np.where(masks!=2)]=0
        loss_mask = tf.constant(loss_mask, dtype=tf.float32)
        train_mask = tf.transpose(train_mask, perm=[0,2,1]) ###(B, K, L)
        loss_mask = tf.transpose(loss_mask, perm=[0,2,1]) ###(B, K, L)

        return train_mask, loss_mask
    
    def get_side_info(self, observed_tp, cond_mask):

        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = tf.tile(tf.expand_dims(time_embed, 2), [1, 1, K, 1])
        feature_embed = self.embed_layer(tf.constant(np.arange(self.target_dim), tf.float32))  # (K,emb)
        feature_embed = tf.tile(tf.expand_dims(tf.expand_dims(feature_embed, 0), 0), [B, L, 1, 1])
        side_info = tf.concat([time_embed, feature_embed], -1)  # (B,L,K,*)
        side_info = tf.transpose(side_info, [0, 3, 2, 1])  # (B,*,K,L)
        if self.is_unconditional == False:
            side_mask = tf.expand_dims(cond_mask, 1)  # (B,1,K,L)
            side_info = tf.concat([side_info, side_mask], 1)


        return side_info

    
    def calc_loss_valid(self, observed_data, cond_mask, loss_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(observed_data, cond_mask, loss_mask, side_info, is_train, set_t=t)
            loss_sum += loss.numpy()
            
        return loss_sum / self.num_steps

    
    def calc_loss(self, observed_data, cond_mask, loss_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = tf.cast(tf.ones(B) * set_t, tf.int64)
        else:
            t = np.random.randint(0, self.num_steps, [B])

        current_alpha = tf.gather(self.alpha_tf, t)
        noise = tf.random.normal(observed_data.shape, 0, 1, tf.float32)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        # target_mask = observed_mask - cond_mask
        target_mask = loss_mask
        residual = (noise - predicted) * target_mask
        num_eval = tf.reduce_sum(target_mask)
        loss = tf.reduce_sum(residual ** 2)/ (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = tf.expand_dims(noisy_data, 1)  # (B,1,K,L)
        else:
            cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
            noisy_target = tf.expand_dims((1 - cond_mask) * noisy_data, 1)
            total_input = tf.concat([cond_obs, noisy_target], 1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = np.zeros([B, n_samples, K, L])

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = tf.random.normal(noisy_obs.shape)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = tf.random.normal(observed_data.shape)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = tf.expand_dims(diff_input, 1) # (B,1,K,L)
                else:
                    cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
                    noisy_target = tf.expand_dims((1 - cond_mask) * current_sample, 1)
                    diff_input = tf.concat([cond_obs, noisy_target], 1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, tf.constant([t]))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = tf.random.normal(current_sample.shape)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise
            imputed_samples[:, i] = current_sample.numpy()
            
        return tf.constant(imputed_samples, tf.float32)

    
    def call(self, batch, is_train=1):
        (observed_data,observed_mask,observed_tp) = self.process_data(batch)
        if self.target_strategy == "random missing with length":
            cond_mask, loss_mask = self.get_rm_in_length_mask(observed_mask, self.k_segments_or_k_misssing)
        elif self.target_strategy == "blackout missing with length":
            cond_mask, loss_mask= self.get_bm_in_length_mask(observed_mask, self.k_segments_or_k_misssing)
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, loss_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (observed_data,observed_mask,observed_tp) = self.process_data(batch)
        if self.target_strategy == "random missing with length":
            cond_mask, loss_mask = self.get_rm_in_length_mask(observed_mask, self.k_segments_or_k_misssing)
        elif self.target_strategy == "blackout missing with length":
            cond_mask, loss_mask = self.get_bm_in_length_mask(observed_mask, self.k_segments_or_k_misssing)

        target_mask = loss_mask
        side_info = self.get_side_info(observed_tp, cond_mask)
        samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        
        return samples, observed_data, target_mask, observed_mask, observed_tp

    
class CSDI_Custom(CSDI_base):
    def __init__(self, config, target_dim=35):
        super(CSDI_Custom, self).__init__(target_dim, config)

    def process_data(self, batch):
        observed_data = tf.cast(batch["observed_data"], tf.float32)
        observed_mask = tf.cast(batch["observed_mask"], tf.float32)
        observed_tp = tf.cast(batch["timepoints"], tf.float32)
        observed_data = tf.transpose(observed_data, [0, 2, 1])
        # observed_mask = tf.transpose(observed_mask, [0, 2, 1])


        return (observed_data,observed_mask,observed_tp)
    
class Custom_Train_Dataset():
    def __init__(self, series, masks, use_index_list=None):
        
        if use_index_list is None:
            self.use_index_list = np.arange(series.shape[0])
        else:
            self.use_index_list = use_index_list

        self.length = series.shape[1]
        self.series = tf.gather(series, self.use_index_list)
        # print(self.series.shape)
        self.masks = masks[self.use_index_list]
        # print(self.masks.shape)

    def getdata(self):
        s = {
            "observed_data": self.series,
            "observed_mask": self.masks,
            "timepoints": tf.tile(tf.reshape(tf.constant(np.arange(self.length)), [-1,self.length]), [len(self.use_index_list),1]),
        }
        return s

class Custom_Impute_Dataset():
    def __init__(self, series, masks, use_index_list=None, path_save=''):

        if use_index_list is None:
            self.use_index_list = np.arange(series.shape[0])
        else:
            self.use_index_list = use_index_list

        self.length = series.shape[1]
        self.series = tf.gather(series, self.use_index_list)
        self.masks = masks[self.use_index_list]

    def getdata(self):
        s = {
            "observed_data": self.series,
            "observed_mask": self.masks,
            "timepoints": tf.tile(tf.reshape(tf.constant(np.arange(self.length)), [-1,self.length]), [len(self.use_index_list),1]),
        }
        return s


def get_dataloader_train_impute(series,
                                masks,
                                batch_size=4,
                                train_split=0.7,
                                valid_split=0.9,
                                len_dataset=100,
                                path_save=''):
    indlist = np.arange(len_dataset)
    # print('train:{} val:{}'.format(int(len(indlist) * train_split),int(len(indlist) * (train_split + valid_split))))
    # print(aaaa)
    tr_i, v_i, te_i = np.split(indlist,
                               [int(len(indlist) * train_split),
                                int(len(indlist) * (train_split + valid_split))])
    
    train_dataset = Custom_Train_Dataset(series=series, masks= masks, use_index_list=tr_i).getdata()
    
    train_loader = tf.data.Dataset.from_tensor_slices(train_dataset)  
    train_loader = train_loader.shuffle(len(train_loader), reshuffle_each_iteration=True).cache()
    train_loader = train_loader.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    train_loader = train_loader.prefetch(tf.data.AUTOTUNE)

    valid_dataset = Custom_Train_Dataset(series=series, masks=masks, use_index_list=v_i).getdata()
    valid_loader = tf.data.Dataset.from_tensor_slices(valid_dataset)  
    valid_loader = valid_loader.shuffle(len(valid_loader), reshuffle_each_iteration=True).cache()
    valid_loader = valid_loader.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    valid_loader = valid_loader.prefetch(tf.data.AUTOTUNE )

    test_dataset = Custom_Train_Dataset(series=series, masks=masks, use_index_list=te_i).getdata()
    test_loader = tf.data.Dataset.from_tensor_slices(test_dataset)  
    # test_loader = test_loader.shuffle(len(test_loader), reshuffle_each_iteration=True).cache()
    test_loader = test_loader.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    test_loader = test_loader.prefetch(tf.data.AUTOTUNE)

    return train_loader, valid_loader, test_loader


def get_dataloader_impute(series, masks, batch_size=4, len_dataset=100):
    indlist = np.arange(len_dataset)
    impute_dataset = Custom_Impute_Dataset(series=series, masks=masks, use_index_list=indlist).getdata()
    impute_loader = tf.data.Dataset.from_tensor_slices(impute_dataset)  
    impute_loader = impute_loader.batch(batch_size)    

    return impute_loader



class CSDIImputer:
    def __init__(self):
        np.random.seed(0)
        random.seed(0)
        
        '''
        CSDI imputer
        3 main functions:
        a) training based on two target_strategies, "random missing with length" or "blackout missing with length".
        b) loading weights of already trained model
        c) impute samples in inference. Note, you must manually load weights after training for inference.
        '''

    def train(self,
              series,
              masks,
              k_segments_or_k_misssing,
              train_split = 0.7,
              valid_split = 0.2,
              epochs = 200,
              samples_generate = 10,
              path_save = "",
              batch_size = 16,
              lr = 1.0e-3,
              layers = 4,
              channels = 64,
              nheads = 8,
              difussion_embedding_dim = 128,
              beta_start = 0.0001,
              beta_end = 0.5,
              num_steps = 50,
              schedule = 'quad',
              is_unconditional = 0,
              timeemb = 128,
              featureemb = 16,
              target_strategy = "random missing with length",
             ):
        
        '''
        CSDI training function. 
       
       
        Requiered parameters
        -series: Assumes series of shape (Samples, Length, Channels).
        -masks: the mask of stock, holiday:-1; nan:0; valid:1, assume the shape (Samples, Length).
        -k_segments_or_k_misssing: k_segments for "blackout missing with length", e.g., 5; k_misssing for "random missing with length", e.g., 50.
        -path_save: full path where to save model weights, configuration file, and means and std devs for de-standardization in inference.
        
        Default parameters
        -train_split: 0 to 1 representing the percentage of train set from whole data.
        -valid_split: 0 to 1. Is an adition to train split where 1 - train_split - valid_split = test_split (implicit in method).
        -epochs: number of epochs to train.
        -samples_generate: number of samples to be generated.
        -batch_size: batch size in training.
        -lr: learning rate.
        -layers: difussion layers.
        -channels: number of difussion channels.
        -nheads: number of difussion 'heads'.
        -difussion_embedding_dim: difussion embedding dimmensions. 
        -beta_start: start noise rate.
        -beta_end: end noise rate.
        -num_steps: number of steps.
        -schedule: scheduler. 
        -is_unconditional: conditional or un-conditional imputation. Boolean.
        -timeemb: temporal embedding dimmensions.
        -featureemb: feature embedding dimmensions.
        -target_strategy: strategy of masking, "random missing with length" or "blackout missing with length" during training.
        -wandbiases_project: weight and biases project.
        -wandbiases_experiment: weight and biases experiment or run.
        -wandbiases_entity: weight and biases entity. 
        '''
       
        config = {}
        
        config['train'] = {}
        config['train']['epochs'] = epochs
        config['train']['batch_size'] = batch_size
        config['train']['lr'] = lr
        config['train']['train_split'] = train_split
        config['train']['valid_split'] = valid_split
        config['train']['path_save'] = path_save
        
       
        config['diffusion'] = {}
        config['diffusion']['layers'] = layers
        config['diffusion']['channels'] = channels
        config['diffusion']['nheads'] = nheads
        config['diffusion']['diffusion_embedding_dim'] = difussion_embedding_dim
        config['diffusion']['beta_start'] = beta_start
        config['diffusion']['beta_end'] = beta_end
        config['diffusion']['num_steps'] = num_steps
        config['diffusion']['schedule'] = schedule
        
        config['model'] = {} 
        config['model']['k_segments_or_k_misssing'] = k_segments_or_k_misssing
        config['model']['is_unconditional'] = is_unconditional
        config['model']['timeemb'] = timeemb
        config['model']['featureemb'] = featureemb
        config['model']['target_strategy'] = target_strategy
        
        print(json.dumps(config, indent=4))

        config_filename = path_save + "config_csdi_training"
        print('configuration file name:', config_filename)
        with open(config_filename + ".json", "w") as f:
            json.dump(config, f, indent=4)


        train_loader, valid_loader, test_loader = get_dataloader_train_impute(
            series = series,
            masks = masks,
            train_split=config["train"]["train_split"],
            valid_split=config["train"]["valid_split"],
            len_dataset=series.shape[0],
            batch_size=config["train"]["batch_size"],
            path_save=config['train']['path_save'])

        model = CSDI_Custom(config, target_dim=series.shape[2])
        train(model=model,
              config=config["train"],
              train_loader=train_loader,
              valid_loader=valid_loader,
              path_save=config['train']['path_save'])

        evaluate(model=model,
                 test_loader=test_loader,
                 nsample=samples_generate,
                 scaler=1,
                 path_save=config['train']['path_save'])
        
        
    def load_weights(self, 
                     path_load_model='',
                     path_config=''):
        
        self.path_load_model_dic = path_load_model
        self.path_config = path_config
    
    
        '''
        Load weights and configuration file for inference.
        
        path_load_model: load model weights
        path_config: load configuration file
        '''
    

    def impute(self,
               sample,
               masks,
               n_samples = 50,
               ):
        
        '''
        Imputation function 
        sample: sample(s) to be imputed (Samples, Length, Channel)
        mask: mask where values to be imputed. 0's to impute, 1's to remain. 
        n_samples: number of samples to be generated
        return imputations with shape (Samples, N imputed samples, Length, Channel)
        '''
        
        if len(sample.shape) == 2:
            self.series_impute = tf.constant(np.expand_dims(sample, axis=0))
        elif len(sample.shape) == 3:
            self.series_impute = sample


        with open(self.path_config, "r") as f:
            config = json.load(f)

        test_loader = get_dataloader_impute(series=self.series_impute, masks=masks, len_dataset=len(self.series_impute),
                                             batch_size=config['train']['batch_size'])

        model = CSDI_Custom(config, target_dim=self.series_impute.shape[2])
        # output = model({
        #     "observed_data": tf.ones([8, 103, 6]),
        #     "observed_mask": tf.ones([8, 103]),
        #     "timepoints": tf.ones([8, 103])})

        model.load_weights(self.path_load_model_dic)

        imputations = evaluate(model=model,
                                test_loader=test_loader,
                                nsample=n_samples,
                                scaler=1,
                                path_save='')
        
    


