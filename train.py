import argparse
import json
import os
from tensorflow import keras
import tensorflow as tf
import numpy as np

from util import calc_diffusion_hyperparams, find_max_epoch, get_mask_rm, get_mask_mnr, get_mask_bm, training_loss
from SSSDS4Imputers import SSSDS4Imputer



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
    model = SSSDS4Imputer(**model_config)
    # define optimizer
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=opt)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            model.load_weights(model_path)
            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    ### Custom data loading and reshaping ###
    training_data = np.load(trainset_config['train_data_path'])
    training_data = np.split(training_data, 160, 0)
    training_data = np.array(training_data)
    training_data = tf.convert_to_tensor(training_data, dtype=tf.dtypes.float32)
    print('Data loaded')

    model.summary()
    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for batch in training_data:
            # print(batch[0].shape)
            if masking == 'rm':
                transposed_mask = get_mask_rm(batch[0], missing_k)
            elif masking == 'mnr':
                transposed_mask = get_mask_mnr(batch[0], missing_k)
            elif masking == 'bm':
                transposed_mask = get_mask_bm(batch[0], missing_k)
            print(masking, transposed_mask.shape)

            mask = np.transpose(transposed_mask)
            mask = np.expand_dims(mask, axis=0)
            mask = np.tile(mask, reps=[batch.shape[0], 1, 1])
            loss_mask = (mask == 0)
            mask = tf.Variable(mask, dtype=tf.dtypes.float32)
            loss_mask = tf.Variable(loss_mask, dtype=tf.dtypes.float32)
            # print('train | batch.shape before:', batch.shape)
            batch = tf.transpose(batch, perm=[0, 2, 1])
            # print('train | batch.shape after:', batch.shape)
            assert batch.shape == mask.shape == loss_mask.shape

            # back-propagation
            X = batch, batch, mask, loss_mask
            loss = training_loss(model, X, diffusion_hyperparams,
                                 only_generate_missing=only_generate_missing)

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_SSSDS4.json',
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
    # print(diffusion_hyperparams)

    global model_config
    model_config = config['wavenet_config']

    train(**train_config)