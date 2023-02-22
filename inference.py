import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from util import get_mask_mnr, get_mask_bm, get_mask_rm
from util import find_max_epoch, sampling, calc_diffusion_hyperparams
from SSSDS4Imputers import SSSDS4Imputer

from sklearn.metrics import mean_squared_error
from statistics import mean


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

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # predefine model
    model = SSSDS4Imputer(**model_config)
    # define optimizer
    opt = keras.optimizers.Adam(learning_rate=2e-4)
    model.compile(loss='mse', optimizer=opt)

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}ckpt'.format(ckpt_iter))
    try:
        model.load_weights(model_path)
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

    ### Custom data loading and reshaping ###

    testing_data = np.load(trainset_config['test_data_path'])
    testing_data = np.split(testing_data, 4, 0)
    testing_data = np.array(testing_data)
    testing_data = tf.convert_to_tensor(testing_data, dtype=tf.dtypes.float32)
    print('Data loaded')

    all_mse = []

    for i, batch in enumerate(testing_data):

        if masking == 'mnr':
            mask_T = get_mask_mnr(batch[0], missing_k)
            # mask = mask_T.permute(1, 0)
            # mask = mask.repeat(batch.size()[0], 1, 1)
            # mask = tf.constant(mask, dtype=tf.dtypes.float32)

        elif masking == 'bm':
            mask_T = get_mask_bm(batch[0], missing_k)

        elif masking == 'rm':
            mask_T = get_mask_rm(batch[0], missing_k)

        mask = np.transpose(mask_T)
        mask = np.expand_dims(mask, axis=0)
        mask = np.tile(mask, reps=[batch.shape[0], 1, 1])
        mask = tf.constant(mask, dtype=tf.dtypes.float32)

        batch = tf.transpose(batch, perm=[0, 2, 1])

        sample_length = batch.shape[2]
        sample_channels = batch.shape[1]
        generated_audio = sampling(model, (num_samples, sample_channels, sample_length),
                                   diffusion_hyperparams,
                                   cond=batch,
                                   mask=mask,
                                   only_generate_missing=only_generate_missing)

        print('generated {} utterances of random_digit at iteration {} in {} seconds'.format(num_samples,
                                                                                             ckpt_iter,
                                                                                             int(start.elapsed_time(
                                                                                                 end) / 1000)))

        generated_audio = generated_audio.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        outfile = f'imputation{i}.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, generated_audio)

        outfile = f'original{i}.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, batch)

        outfile = f'mask{i}.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, mask)

        print('saved generated samples at iteration %s' % ckpt_iter)

        mse = mean_squared_error(generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)])
        all_mse.append(mse)

    print('Total MSE:', mean(all_mse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_SSSDS4.json',
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
    model_config = config['wavenet_config']

    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=args.num_samples,
             use_model=train_config["use_model"],
             data_path=trainset_config["test_data_path"],
             masking=train_config["masking"],
             missing_k=train_config["missing_k"],
             only_generate_missing=train_config["only_generate_missing"])