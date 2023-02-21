import os
import numpy as np
import random
import tensorflow as tf


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        # name = f.split('.')[0]
        # if name[-4:] == 'ckpt':
        #     try:
        #         epoch = max(epoch, int(name[:-4]))

        # if len(f) <= 3:
        #     continue
        # if f[-3:] == '.h5':
        #     try:
        #         epoch = max(epoch, int(f[:-3]))

        # if f[-3:] == 'cpm':
        #     try:
        #         epoch = max(epoch, int(f[:-3]))

        # if len(f) <= 4:
        #     continue
        # if f[-4:] == '.pkl':
        #     try:
        #         epoch = max(epoch, int(f[:-4]))

        name = f.split('.')
        # if len(name) != 3:
        #     continue
        # if name[1] == 'pkl':
        if len(name) != 2:
            continue
        name = name[0]
        if name[-4:] == 'ckpt':
            try:
                epoch = max(epoch, int(name[:-4]))
            except:
                continue
    return epoch


# Utilities for diffusion models
def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """
    # print('util | calc_diffusion_step_embedding | diffusion_steps', diffusion_steps)
    # print('util | calc_diffusion_step_embedding | diffusion_step_embed_dim_in', diffusion_step_embed_dim_in)
    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = tf.math.exp(tf.range(half_dim, dtype=_embed.dtype) * -_embed)
    _embed = tf.cast(diffusion_steps, dtype=_embed.dtype) * _embed
    diffusion_step_embed = tf.concat((tf.math.sin(_embed),
                                      tf.math.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = np.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = tf.math.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, size, diffusion_hyperparams, cond, mask, only_generate_missing=0, guidance_weight=0):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = tf.random.normal(size)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            if only_generate_missing == 1:
                x = x * (1 - mask).float() + cond * mask.float()
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            x = (x - (1 - Alpha[t]) / tf.math.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / tf.math.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * tf.random.normal(size)  # add the variance term to x_{t-1}

    return x


def training_loss(net, X, diffusion_hyperparams, only_generate_missing=1):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = np.random.randint(T, size=B)
    z = tf.random.normal(audio.shape)
    if only_generate_missing == 1:
        z = audio * mask + z * (1 - mask)
    # print('util | training_loss | Alpha_bar: ', Alpha_bar)
    # print('util | training_loss | diffusion_steps: ', diffusion_steps)
    # print('util | training_loss | Alpha_bar[diffusion_steps]: ', Alpha_bar[diffusion_steps])
    transformed_X = tf.math.sqrt( tf.constant(Alpha_bar[diffusion_steps], dtype=tf.float32, shape=[B,1,1]) ) * audio +\
                    tf.math.sqrt( tf.constant((1 - Alpha_bar[diffusion_steps]), dtype=tf.float32, shape=[B,1,1]) ) * z
    # epsilon_theta = net(
    #     (transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta
    # print('util | training_loss | transformed_X.shape: ', transformed_X.shape)
    # print('util | training_loss | z.shape: ', z.shape)"""

    # noise, conditional, mask, diffusion_steps = input_data
    if only_generate_missing == 1:
        return net.train_on_batch( x=(transformed_X, cond, mask, tf.constant(diffusion_steps, shape=[B,1])), y=z*loss_mask )
        # return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
    elif only_generate_missing == 0:
        return net.train_on_batch( x=(transformed_X, cond, mask, tf.constant(diffusion_steps, shape=[B,1])), y=z )
        # return loss_fn(epsilon_theta, z)


def get_mask_rm(sample, k):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    mask = np.ones(sample.shape)
    length_index = range(mask.shape[0])  # lenght of series indexes
    for channel in range(mask.shape[1]):
        perm = np.random.permutation(len(length_index))
        idx = perm[0:k]
        mask[:, channel][idx] = 0
    return mask


def get_mask_mnr(sample, k):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    mask = np.ones(sample.shape)
    length_index = range(mask.shape[0])
    list_of_segments_index = np.array_split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0
    return mask


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""
    mask = np.ones(sample.shape)
    length_index = range(mask.shape[0])
    list_of_segments_index = np.array_split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0
    return mask
