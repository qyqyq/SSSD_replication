import math
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from S4Model import S4Layer
from util import calc_diffusion_step_embedding

class Residual_block(keras.layers.Layer):
    def __init__(self, res_channels, skip_channels,
                 diffusion_step_embed_dim_out, in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        self.fc_t = keras.layers.Dense(self.res_channels)

        self.S41 = S4Layer(features=2 * self.res_channels,
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)

        self.conv_layer = keras.layers.Conv2D(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2 * self.res_channels,
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)

        self.cond_conv = keras.layers.Conv2D(2 * in_channels, 2 * self.res_channels, kernel_size=1)

        self.res_conv = keras.layers.Conv1D(res_channels, res_channels, kernel_size=1)
        self.res_conv = tfa.layers.WeightNormalization(self.res_conv)
        keras.initializers.HeNormal(self.res_conv.weight)

        self.skip_conv = keras.layers.Conv1D(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = tfa.layers.WeightNormalization(self.skip_conv)
        keras.initializers.HeNormal(self.skip_conv.weight)

    def call(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])
        h = h + part_t

        h = self.conv_layer(h)
        h = self.S41(h.permute(2, 0, 1)).permute(1, 2, 0)

        assert cond is not None
        cond = self.cond_conv(cond)
        h += cond

        h = self.S42(h.permute(2, 0, 1)).permute(1, 2, 0)

        out = keras.activations.tanh(h[:, :self.res_channels, :]) * keras.activations.sigmoid(h[:, self.res_channels:, :])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability

class Residual_group(keras.layers.Layer):
    def __init__(self, res_channels, skip_channels, num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = keras.layers.Dense(diffusion_step_embed_dim_mid)
        self.fc_t2 = keras.layers.Dense(diffusion_step_embed_dim_out)

        self.residual_blocks = []
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))

    def call(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = keras.activations.swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = keras.activations.swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))
            skip += skip_n

        return skip * math.sqrt(1.0 / self.num_res_layers)

class SSSDS4Imputer(keras.Model):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels,
                 num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSDS4Imputer, self).__init__()
        self.init_conv = keras.layers.Conv2D(res_channels, kernel_size=1, activation='relu')

        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)

        self.final_conv = keras.Sequential([
            keras.layers.Conv2D(skip_channels, skip_channels, kernel_size=1, activation='relu'),
            keras.layers.Conv1D(skip_channels, out_channels, kernel_size=1, padding='same') # padding: 'same', padding zeroes evenly to the left/right
        ])

    def call(self, input_data):
        noise, conditional, mask, diffusion_steps = input_data

        conditional = conditional * mask
        conditional = tf.concat([conditional, mask.float()], axis=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y