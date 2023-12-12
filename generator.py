from config import *
from blocks import *

from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa

def make_generator_model(choice):
    if choice == 'sagan':
        g_kernel = 5
        gf_dim = 8
        inputs = layers.Input(shape= (noise_dim,))
        x = tfa.layers.SpectralNormalization(layers.Dense(data_dim*gf_dim, input_shape=(noise_dim,)))(inputs)
        x = layers.Reshape((data_dim//16, gf_dim * 16))(x)
        x = G_res_block(x, gf_dim*8, g_kernel) # data_dim //8
        x = G_res_block(x, gf_dim*4, g_kernel) # data_dim //4
        x = attn_block(x, hidden_ratio=8, g_ratio=2) # data_dim //4
        x = G_res_block(x, gf_dim*2, g_kernel) # data_dim //2
        x = G_res_block(x, gf_dim, g_kernel) # data_dim
        y = G_out_sn_block(x, data_channel, g_kernel) # data_dim
        model = tf.keras.Model(inputs, y)

    elif choice == 'biggan':
        g_kernel = 5
        gf_dim = 8
        additional = {'kernel_initializer': tf.keras.initializers.Orthogonal(),
                'kernel_regularizer': orthogonal_reg}
        inputs = layers.Input(shape= (noise_dim,))
        x = tfa.layers.SpectralNormalization(layers.Dense(data_dim*gf_dim, input_shape=(noise_dim,), **additional))(inputs)
        x = layers.Reshape((data_dim//16, gf_dim * 16))(x)
        x = G_res_block(x, gf_dim*8, g_kernel, **additional) # data_dim //8
        x = G_res_block(x, gf_dim*4, g_kernel, **additional) # data_dim //4
        x = attn_block(x, hidden_ratio=8, g_ratio=2, **additional) # data_dim //4
        x = G_res_block(x, gf_dim*2, g_kernel, **additional) # data_dim //2
        x = G_res_block(x, gf_dim, g_kernel, **additional) # data_dim
        y = G_out_sn_block(x, data_channel, g_kernel, **additional) # data_dim
        model = tf.keras.Model(inputs, y)

    elif choice == 'biggan_deep':
        g_kernel = 5
        gf_dim = 8
        additional = {'kernel_initializer': tf.keras.initializers.Orthogonal(),
                'kernel_regularizer': orthogonal_reg}
        inputs = layers.Input(shape= (noise_dim,))
        x = tfa.layers.SpectralNormalization(layers.Dense(data_dim*gf_dim, input_shape=(noise_dim,), **additional))(inputs)
        x = layers.Reshape((data_dim//16, gf_dim * 16))(x)
        x = G_res_bottleneck_block(x, gf_dim*16, g_kernel, upsample=False, hidden_ratio=4, **additional)
        x = G_res_bottleneck_block(x, gf_dim*8, g_kernel, upsample=True, hidden_ratio=4, **additional) # data_dim //8
        x = G_res_bottleneck_block(x, gf_dim*8, g_kernel, upsample=False, hidden_ratio=4, **additional)
        x = G_res_bottleneck_block(x, gf_dim*4, g_kernel, upsample=True, hidden_ratio=4, **additional) # data_dim //4
        x = attn_block(x, hidden_ratio=8, g_ratio=2, **additional) # data_dim //4
        x = G_res_bottleneck_block(x, gf_dim*4, g_kernel, upsample=False, hidden_ratio=4, **additional)
        x = G_res_bottleneck_block(x, gf_dim*2, g_kernel, upsample=True, hidden_ratio=4, **additional) # data_dim //2
        x = G_res_bottleneck_block(x, gf_dim*2, g_kernel, upsample=False, hidden_ratio=4, **additional)
        x = G_res_bottleneck_block(x, gf_dim, g_kernel, upsample=True, hidden_ratio=4, **additional) # data_dim
        y = G_out_sn_block(x, data_channel, g_kernel, **additional) # data_dim
        model = tf.keras.Model(inputs, y)

    elif choice == 'ylgan':
        g_kernel = 5
        gf_dim = 8
        inputs = layers.Input(shape= (noise_dim,))
        x = tfa.layers.SpectralNormalization(layers.Dense(data_dim*gf_dim, input_shape=(noise_dim,)))(inputs)
        x = layers.Reshape((data_dim//16, gf_dim * 16))(x)
        x = G_res_block(x, gf_dim*8, g_kernel) # data_dim //8
        x = G_res_block(x, gf_dim*4, g_kernel) # data_dim //4
        x = multi_head_attn_block(x, hidden_ratio=4, g_ratio=2, nH=4, sparse=True) # data_dim //4
        x = G_res_block(x, gf_dim*2, g_kernel) # data_dim //2
        x = G_res_block(x, gf_dim, g_kernel) # data_dim
        y = G_out_sn_block(x, data_channel, g_kernel) # data_dim
        model = tf.keras.Model(inputs, y)

    elif choice == 'transgan':
        def mlp_block(x, ratio=4):
            x0 = x
            num_channels = x.shape[-1]
            x = tfa.layers.SpectralNormalization(layers.Conv1D(num_channels * ratio, 1, strides=1, padding='same'))(x)
            x = tf.keras.activations.gelu(x)
            x = tfa.layers.SpectralNormalization(layers.Conv1D(num_channels, 1, strides=1, padding='same'))(x)
            return x0 + x
        def block(x):
            x = layers.LayerNormalization()(x)
            x = multi_head_attn_block(x, hidden_ratio=2, g_ratio=2, nH=4, sparse=False)
            x = layers.LayerNormalization()(x)
            x = mlp_block(x)
            return x
        def up_sampling(x):
            _, location_num, num_channels = x.shape.as_list()
            return tf.reshape(x, [-1, location_num*2, num_channels//2])
        gf_dim = 16
        inputs = layers.Input(shape= (noise_dim,))
        x = tfa.layers.SpectralNormalization(layers.Dense(data_dim*gf_dim, input_shape=(noise_dim,)))(inputs)
        x = layers.Reshape((data_dim//4, gf_dim*4))(x)
        x = PosEmbed()(x)
        x = block(x)
        x = up_sampling(x)
        x = PosEmbed()(x)
        x = block(x)
        x = up_sampling(x)
        x = PosEmbed()(x)
        x = block(x)
        y = tfa.layers.SpectralNormalization(layers.Conv1D(data_channel, 1, strides=1, padding='same'))(x)
        model = tf.keras.Model(inputs, y)

    elif choice in ['sn_lsgan', 'sn_dcgan']:
        g_kernel = 5
        gf_dim = 32
        act = layers.LeakyReLU
        def block(model, out_channels):
            model.add(layers.BatchNormalization())
            model.add(act())
            model.add(tfa.layers.SpectralNormalization(layers.Conv1DTranspose(out_channels, g_kernel, strides=2, padding='same')))

        model = tf.keras.Sequential()
        model.add(tfa.layers.SpectralNormalization(layers.Dense(data_dim * gf_dim, input_shape=(noise_dim,))))
        model.add(layers.Reshape((data_dim//8, gf_dim*8)))

        # block(model, gf_dim*8)
        block(model, gf_dim*4)
        block(model, gf_dim*2)
        block(model, gf_dim)

        model.add(layers.BatchNormalization())
        model.add(act())
        model.add(tfa.layers.SpectralNormalization(layers.Conv1DTranspose(data_channel, g_kernel, strides=1, padding='same', activation= 'linear')))

    elif choice in ['lsgan', 'dcgan', 'wgan_gp', 'dragan', 'ragan', 'ralsgan']:
        g_kernel = 5
        gf_dim = 32
        act = layers.LeakyReLU

        model = tf.keras.Sequential()
        model.add(layers.Dense(data_dim * gf_dim, input_shape=(noise_dim,)))
        model.add(layers.Reshape((data_dim//8, gf_dim*8)))

        G_single_block(model, gf_dim*4, g_kernel, act)
        G_single_block(model, gf_dim*2, g_kernel, act)
        G_single_block(model, gf_dim, g_kernel, act)

        model.add(layers.BatchNormalization())
        model.add(act())
        model.add(layers.Conv1DTranspose(data_channel, g_kernel, strides=1, padding='same', activation= 'linear'))

    elif choice == 'wgan':
        g_kernel = 5
        gf_dim = 32
        act = layers.LeakyReLU
        additional = {'kernel_initializer': tf.random_normal_initializer(stddev=0.02)}

        model = tf.keras.Sequential()
        model.add(layers.Dense(data_dim * gf_dim, input_shape=(noise_dim,), **additional))
        model.add(layers.Reshape((data_dim//8, gf_dim*8)))

        G_single_block(model, gf_dim*4, g_kernel, act, **additional)
        G_single_block(model, gf_dim*2, g_kernel, act, **additional)
        G_single_block(model, gf_dim, g_kernel, act, **additional)

        model.add(layers.BatchNormalization())
        model.add(act())
        model.add(layers.Conv1DTranspose(data_channel, g_kernel, strides=1, padding='same', activation= 'linear', **additional))

    else:
        raise(f"Model {choice} is not specified!")

    return model



def generator_loss(fake_output, choice, real_output = None):
    if choice in ['sagan','wgan','wgan_gp','ylgan','biggan','biggan_deep','transgan']:
        loss = -tf.reduce_mean(fake_output)
    elif choice == 'lsgan':
        loss = ls_loss(tf.ones_like(fake_output), fake_output)
    elif choice in ['dcgan', 'dragan']:
        loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    elif choice == 'ragan':
        real_loss = cross_entropy(tf.zeros_like(real_output), real_output - tf.reduce_mean(fake_output))
        fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output - tf.reduce_mean(real_output))
        loss = real_loss + fake_loss
    elif choice == 'ralsgan':
        real_loss = ls_loss(-tf.ones_like(real_output), real_output - tf.reduce_mean(fake_output))
        fake_loss = ls_loss(tf.ones_like(fake_output), fake_output - tf.reduce_mean(real_output))
        loss = real_loss + fake_loss
    else:
        raise(f"Model {choice} is not specified!")
    return loss



def generator_optimizer_fun(choice):
    if choice in ['sagan','lsgan','ragan','ralsgan','dcgan']:
        return tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    elif choice == 'ylgan':
        return tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    elif choice in ['biggan','biggan_deep']:
        return tf.keras.optimizers.Adam(1e-4, beta_1=0)
    elif choice == 'transgan':
        return tf.keras.optimizers.Adam(1e-4, beta_1=0)
    elif choice == 'dragan':
        return tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    elif choice == 'wgan_gp':
        return tf.keras.optimizers.Adam(1e-4, beta_1=0., beta_2=0.9)
    elif choice == 'wgan':
        return tf.keras.optimizers.RMSprop(5e-5)
    else:
        raise(f"Model {choice} is not specified!")