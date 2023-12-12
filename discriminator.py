### The Discriminator
# The discriminator is a CNN-based image classifier.

from config import *
from blocks import *

from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa

def make_discriminator_model(choice):
    if choice == 'sagan':
        d_kernel = 5
        df_dim = 8
        inputs = layers.Input(shape= (data_dim, data_channel))
        x = D_res_start_block(inputs, df_dim, d_kernel)  # df_dim//2
        x = D_res_block(x, df_dim * 2, d_kernel)  # df_dim//4
        x = attn_block(x, hidden_ratio=8, g_ratio=2)  # df_dim//4
        x = D_res_block(x, df_dim * 4, d_kernel)  # df_dim//8
        x = D_res_block(x, df_dim * 8, d_kernel)  # df_dim//16
        x = D_res_block(x, df_dim * 16, d_kernel)  # df_dim//32
        x = D_res_keep_block(x, d_kernel)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = tf.reduce_sum(x, [1])
        y = tfa.layers.SpectralNormalization(layers.Dense(1))(x)
        model = tf.keras.Model(inputs, y)

    elif choice == 'biggan':
        d_kernel = 5
        df_dim = 8
        additional = {'kernel_initializer': tf.keras.initializers.Orthogonal(),
                'kernel_regularizer': orthogonal_reg}
        inputs = layers.Input(shape= (data_dim, data_channel))
        x = D_res_start_block(inputs, df_dim, d_kernel, **additional)  # df_dim//2
        x = D_res_block(x, df_dim * 2, d_kernel, **additional)  # df_dim//4
        x = attn_block(x, hidden_ratio=8, g_ratio=2, **additional)  # df_dim//4
        x = D_res_block(x, df_dim * 4, d_kernel, **additional)  # df_dim//8
        x = D_res_block(x, df_dim * 8, d_kernel, **additional)  # df_dim//16
        x = D_res_block(x, df_dim * 16, d_kernel, **additional)  # df_dim//32
        x = D_res_keep_block(x, d_kernel, **additional)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = tf.reduce_sum(x, [1])
        y = tfa.layers.SpectralNormalization(layers.Dense(1,**additional))(x)
        model = tf.keras.Model(inputs, y)

    elif choice == 'biggan_deep':
        d_kernel = 5
        df_dim = 8
        additional = {'kernel_initializer': tf.keras.initializers.Orthogonal(),
                'kernel_regularizer': orthogonal_reg}
        inputs = layers.Input(shape= (data_dim, data_channel))
        x = tfa.layers.SpectralNormalization(layers.Conv1D(df_dim, d_kernel, strides=1, padding='same', **additional))(inputs)
        x = D_res_bottleneck_block(x, df_dim, d_kernel, downsample=True, hidden_ratio=2, **additional)  # df_dim//2
        x = D_res_bottleneck_block(x, df_dim * 2, d_kernel, downsample=False, hidden_ratio=2, **additional)  # df_dim//2
        x = D_res_bottleneck_block(x, df_dim * 2, d_kernel, downsample=True, hidden_ratio=2, **additional)  # df_dim//4
        x = D_res_bottleneck_block(x, df_dim * 4, d_kernel, downsample=False, hidden_ratio=2, **additional)  # df_dim//4
        x = attn_block(x, hidden_ratio=8, g_ratio=2, **additional)  # df_dim//4
        x = D_res_bottleneck_block(x, df_dim * 4, d_kernel, downsample=True, hidden_ratio=2, **additional)  # df_dim//8
        x = D_res_bottleneck_block(x, df_dim * 8, d_kernel, downsample=False, hidden_ratio=2, **additional)  # df_dim//8
        x = D_res_bottleneck_block(x, df_dim * 8, d_kernel, downsample=True, hidden_ratio=2, **additional)  # df_dim//16
        x = D_res_bottleneck_block(x, df_dim * 16, d_kernel, downsample=False, hidden_ratio=2, **additional)  # df_dim//16
        x = D_res_bottleneck_block(x, df_dim * 16, d_kernel, downsample=False, hidden_ratio=2, **additional)  # df_dim//16
        x = D_res_bottleneck_block(x, df_dim * 16, d_kernel, downsample=False, hidden_ratio=2, **additional)  # df_dim//16
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = tf.reduce_sum(x, [1])
        y = tfa.layers.SpectralNormalization(layers.Dense(1,**additional))(x)
        model = tf.keras.Model(inputs, y)

    elif choice == 'ylgan':
        d_kernel = 5
        df_dim = 8
        inputs = layers.Input(shape= (data_dim, data_channel))
        x = D_res_start_block(inputs, df_dim, d_kernel)  # df_dim//2
        x = D_res_block(x, df_dim * 2, d_kernel)  # df_dim//4
        x = multi_head_attn_block(x, hidden_ratio=4, g_ratio=2, nH=4, sparse=True)  # df_dim//4
        x = D_res_block(x, df_dim * 4, d_kernel)  # df_dim//8
        x = D_res_block(x, df_dim * 8, d_kernel)  # df_dim//16
        x = D_res_block(x, df_dim * 16, d_kernel)  # df_dim//32
        x = D_res_keep_block(x, d_kernel)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = tf.reduce_sum(x, [1])
        y = tfa.layers.SpectralNormalization(layers.Dense(1))(x)
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
        def down_sampling(x):
            _, location_num, num_channels = x.shape.as_list()
            return tf.reshape(x, [-1, location_num//2, num_channels*2])
        df_dim = 64
        inputs = layers.Input(shape= (data_dim, data_channel))
        x = tfa.layers.SpectralNormalization(layers.Conv1D(df_dim, 1, strides=1, padding='same'))(inputs)
        # x = CloseToken()(x)
        x = PosEmbed()(x)
        x = block(x)
        x = block(x)
        x = block(x)
        x = layers.LayerNormalization()(x)
        # x = tf.reduce_mean(x, [1])
        x = tf.keras.layers.Flatten()(x)
        y = tfa.layers.SpectralNormalization(layers.Dense(1))(x)
        model = tf.keras.Model(inputs, y)

    elif choice in ['sn_lsgan', 'sn_dcgan']:
        d_kernel = 5
        df_dim = 32
        def block(model, out_channels):
            model.add(tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, d_kernel, strides=2, padding='same')))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(data_dim, data_channel)))
        model.add(tfa.layers.SpectralNormalization(layers.Conv1D(df_dim, d_kernel, strides=2, padding='same')))
        model.add(layers.LeakyReLU())

        block(model, df_dim*2)
        block(model, df_dim*4)
        # block(model, df_dim*8)

        model.add(layers.Flatten())
        model.add(tfa.layers.SpectralNormalization(layers.Dense(1)))

    elif choice in ['lsgan', 'dcgan', 'wgan_gp', 'dragan', 'ragan', 'ralsgan']:
        d_kernel = 5
        df_dim = 32

        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(data_dim, data_channel)))
        model.add(layers.Conv1D(df_dim, d_kernel, strides=2, padding='same'))
        model.add(layers.LeakyReLU())

        D_single_block(model, df_dim*2, d_kernel)
        D_single_block(model, df_dim*4, d_kernel)

        model.add(layers.Flatten())
        model.add(layers.Dense(1, use_bias=False))

    elif choice == 'wgan':
        d_kernel = 5
        df_dim = 32
        additional = {'kernel_initializer': tf.random_normal_initializer(stddev=0.02),
                'kernel_constraint': lambda w: tf.clip_by_value(w,-0.01,0.01)}

        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(data_dim, data_channel)))
        model.add(layers.Conv1D(df_dim, d_kernel, strides=2, padding='same', **additional))
        model.add(layers.LeakyReLU())

        D_single_block(model, df_dim*2, d_kernel, **additional)
        D_single_block(model, df_dim*4, d_kernel, **additional)

        model.add(layers.Flatten())
        model.add(layers.Dense(1, use_bias=False, **additional))

    else:
        raise(f"Model {choice} is not specified!")

    return model



def discriminator_loss(discriminator, real_output, fake_output, choice = 'sagan', generated_images = None, images= None):
    if choice in ['sagan','ylgan','biggan','biggan_deep']: # hinge loss
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
        total_loss = real_loss + fake_loss
    elif choice == 'transgan':
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
        total_loss = real_loss + fake_loss
    elif choice == 'lsgan': # LS loss
        real_loss = ls_loss(tf.ones_like(real_output), real_output)
        fake_loss = ls_loss(-tf.ones_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
    elif choice == 'dcgan': # standard loss
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
    elif choice == 'ragan': # realistic loss
        real_loss = cross_entropy(tf.ones_like(real_output), real_output - tf.reduce_mean(fake_output))
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output - tf.reduce_mean(real_output))
        total_loss = real_loss + fake_loss
    elif choice == 'ralsgan':
        real_loss = ls_loss(tf.ones_like(real_output), real_output - tf.reduce_mean(fake_output))
        fake_loss = ls_loss(-tf.ones_like(fake_output), fake_output - tf.reduce_mean(real_output))
        total_loss = real_loss + fake_loss
    elif choice == 'wgan':
        real_loss = -tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        total_loss = real_loss + fake_loss
    elif choice == 'wgan_gp':
        real_loss = -tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)

        alpha = tf.random.uniform([images.shape[0],1,1])
        interpolates = alpha*images + ((1-alpha)*generated_images)
        disc_interpolates = discriminator(interpolates, training=True)
        gradients = tf.gradients(disc_interpolates, interpolates)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1)**2)
        lam = 10
        total_loss = real_loss + fake_loss + lam*gradient_penalty

    elif choice == 'dragan':
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

        alpha = tf.random.uniform([images.shape[0],1,1])
        interpolates = alpha*images + ((1-alpha)*generated_images)
        disc_interpolates = discriminator(interpolates, training=True)
        gradients = tf.gradients(disc_interpolates, interpolates)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1)**2)
        lam = 10
        total_loss = real_loss + fake_loss + lam*gradient_penalty

    else:
        raise(f"Model {choice} is not specified!")

    return total_loss



def discriminator_optimizer_fun(choice):
    if choice == 'sagan':
        return tf.keras.optimizers.Adam(4e-4, beta_1=0.5)
    elif choice == 'ylgan':
        return tf.keras.optimizers.Adam(4e-4, beta_1=0.5)
    elif choice in ['biggan','biggan_deep']:
        return tf.keras.optimizers.Adam(4e-4, beta_1=0)
    elif choice == 'transgan':
        return tf.keras.optimizers.Adam(5*1e-4, beta_1=0)
    elif choice == 'wgan_gp':
        return tf.keras.optimizers.Adam(5*1e-4, beta_1=0., beta_2=0.9)
    elif choice == 'dragan':
        return tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    elif choice == 'wgan':
        return tf.keras.optimizers.RMSprop(5*5e-5)
    elif choice == 'lsgan':
        return tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    elif choice in ['ragan','ralsgan']:
        return tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    elif choice == 'dcgan':
        return tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    else:
        raise(f"Model {choice} is not specified!")