## Create the models
# Both the generator and discriminator are defined using the
# [Keras Sequential API](https://www.tensorflow.org/guide/keras#sequential_model).

import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa

# self defined layers
# residual link used in attention
class ResidualLink(layers.Layer):
    def __init__(self):
        super(ResidualLink, self).__init__()
    def build(self, input_shape):
        self.sigma = tf.Variable(0.0, trainable=True)
    def call(self, x, attn):
        return x + self.sigma * attn

# position embedding in transformers
class PosEmbed(layers.Layer):
    def __init__(self):
        super(PosEmbed, self).__init__()
    def build(self, input_shape):
        _, location_num, num_channels = input_shape.as_list()
        self.paras = tf.Variable(tf.random.truncated_normal([1,location_num, num_channels], stddev=0.02), trainable=True)
    def call(self, inputs):
        return inputs + self.paras

# not used
class CloseToken(layers.Layer):
    def __init__(self):
        super(CloseToken, self).__init__()
    def build(self, input_shape):
        _, location_num, num_channels = input_shape.as_list()
        paras = tf.Variable(tf.random.truncated_normal([1,1,num_channels], stddev=0.02), trainable=True)
        self.paras = tf.pad(paras, tf.constant([[0, 0,], [0, location_num], [0, 0]]), "CONSTANT")
    def call(self, inputs):
        return tf.pad(inputs, tf.constant([[0, 0,], [1, 0], [0, 0]]), "CONSTANT") + self.paras

# attention block, used in SAGAN and BigGAN
def attn_block(x, hidden_ratio, g_ratio, **kwargs):
    num_channels = x.get_shape().as_list()[-1]
    theta = tfa.layers.SpectralNormalization(layers.Conv1D(num_channels // hidden_ratio, 1, strides=1, padding='same', **kwargs))(x)
    phi = tfa.layers.SpectralNormalization(layers.Conv1D(num_channels // hidden_ratio, 1, strides=1, padding='same', **kwargs))(x)
    phi = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(phi)
    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn)
    g = tfa.layers.SpectralNormalization(layers.Conv1D(num_channels // g_ratio, 1, strides=1, padding='same', **kwargs))(x)
    g = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(g)
    attn_g = tf.matmul(attn, g)
    attn_g = tfa.layers.SpectralNormalization(layers.Conv1D(num_channels, 1, strides=1, padding='same', **kwargs))(attn_g)
    return ResidualLink()(x, attn_g)

# multi-head attention block, used in YLGAN and TransGAN, supports sparse attention
def multi_head_attn_block(x, hidden_ratio, g_ratio, nH, sparse, **kwargs):
    def sparse_mask(long, short, kind):
        '''
        mask for sparse attention,
        kind from LeftFloorMask, RightFloorMask, LeftRepetitiveMask and RightRepetitiveMask,
        used in Your Local GAN
        '''
        stride = int(np.sqrt(short))
        assert long % short == 0
        multiple = long//short
        if kind in ['LeftFloorMask', 'RightFloorMask']:
            indices = []
            for row in range(short):
                for col in range(row - (row % stride), row + 1):
                    indices.append([row, col])
            indices = np.array(indices)
            mask = np.zeros([short, short], dtype=np.bool)
            if kind == 'LeftFloorMask':
                mask[indices[:, 0], indices[:, 1]] = True
            else:
                mask[indices[:, 1], indices[:, 0]] = True

        if kind in ['LeftRepetitiveMask', 'RightRepetitiveMask']:
            if kind == 'RightRepetitiveMask':
                col_indices = np.arange(0,short,stride)
            else:
                col_indices = np.arange(stride - 1,short,stride)
            mask = np.eye(short, dtype=bool)
            for col in col_indices:
                mask[:,col] = True
        return np.vstack([mask]*multiple)

    def get_grid_masks(long, short):
        return np.array([sparse_mask(long,short,'RightFloorMask'),
        sparse_mask(long,short,'LeftFloorMask'),
        sparse_mask(long,short,'RightRepetitiveMask'),
        sparse_mask(long,short,'LeftRepetitiveMask')])

    _, location_num, num_channels = x.shape.as_list()
    assert num_channels % (hidden_ratio*nH) == 0
    downsampled_num = location_num // 2
    hidden_size = num_channels // hidden_ratio
    head_size = hidden_size // nH # size_per_head

    # theta path
    theta = tfa.layers.SpectralNormalization(layers.Conv1D(hidden_size, 1, strides=1, padding='same', **kwargs))(x)
    theta = tf.reshape(theta, [-1, location_num, nH, head_size])
    theta = tf.transpose(theta, [0, 2, 1, 3])

    # phi path
    phi = tfa.layers.SpectralNormalization(layers.Conv1D(hidden_size, 1, strides=1, padding='same', **kwargs))(x)
    phi = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(phi)
    phi = tf.reshape(phi, [-1, downsampled_num, nH, head_size])
    phi = tf.transpose(phi, [0, 2, 1, 3])

    attn = tf.matmul(theta, phi, transpose_b=True)
    if sparse:
        masks = tf.constant(get_grid_masks(location_num, downsampled_num)) # acquire masks
        attn = tf.keras.layers.Softmax()(attn, masks)
    else:
        attn = tf.keras.layers.Softmax()(attn)

    # g path
    g_hidden = num_channels // g_ratio
    g_head_size = g_hidden // nH
    g = tfa.layers.SpectralNormalization(layers.Conv1D(g_hidden, 1, strides=1, padding='same', **kwargs))(x)
    g = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(g)
    g = tf.reshape(g, [-1, downsampled_num, nH, g_head_size])
    # swap for heads
    g = tf.transpose(g, [0, 2, 1, 3])

    attn_g = tf.matmul(attn, g)
    # put heads to the end
    attn_g = tf.transpose(attn_g, [0, 2, 3, 1])
    attn_g = tf.reshape(attn_g, [-1, location_num, g_hidden])
    attn_g = tfa.layers.SpectralNormalization(layers.Conv1D(num_channels, 1, strides=1, padding='same', **kwargs))(attn_g)
    return ResidualLink()(x, attn_g)

def usample(x):
    x = layers.UpSampling1D(size=2)(x)
    return x

# ResNet generator block, used in SAGAN, etc.
def G_res_block(x, out_channels, g_kernel, **kwargs):
    x_0 = x
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = usample(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, g_kernel, strides=1, padding='same', **kwargs))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, g_kernel, strides=1, padding='same', **kwargs))(x)
    x_0 = usample(x_0)
    x_0 = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, 1, strides=1, padding='same', **kwargs))(x_0)
    return x_0 + x

# ResNet generator bottleneck structure
def G_res_bottleneck_block(x, out_channels, g_kernel, upsample, hidden_ratio, **kwargs):
    num_channels = x.get_shape().as_list()[-1]
    hidden_size = num_channels // hidden_ratio
    x_0 = x
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(hidden_size, 1, strides=1, padding='same', **kwargs))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if upsample:
        x = usample(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(hidden_size, g_kernel, strides=1, padding='same', **kwargs))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(hidden_size, g_kernel, strides=1, padding='same', **kwargs))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, 1, strides=1, padding='same', **kwargs))(x)
    if out_channels < num_channels:
        x_0 = x_0[:,:,:out_channels]
    if upsample:
        x_0 = usample(x_0)
    return x_0 + x

# ResNet generator output layer
def G_out_sn_block(x, out_channels, g_kernel, **kwargs):
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, g_kernel, strides=1, padding='same', **kwargs))(x)
    return x

def dsample(x):
    return layers.AveragePooling1D(pool_size=2, padding='valid')(x)

# ResNet discrinimtor start block
def D_res_start_block(x, out_channels, d_kernel, **kwargs):
    x_0 = x
    x = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, d_kernel, strides=1, padding='same', **kwargs))(x)
    x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, d_kernel, strides=2, padding='same', **kwargs))(x)
    # x = dsample(x)
    x_0 = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, 2, strides=2, padding='same', **kwargs))(x_0)
    # x_0 = dsample(x_0) ## change position
    return x + x_0

# ResNet discrinimtor block
def D_res_block(x, out_channels, d_kernel, **kwargs):
    x_0 = x
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, d_kernel, strides=1, padding='same', **kwargs))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, d_kernel, strides=2, padding='same', **kwargs))(x)
    # x = dsample(x)
    x_0 = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, 2, strides=2, padding='same', **kwargs))(x_0)
    # x_0 = dsample(x_0)
    return x_0 + x

# ResNet discrinimtor bottleneck block
def D_res_bottleneck_block(x, out_channels, d_kernel, downsample, hidden_ratio, **kwargs):
    num_channels = x.get_shape().as_list()[-1]
    hidden_size = num_channels // hidden_ratio
    strd = 2 if downsample else 1
    x_0 = x
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(hidden_size, 1, strides=1, padding='same', **kwargs))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(hidden_size, d_kernel, strides=1, padding='same', **kwargs))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(hidden_size, d_kernel, strides=strd, padding='same', **kwargs))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels, 1, strides=1, padding='same', **kwargs))(x)
    if downsample:
        x_0 = dsample(x_0)
    if out_channels > num_channels:
        x_c = tfa.layers.SpectralNormalization(layers.Conv1D(out_channels-num_channels, 1, strides=1, padding='same', **kwargs))(x_0)
        x_0 = tf.concat([x_0,x_c],axis=-1)
    return x_0 + x

# ResNet discrinimtor last block
def D_res_keep_block(x, d_kernel, **kwargs):
    input_channels = x.shape.as_list()[-1]
    x_0 = x
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(input_channels, d_kernel, strides=1, padding='same', **kwargs))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv1D(input_channels, d_kernel, strides=1, padding='same', **kwargs))(x)
    return x_0 + x

# G and D blocks without residual layers
def G_single_block(model, out_channels, g_kernel, act, **kwargs):
    model.add(layers.BatchNormalization())
    model.add(act())
    model.add(layers.Conv1DTranspose(out_channels, g_kernel, strides=2, padding='same', **kwargs))

def D_single_block(model, out_channels, d_kernel, **kwargs):
    model.add(layers.Conv1D(out_channels, d_kernel, strides=2, padding='same', use_bias=False, **kwargs))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

# regularizer used in BigGAN
def orthogonal_reg(w, beta=1e-4):
    w = tf.reshape(w,[-1, w.shape[-1]])
    if w.shape[0]<w.shape[1]:
        mat = tf.matmul(w,tf.transpose(w))*(1-tf.eye(w.shape[0]))
    else:
        mat = tf.matmul(tf.transpose(w),w)*(1-tf.eye(w.shape[1]))
    return beta * tf.reduce_sum(mat**2)