# Specific for real data
BUFFER_SIZE = 5996 # lenght of timeseries
BATCH_SIZE = 32
data_dim = 32
noise_dim = 100
data_channel = 10

import tensorflow as tf
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
ls_loss = lambda a,b : tf.reduce_mean(tf.nn.l2_loss(a - b))