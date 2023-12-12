## Define the training loop
# The training loop begins with generator receiving a random seed as input.
# That seed is used to produce an image. The discriminator is then used to classify real
# images (drawn from the training set) and fakes images (produced by the generator).
# The loss is calculated for each of these models, and the gradients are used to update
# the generator and discriminator.

from config import *
from generator import generator_loss, generator_optimizer_fun
from discriminator import discriminator_loss, discriminator_optimizer_fun
from visualization import plot_loss

import os
import time
import numpy as np
import tensorflow as tf
from IPython import display

#choices = ['dcgan', 'lsgan', 'sagan', 'wgan', 'wgan_gp', 'dragan', 'ragan', 'ralsgan', 'ylgan', 'biggan', 'biggan_deep', 'transgan']

@tf.function
def train_dcgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "dcgan")

@tf.function
def train_lsgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "lsgan")

@tf.function
def train_sagan(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "sagan")

@tf.function
def train_wgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "wgan")

@tf.function
def train_wgan_gp(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "wgan_gp")

@tf.function
def train_dragan(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "dragan")

@tf.function
def train_ragan(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "ragan")

@tf.function
def train_lsgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "lsgan")

@tf.function
def train_ralsgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "ralsgan")

@tf.function
def train_ylgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "ylgan")

@tf.function
def train_biggan(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "biggan")

@tf.function
def train_biggan_deep(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "biggan_deep")

@tf.function
def train_transgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    return train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, "transgan")


def my_train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, choice):
    if choice == "dcgan":
        return train_dcgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images)
    elif choice == "lsgan":
        return train_lsgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images)
    elif choice == "sagan":
        return train_sagan(generator, discriminator, generator_optimizer, discriminator_optimizer, images)
    elif choice == "wgan":
        return train_wgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images)
    elif choice == "wgan_gp":
        return train_wgan_gp(generator, discriminator, generator_optimizer, discriminator_optimizer, images)
    elif choice == "dragan":
        return train_dragan(generator, discriminator, generator_optimizer, discriminator_optimizer, images)
    elif choice == "ragan":
        return train_ragan(generator, discriminator, generator_optimizer, discriminator_optimizer, images)
    elif choice == "ralsgan":
        return train_ralsgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images)
    elif choice == "ylgan":
        return train_ylgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images)
    elif choice == "biggan":
        return train_biggan(generator, discriminator, generator_optimizer, discriminator_optimizer, images)
    elif choice == "biggan_deep":
        return train_biggan_deep(generator, discriminator, generator_optimizer, discriminator_optimizer, images)
    elif choice == "transgan":
        return train_transgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
#@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, choice):
    noise = tf.random.normal([images.shape[0], noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        if choice in ['ragan','ralsgan']:
            gen_loss = generator_loss(fake_output, choice, real_output=real_output)
        else:
            gen_loss = generator_loss(fake_output, choice)
        if choice in ['wgan_gp', 'dragan']:
            disc_loss = discriminator_loss(discriminator, real_output, fake_output, choice, generated_images = generated_images, images= images)
        else:
            disc_loss = discriminator_loss(discriminator, real_output, fake_output, choice)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# train the network on the data
def train(generator, discriminator, G_loss, D_loss, dataset, test, epochs, choice):
    generator_optimizer = generator_optimizer_fun(choice)
    discriminator_optimizer = discriminator_optimizer_fun(choice)

    start = time.time()
    for epoch in range(epochs):

        G_list = []; D_list = []

        for image_batch in dataset:
            gen_loss, disc_loss = my_train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, image_batch, choice)
            G_list.append(gen_loss)
            D_list.append(disc_loss)

        G_loss.append(np.mean(G_list))
        D_loss.append(np.mean(D_list))

        # Produce images for the GIF as we go
        if epoch % 10 == 0:
            display.clear_output(wait=True)
            plot_loss(G_loss, D_loss, choice, epoch, epochs)
        #for fun, seed in test:
        #    fun(generator, epoch + 1, seed) # plot test results

        # Save the model every 15 epochs
        # if (epoch + 1) % 15 == 0:
        #   checkpoint.save(file_prefix = checkpoint_prefix) ## temp

        if epoch % 10 == 0:
            print ('Time for last 10 epochs (epoch {} now) is {} sec'.format(epoch + 1, time.time()-start))
            start = time.time()
            if choice not in os.listdir("models"):
                os.mkdir(os.path.join("models", choice))
            generator.save(os.path.join("models", choice, f"{choice}_{epoch}.keras"))

    # # Generate after the final epoch
    # display.clear_output(wait=True)
    # generate_and_save_images(generator, epochs, seed) ##temp