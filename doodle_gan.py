import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
#plt.imshow(np.reshape(data[2], [28, 28]))
#plt.show()


def generator(z, reuse=None):                                                # generator
    with tf.variable_scope('gen', reuse=reuse):
        hidden1 = tf.layers.dense(inputs=z, units=128, activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=128, activation=tf.nn.leaky_relu)

        output = tf.layers.dense(inputs=hidden2, units=784, activation=tf.nn.tanh)
        return output


def discriminator(x, reuse=None):                               # discriminator
    with tf.variable_scope('dis', reuse=reuse):
        hidden1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=128, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(inputs=hidden2, units=1)
        output = tf.sigmoid(logits)

        return output, logits


def calc_loss(logits, labels):                          # loss function
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


data = np.load('./doodle_data/aircraft.npy')            # load the data

real_images = tf.placeholder(tf.float32, shape=[None, 784])
z = tf.placeholder(tf.float32, shape=[None, 100])

G = generator(z)

D_output_real, D_logits_real = discriminator(real_images)                   # trained with real images
D_output_fake, D_logits_fake = discriminator(G, reuse=True)                 # to detect fake ones


D_real_loss = calc_loss(D_logits_real, tf.ones_like(D_logits_real) * 0.9)   # Calculate loss
D_fake_loss = calc_loss(D_logits_fake, tf.zeros_like(D_logits_real))
D_loss = D_real_loss + D_fake_loss
G_loss = calc_loss(D_logits_fake, tf.ones_like(D_logits_fake))


learning_rate = 0.001
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)     # Optimizers - discriminator
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)     # Optimizers - generator

batch_size = 100
epochs = 500
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=g_vars)

samples = []                                # Save a sample for each epoch

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)          # House keeping stuff

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:                  # Training
    sess.run(init)

    for epoch in range(epochs):                                         # Each epoch is an entire run through data set
        num_of_batches = len(data) / batch_size
        start = 0
        end = start + batch_size
        for i in range(int(num_of_batches)):
            batch_images = data[start:end]
            batch_fake = np.random.uniform(-1, 1, size=(batch_size, 100))           # random noise data for generator

            _ = sess.run(D_trainer, feed_dict={real_images: batch_images, z: batch_fake})
            _ = sess.run(G_trainer, feed_dict={z: batch_fake})

            start = start + batch_size      # next batch
            end = start + batch_size

        print("Current Epoch ", epoch+1)
        sample_input = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = sess.run(generator(z, reuse=True), feed_dict={z: sample_input})

        samples.append(gen_sample)

    saver.save(sess, './models/500_epoch_model.ckpt')


step = 50
for j in range(int(len(samples)/step)):

    plt.imshow(np.reshape(samples[j * step], [28, 28]))
    plt.show()


