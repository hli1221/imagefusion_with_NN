# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from TF_PCA import TF_PCA
from style_transfer_net import StyleTransferNet
from utils import get_train_images
import SSIM

STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

TRAINING_IMAGE_SHAPE = (256, 256, 3) # (height, width, color_channels)
TRAINING_IMAGE_SHAPE_OR = (256, 256, 1) # (height, width, color_channels)

EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
EPSILON = 1e-5

def train_recons(original_imgs_path, encoder_path, save_path,  model_pre_path, debug=False, logging_period=100):
    if debug:
        from datetime import datetime
        start_time = datetime.now()

    num_imgs = len(original_imgs_path)
    # num_imgs = 10000
    original_imgs_path = original_imgs_path[:num_imgs]
    mod = num_imgs % BATCH_SIZE

    print('Train images number %d.\n' % num_imgs)
    print('Train images samples %s.\n' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]

    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    HEIGHT_OR, WIDTH_OR, CHANNELS_OR = TRAINING_IMAGE_SHAPE_OR
    INPUT_SHAPE_OR = (BATCH_SIZE, HEIGHT_OR, WIDTH_OR, CHANNELS_OR)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        original = tf.placeholder(tf.float32, shape=INPUT_SHAPE_OR, name='original')
        source = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='source_a')

        print('source:', source.shape)

        # create the style transfer net
        stn = StyleTransferNet(encoder_path, model_pre_path)

        # pass content and style to the stn, getting the generated_img, fused image
        generated_img = stn.transform_recons(source)

        # # get the target feature maps which is the output of AdaIN
        # target_features = stn.target_features

        pixel_loss = tf.reduce_sum(tf.reduce_mean(tf.square(original - generated_img), axis=[1, 2]))
        pixel_loss = pixel_loss/(HEIGHT*WIDTH)

        # pixel_loss = pixel_loss*10
        # compute the SSIM loss
        ssim_loss = 1 - SSIM.tf_ssim(original, generated_img)

        # compute the total loss
        loss = pixel_loss + ssim_loss

        # Training step
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # ** Start Training **
        step = 0
        count_loss = 0
        n_batches = int(len(original_imgs_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')
            start_time = datetime.now()

        Loss_all = [i for i in range(EPOCHS * n_batches)]
        for epoch in range(EPOCHS):

            np.random.shuffle(original_imgs_path)

            for batch in range(n_batches):
                # retrive a batch of content and style images

                original_path = original_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]

                original_batch = get_train_images(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                source_batch = get_train_images(original_path, crop_height=HEIGHT, crop_width=WIDTH)

                original_batch = original_batch.reshape([BATCH_SIZE, 256, 256, 1])

                # run the training step
                sess.run(train_op, feed_dict={original: original_batch, source: source_batch})
                step += 1
                # if step % 1000 == 0:
                #     saver.save(sess, save_path, global_step=step)
                if debug:
                    is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                    if is_last_step or step % logging_period == 0:
                        elapsed_time = datetime.now() - start_time
                        _pixel_loss, _ssim_loss, _loss = sess.run([pixel_loss, ssim_loss, loss],
                            feed_dict={original: original_batch, source: source_batch})
                        Loss_all[count_loss] = _loss
                        count_loss += 1
                        print('step: %d,  total loss: %.3f,  elapsed time: %s' % (step, _loss, elapsed_time))
                        print('pixel loss: %.3f' % (_pixel_loss))
                        print('ssim loss : %.3f\n' % (_ssim_loss))

        # ** Done Training & Save the model **
        saver.save(sess, save_path)

        iter_index = [i for i in range(count_loss)]
        plt.plot(iter_index, Loss_all[:count_loss])
        plt.show()

        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Model is saved to: %s' % save_path)

