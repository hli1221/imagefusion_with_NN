# Use a trained Image Transform Net to generate
# a style transferred image with a specific style

import tensorflow as tf

from style_transfer_net import StyleTransferNet
from utils import get_images, save_images, get_train_images
from l1norm_max_choose import L1_Max

def generate(content_path, style_path, encoder_path, model_path, model_pre_path, output_path=None):

    outputs = _handler(content_path, style_path, encoder_path, model_path, model_pre_path, output_path=output_path)
    return list(outputs)


def _handler(content_path, style_path, encoder_path, model_path, model_pre_path, output_path=None):

    with tf.Graph().as_default(), tf.Session() as sess:
        index = 2
        content_path = content_path + str(index) + '.jpg'
        style_path = style_path + str(index) + '.jpg'

        content_img = get_train_images(content_path)
        style_img = get_train_images(style_path)

        # build the dataflow graph
        content = tf.placeholder(
            tf.float32, shape=content_img.shape, name='content')
        style = tf.placeholder(
            tf.float32, shape=style_img.shape, name='style')

        stn = StyleTransferNet(encoder_path, model_pre_path)

        enc_c, enc_s = stn.encoder_process(content, style)

        target = tf.placeholder(
            tf.float32, shape=enc_c.shape, name='target')

        # output_image = stn.transform(content, style)
        output_image = stn.decoder_process(target)

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        # get the output
        enc_c, enc_s = sess.run([enc_c, enc_s],
            feed_dict={content: content_img, style: style_img})
        feature = L1_Max(enc_c, enc_s)
        # feature = enc_s
        output = sess.run(output_image, feed_dict={target: feature})

    if output_path is not None:
        save_images(content_path, output, output_path,
                    prefix='fused' + str(index) + '_', suffix='deep')

    return output