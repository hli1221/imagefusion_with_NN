# Style Transfer Network
# Encoder -> AdaIN -> Decoder

import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from adaptive_instance_norm import AdaIN
from l1norm_max_choose import L1_Max

class StyleTransferNet(object):

    def __init__(self, encoder_weights_path, model_pre_path):
        self.encoder = Encoder(encoder_weights_path)
        self.decoder = Decoder(model_pre_path)

    def transform(self, content, style):
        # # switch RGB to BGR
        # content = tf.reverse(content, axis=[-1])
        # style   = tf.reverse(style,   axis=[-1])
        #
        # preprocess image
        content = self.encoder.removemean(content)
        style   = self.encoder.removemean(style)
        # content = self.encoder.preprocess(content)
        # style = self.encoder.preprocess(style)

        # encode image
        enc_c, enc_c_layers = self.encoder.encode(content)
        enc_s, enc_s_layers = self.encoder.encode(style)

        self.encoded_content_layers = enc_c_layers
        self.encoded_style_layers   = enc_s_layers

        # pass the encoded images to AdaIN, change to l1 norm + max-choose
        target_features = L1_Max(enc_c, enc_s)
        # target_features = enc_c
        self.target_features = target_features

        print('target_features:', target_features.shape)

        # decode target features back to image
        generated_img = self.decoder.decode(target_features)

        # deprocess image
        generated_img = self.encoder.addmean(generated_img)
        # generated_img = self.encoder.deprocess(generated_img)

        # # switch BGR back to RGB
        # generated_img = tf.reverse(generated_img, axis=[-1])

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        return generated_img

    def encoder_process(self, content, style):
        # preprocess image
        content = self.encoder.removemean(content)
        style = self.encoder.removemean(style)
        # content = self.encoder.preprocess(content)
        # style = self.encoder.preprocess(style)

        # encode image
        enc_c, enc_c_layers = self.encoder.encode(content)
        enc_s, enc_s_layers = self.encoder.encode(style)

        self.encoded_content_layers = enc_c_layers
        self.encoded_style_layers = enc_s_layers

        return enc_c, enc_s

    def decoder_process(self, features):
        generated_img = self.decoder.decode(features)

        # deprocess image
        generated_img = self.encoder.addmean(generated_img)
        # generated_img = self.encoder.deprocess(generated_img)

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        return generated_img

    def transform_recons(self, content):

        content = self.encoder.removemean(content)

        # encode image
        enc_c, enc_c_layers = self.encoder.encode(content)

        self.encoded_content_layers = enc_c_layers

        target_features = enc_c
        self.target_features = target_features

        print('target_features_recons:', target_features.shape)

        # decode target features back to image
        generated_img = self.decoder.decode(target_features)

        # deprocess image
        generated_img = self.encoder.addmean(generated_img)
        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        return generated_img

