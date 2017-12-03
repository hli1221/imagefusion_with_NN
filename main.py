# Demo - train the style transfer network & use it to generate an image

from __future__ import print_function

import time

from train import train
from train_recons import train_recons
from generate import generate
from utils import list_images


IS_TRAINING = True
# IS_TRAINING = False

ENCODER_WEIGHTS_PATH = './models/vgg_weights/vgg19_normalised.npz'
MODEL_SAVE_PATH = './models/reconstructure_decode_base_style/recons_decode_model_w1e0_1epoches.ckpt'
model_pre_path  = './models/reconstructure_decode_base_style/recons_decode_model_w1e0_10k.ckpt'
# model_pre_path  = './models/style_weight_1e0_pre0/style_weight_1e0.ckpt'

def main():

    if IS_TRAINING:

        # original_imgs_path_name = 'D:/ImageDatabase/Image_fusion_MSCOCO/original/'
        # sourceA_imgs_path  = list_images('D:/ImageDatabase/Image_fusion_MSCOCO/source_a')
        # sourceB_imgs_path_name  = 'D:/ImageDatabase/Image_fusion_MSCOCO/source_b/'

        original_imgs_path = list_images('D:/ImageDatabase/Image_fusion_MSCOCO/original/')

        print('\nBegin to train the network ...\n')

        # train(ssim_weight, original_imgs_path_name,sourceA_imgs_path, sourceB_imgs_path_name, ENCODER_WEIGHTS_PATH, model_save_path,model_pre_path, debug=True)
        train_recons(original_imgs_path, ENCODER_WEIGHTS_PATH, MODEL_SAVE_PATH, model_pre_path, debug=True)

        print('\nSuccessfully! Done training...\n')
    else:

        sourceA_name = 'visible'
        sourceB_name = 'infrared'
        print('\nBegin to generate pictures ...\n')

        content_path = 'images/IV/' + sourceA_name
        style_path   = 'images/IV/' + sourceB_name

        output_save_path = 'outputs'

        generated_images = generate(content_path, style_path, ENCODER_WEIGHTS_PATH, MODEL_SAVE_PATH, model_pre_path,
            output_path=output_save_path)

        # print('\ntype(generated_images):', type(generated_images))
        # print('\nlen(generated_images):', len(generated_images), '\n')


if __name__ == '__main__':
    main()

