# Demo - train the style transfer network & use it to generate an image

from __future__ import print_function

import time

from train import train
from generate import generate
from utils import list_images


# IS_TRAINING = True
IS_TRAINING = False

ENCODER_WEIGHTS_PATH = './models/vgg_weights/vgg19_normalised.npz'

STYLE_WEIGHTS = [1]

MODEL_SAVE_PATHS = [
    'models/image_fusion_model_w1_30k.ckpt',
]

def main():

    if IS_TRAINING:

        original_imgs_path_name = 'D:/Image_Database/ImageFusion_database/MS_COCO_G/original/'
        sourceA_imgs_path  = list_images('D:/Image_Database/ImageFusion_database/MS_COCO_G/source_a')
        sourceB_imgs_path_name  = 'D:/Image_Database/ImageFusion_database/MS_COCO_G/source_b/'

        for ssim_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\nBegin to train the network with the style weight: %.2f ...\n' % ssim_weight)

            train(ssim_weight, original_imgs_path_name,sourceA_imgs_path, sourceB_imgs_path_name, ENCODER_WEIGHTS_PATH, model_save_path, debug=True)

            print('\nSuccessfully! Done training...\n')
    else:

        style_name = 'infrared2'

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\nBegin to generate pictures with the style weight: %.2f ...\n' % style_weight)

            contents_path = list_images('images/content')
            style_path    = 'images/style/' + style_name + '.jpg'
            output_save_path = 'outputs'

            generated_images = generate(contents_path, style_path, ENCODER_WEIGHTS_PATH, model_save_path, is_same_size=True,
                output_path=output_save_path, prefix=style_name + '-', suffix='-' + str(style_weight))

            print('\ntype(generated_images):', type(generated_images))
            print('\nlen(generated_images):', len(generated_images), '\n')


if __name__ == '__main__':
    main()

