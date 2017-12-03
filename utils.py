# Utility

import numpy as np

from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from scipy.misc import imread, imsave, imresize
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
from PIL import Image

def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images


def get_train_images(paths, resize_len=512, crop_height=256, crop_width=256, flag = True):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    ny = 0
    nx = 0
    for path in paths:
        image = imread(path, mode='L')
        if image.shape == ():
            return image.shape
        height, width = image.shape
        ny = height
        nx = width
        # if height < width:
        #     new_height = resize_len
        #     new_width  = int(width * new_height / height)
        # else:
        #     new_width  = resize_len
        #     new_height = int(height * new_width / width)
        #
        # image = imresize(image, [new_height, new_width], interp='nearest')
        #
        # # crop the image
        # start_h = np.random.choice(new_height - crop_height + 1)
        # start_w = np.random.choice(new_width - crop_width + 1)
        # image = image[start_h:(start_h + crop_height), start_w:(start_w + crop_width), :]

        images.append(image)
    # images = tf.convert_to_tensor(images)
    # images = tf.reshape(images, [1, ny, nx, 1])
    # print('images shape final:', images.shape)
    if flag:
        images = np.stack(images, axis=0)
        images = np.stack((images, images, images), axis=-1)
    else:
        images = np.stack(images, axis=0)
        images = np.stack(images, axis=-1)
    # images = np.stack(images, axis=0)
    return images


def get_images(paths, height=None, width=None):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = imread(path, mode='RGB')

        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')

        images.append(image)

    images = np.stack(images, axis=0)
    print('images shape gen:', images.shape)
    return images


def save_images(paths, datas, save_path, prefix=None, suffix=None):
    if isinstance(paths, str):
        paths = [paths]

    assert(len(paths) == len(datas))

    if not exists(save_path):
        mkdir(save_path)

    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''

    for i, path in enumerate(paths):
        data = datas[i]
        # print('data ==>>\n', data)
        data = data.reshape([data.shape[0], data.shape[1]])
        # print('data reshape==>>\n', data)

        name, ext = splitext(path)
        name = name.split(sep)[-1]
        
        path = join(save_path, prefix + suffix + ext)
        print('data path==>>', path)


        # new_im = Image.fromarray(data)
        # new_im.show()

        imsave(path, data)

