import tensorflow as tf
import numpy as np


def L1_Max(source_en_a, source_en_b, epsilon=1e-5):
    result = []
    narry_a = source_en_a
    narry_b = source_en_b
    meanA, varA = tf.nn.moments(source_en_a, [1, 2], keep_dims=True)
    meanB, varB = tf.nn.moments(source_en_b, [1, 2], keep_dims=True)
    sigmaA = tf.sqrt(tf.add(varA, epsilon))
    sigmaB = tf.sqrt(tf.add(varB, epsilon))

    dimension = source_en_a.get_shape().as_list()
    for i in range(dimension[3]):
        temp_a = sigmaA[0, 0, 0, i]
        temp_b = sigmaB[0, 0, 0, i]
        temp = temp_a - temp_b
        if tf.minimum(temp,0) == 0:
            result.append(narry_a[0, :, :, i])
        else:
            result.append(narry_b[0, :, :, i])

    resule_tf = tf.convert_to_tensor(result)
    resule_tf = tf.reshape(resule_tf, [dimension[0], dimension[1], dimension[2], dimension[3]])
    return resule_tf


