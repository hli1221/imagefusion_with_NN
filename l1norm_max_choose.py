import tensorflow as tf
import numpy as np


def L1_Max(source_en_a, source_en_b, epsilon=1e-5):
    result = []
    narry_a = source_en_a
    narry_b = source_en_b
    # meanA, varA = tf.nn.moments(source_en_a, [1, 2], keep_dims=True)
    # meanB, varB = tf.nn.moments(source_en_b, [1, 2], keep_dims=True)
    # sigmaA = tf.sqrt(tf.add(varA, epsilon))
    # sigmaB = tf.sqrt(tf.add(varB, epsilon))
    dimension = source_en_a.shape

    # caculate L1-norm
    temp_abs_a = tf.abs(narry_a)
    temp_abs_b = tf.abs(narry_b)
    l1_a = tf.reduce_sum(temp_abs_a,3)
    l1_b = tf.reduce_sum(temp_abs_b,3)

    # caculate the map for source images
    mask_value = l1_a - l1_b
    mask_sign = tf.sign(mask_value)

    mask_sign_a = (mask_sign + 1)/2
    mask_sign_b = (tf.abs(mask_sign - 1))/2

    array_MASK_a = mask_sign_a.eval()
    array_MASK_b = mask_sign_b.eval()

    for i in range(dimension[3]):
        temp_matrix = array_MASK_a*narry_a[0,:,:,i] + array_MASK_b*narry_b[0,:,:,i]
        result.append(temp_matrix)
    result = np.stack(result, axis=-1)

    count_a = tf.reduce_sum(mask_sign_a)
    count_b = tf.reduce_sum(mask_sign_b)
    print("count_a ====>>>", count_a.eval())
    print("count_b ====>>>", count_b.eval())

    print("result ====>>>", result.shape)
    resule_tf = result
    # resule_tf = tf.convert_to_tensor(result)
    # print("resule_tf shape====>>", resule_tf.get_shape())
    #
    # resule_tf = tf.reshape(resule_tf, [dimension[0], dimension[1], dimension[2], dimension[3]])
    return resule_tf


