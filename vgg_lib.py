import tensorflow as tf
import numpy as np

import vgg_parameter
#from vgg_cifar10_aliyun import vgg_parameter



def create_weight(shape):
    weight = tf.Variable(tf.truncated_normal(shape,  stddev=0.1))
    return weight

def create_bias(shape):
    bias = tf.Variable(tf.constant(0.1, shape = shape))
    return bias

def conv(input, weight, bias, stride = [1,1,1,1], padding_value = "VALID", activation_function = None, norm = None):
    conv_result = tf.nn.conv2d(input, weight, strides=stride, padding=padding_value)
    result = tf.nn.bias_add(conv_result, bias)
    if norm is not None:
        with tf.Session() as sess:
            conv_mean, conv_var = tf.nn.moments(result, axes=[0])
            scale = tf.Variable(tf.ones(tf.shape(bias).eval()))
            shift = tf.Variable(tf.zeros(tf.shape(bias).eval()))
            epsilon = 0.001
            result = tf.nn.batch_normalization(result, conv_mean, conv_var, shift, scale, epsilon)
    if activation_function is not None:
        result = activation_function(result)

    return result
def pooling(input, ksize = [1,2,2,1], stride = [1,2,2,1], padding_value = "VALID"):
    result = tf.nn.max_pool(input, ksize=ksize, strides=stride, padding=padding_value)
    return result


def fc_layer(input, weight, bias, if_drop_out = None, activation_function = None, norm = None):
    result = tf.matmul(input, weight) + bias
    if norm is not None:
        with tf.Session() as sess:
            fc_mean, fc_var = tf.nn.moments(result, axes=[0])
            scale = tf.Variable(tf.ones(tf.shape(bias).eval()))
            shift = tf.Variable(tf.zeros(tf.shape(bias).eval()))
            epsilon = 0.001
            result = tf.nn.batch_normalization(result, fc_mean, fc_var, shift, scale, epsilon)
    if activation_function is not None:
        result = activation_function(result)
    if if_drop_out is not None:
        return tf.nn.dropout(result, 0.8)

    return result

def vgg16(input):
    #conv_1
    conv1_weight = create_weight(vgg_parameter.conv1_weight_size)
    conv1_bias = create_bias(vgg_parameter.conv1_bias_size)
    conv1_output = conv(input, conv1_weight, conv1_bias, vgg_parameter.conv1_stride_size,vgg_parameter.conv1_padding_value,activation_function=tf.nn.relu)
    #conv_2

    conv2_weight = create_weight(vgg_parameter.conv2_weight_size)
    conv2_bias = create_bias(vgg_parameter.conv2_bias_size)
    conv2_output = conv(conv1_output, conv2_weight, conv2_bias, vgg_parameter.conv2_stride_size,vgg_parameter.conv2_padding_value,tf.nn.relu)

    #pooling_3
    pooling3_output = pooling(conv2_output, ksize=vgg_parameter.pooling3_k_size, stride=vgg_parameter.pooling3_stride_size,padding_value=vgg_parameter.pooling3_padding_value)

    norm1 = tf.nn.lrn(pooling3_output,4,  bias=1.0, alpha=0.001/9.0, beta = 0.75 )
    #conv_4
    conv4_weight = create_weight(vgg_parameter.conv4_weight_size)
    conv4_bias = create_bias(vgg_parameter.conv4_bias_size)
    conv4_output = conv(norm1, conv4_weight, conv4_bias, vgg_parameter.conv4_stride_size,vgg_parameter.conv4_padding_value,tf.nn.relu)
    #conv_5

    conv5_weight = create_weight(vgg_parameter.conv5_weight_size)
    conv5_bias = create_bias(vgg_parameter.conv5_bias_size)
    conv5_output = conv(conv4_output, conv5_weight, conv5_bias, vgg_parameter.conv5_stride_size,vgg_parameter.conv5_padding_value,tf.nn.relu )

    #pooling_6
    pooling6_output = pooling(conv5_output, ksize=vgg_parameter.pooling6_k_size, stride=vgg_parameter.pooling6_stride_size,padding_value=vgg_parameter.pooling6_padding_value)
    norm2 = tf.nn.lrn(pooling6_output, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # conv_7
    conv7_weight = create_weight(vgg_parameter.conv7_weight_size)
    conv7_bias = create_bias(vgg_parameter.conv7_bias_size)
    conv7_output = conv(norm2, conv7_weight, conv7_bias, vgg_parameter.conv7_stride_size, vgg_parameter.conv7_padding_value,tf.nn.relu)
    # conv_8
    conv8_weight = create_weight(vgg_parameter.conv8_weight_size)
    conv8_bias = create_bias(vgg_parameter.conv8_bias_size)
    conv8_output = conv(conv7_output, conv8_weight, conv8_bias, vgg_parameter.conv8_stride_size, vgg_parameter.conv8_padding_value,tf.nn.relu)
    # conv_9
    conv9_weight = create_weight(vgg_parameter.conv9_weight_size)
    conv9_bias = create_bias(vgg_parameter.conv9_bias_size)
    conv9_output = conv(conv8_output, conv9_weight, conv9_bias, vgg_parameter.conv9_stride_size, vgg_parameter.conv9_padding_value,tf.nn.relu)
    # pooling_10
    pooling10_output = pooling(conv9_output, ksize=vgg_parameter.pooling10_k_size, stride=vgg_parameter.pooling10_stride_size,
                             padding_value=vgg_parameter.pooling10_padding_value)
    norm3 = tf.nn.lrn(pooling10_output, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    #
    # conv_15
    norm_reshape = tf.reshape(norm3, [vgg_parameter.BATCH_SIZE, -1])
    out_dim = norm_reshape.get_shape()[1].value

    # fc_16
    fc_16_weight = create_weight([out_dim, 384])
    fc_16_bias = create_bias([384])
    fc_16_output = fc_layer(norm_reshape, fc_16_weight, fc_16_bias, activation_function=tf.nn.relu)
    # fc_17

    fc_17_weight = create_weight(vgg_parameter.fc_17_weight_size)
    fc_17_bias = create_bias(vgg_parameter.fc_17_bias_size)
    fc_17_output = fc_layer(fc_16_output, fc_17_weight, fc_17_bias,  activation_function=tf.nn.relu)

    # fc_18
    fc_18_weight = create_weight(vgg_parameter.fc_18_weight_size)
    fc_18_bias = create_bias(vgg_parameter.fc_18_bias_size)
    fc_18_output = fc_layer(fc_17_output, fc_18_weight, fc_18_bias, norm=None)

    return fc_18_output









if __name__ == "__main__":
    print("test")