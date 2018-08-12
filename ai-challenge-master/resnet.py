import tensorflow as tf
import tensorflow.contrib.slim as slim



def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels):

    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

def deconv_filter(output_channels, in_channels):
    deconv_filter = tf.Variable(tf.truncated_normal([2, 2, output_channels, in_channels], stddev=0.02))
    return deconv_filter

def conv2d_transpose(input, out_channels, in_channels, out_shape, strides, size):
    y = tf.nn.conv2d_transpose(input, deconv_filter(out_channels, in_channels), out_shape, strides=strides)
    x = y
    x.set_shape([None, size, size, out_channels])
    return x

def enet(input):
    with tf.variable_scope("generator"):
        # ----------------initial_block----------------------------------------------
        conv_init = slim.conv2d(input, 13, [3, 3], stride=2, activation_fn=lrelu, scope='conv_init')
        maxpool_init = slim.max_pool2d(input, [2, 2], padding='SAME')
        initial_out = tf.concat([conv_init, maxpool_init], 3)

        # ----------------bottleneck_down_1------------------------------------------------
        conv_bottleneck_down_1_1 = slim.conv2d(initial_out, 16, [2, 2], stride=2,
                                         activation_fn=lrelu, scope='conv_bottleneck_down_1_1', padding='SAME')
        conv_bottleneck_down_1_2 = slim.conv2d(conv_bottleneck_down_1_1, 16, [3, 3], stride=1,
                                         activation_fn=lrelu, scope='conv_bottleneck_down_1_2', padding='SAME')
        conv_bottleneck_down_1_3 = slim.conv2d(conv_bottleneck_down_1_2, 16, [1, 1], stride=1,
                                         activation_fn=lrelu, scope='conv_bottleneck_down_1_3', padding='SAME')
        maxpool_1 = slim.max_pool2d(initial_out, [2, 2], padding='SAME')
        bottleneck_down_1 = maxpool_1 + conv_bottleneck_down_1_3

        # ----------------bottleneck_down_2------------------------------------------------
        conv_bottleneck_down_2_1 = slim.conv2d(bottleneck_down_1, 16, [2, 2], stride=2,
                                         activation_fn=lrelu, scope='conv_bottleneck_down_2_1', padding='SAME')
        conv_bottleneck_down_2_2 = slim.conv2d(conv_bottleneck_down_2_1, 16, [3, 3], stride=1,
                                         activation_fn=lrelu, scope='conv_bottleneck_down_2_2', padding='SAME')
        conv_bottleneck_down_2_3 = slim.conv2d(conv_bottleneck_down_2_2, 16, [1, 1], stride=1,
                                         activation_fn=lrelu, scope='conv_bottleneck_down_2_3', padding='SAME')
        maxpool_2 = slim.max_pool2d(bottleneck_down_1, [2, 2], padding='SAME')
        bottleneck_down_2 = maxpool_2 + conv_bottleneck_down_2_3

        # ----------------bottleneck_up_1------------------------------------------------
        conv_bottleneck_up_1_1 = slim.conv2d(bottleneck_down_2, 16, [1, 1], stride=1,
                                         activation_fn=lrelu, scope='conv_bottleneck_up_1_1', padding='SAME')
        conv_bottleneck_up_1_2 = conv2d_transpose(conv_bottleneck_up_1_1, 16, 16,
                                                  tf.shape(bottleneck_down_1), strides=[1, 2, 2, 1], size=25)
        conv_bottleneck_up_1_3 = slim.conv2d(conv_bottleneck_up_1_2, 16, [1, 1], stride=1,
                                             activation_fn=lrelu, scope='conv_bottleneck_up_1_3', padding='SAME')
        # ----------------bottleneck_up_2------------------------------------------------
        conv_bottleneck_up_2_1 = slim.conv2d(conv_bottleneck_up_1_3, 16, [1, 1], stride=1,
                                         activation_fn=lrelu, scope='conv_bottleneck_up_2_1', padding='SAME')
        conv_bottleneck_up_2_2 = conv2d_transpose(conv_bottleneck_up_2_1, 16, 16,
                                                  tf.shape(initial_out), strides=[1, 2, 2, 1], size=50)
        conv_bottleneck_up_2_3 = slim.conv2d(conv_bottleneck_up_2_2, 16, [1, 1], stride=1,
                                             activation_fn=lrelu, scope='conv_bottleneck_up_2_3', padding='SAME')

        # ----------------bottleneck_up_3------------------------------------------------
        conv_bottleneck_up_3_1 = slim.conv2d(conv_bottleneck_up_2_3, 16, [1, 1], stride=1,
                                         activation_fn=lrelu, scope='conv_bottleneck_up_3_1', padding='SAME')
        conv_bottleneck_up_3_2 = conv2d_transpose(conv_bottleneck_up_3_1, 3, 16,
                                                  tf.shape(input), strides=[1, 2, 2, 1], size=100)
        conv_bottleneck_up_3_3 = slim.conv2d(conv_bottleneck_up_3_2, 3, [1, 1], stride=1,
                                             activation_fn=lrelu, scope='conv_bottleneck_up_3_3', padding='SAME')

        out = conv_bottleneck_up_3_3

        return out



def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift





