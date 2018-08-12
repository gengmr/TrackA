import tensorflow as tf
import tensorflow.contrib.slim as slim



def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels):

    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [4, 4, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

def unet(input):
    with tf.variable_scope("generator"):
        input1 = slim.conv2d(input, 3, [3, 3], stride=2, rate=1, activation_fn=None, scope='downsample')
        input2 = slim.conv2d(input, 3, [1, 1], stride=1, rate=1, activation_fn=lrelu, scope='in_conv')
        conv1=slim.conv2d(input1,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1=slim.conv2d(conv1,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
        pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

        conv2=slim.conv2d(pool1,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv2=slim.conv2d(conv2,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

        conv3=slim.conv2d(pool2,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        conv3=slim.conv2d(conv3,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
        pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

        conv4=slim.conv2d(pool3,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
        conv4=slim.conv2d(conv4,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
        pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

        conv5=slim.conv2d(pool4,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
        conv5=slim.conv2d(conv5,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_2')

        up6 =  upsample_and_concat( conv5, conv4, 16, 16  )
        conv6=slim.conv2d(up6,  16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
        conv6=slim.conv2d(conv6,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')

        up7 =  upsample_and_concat( conv6, conv3, 16, 16  )
        conv7=slim.conv2d(up7,  16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7=slim.conv2d(conv7,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')

        up8 =  upsample_and_concat( conv7, conv2, 16, 16 )
        conv8=slim.conv2d(up8,  16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        conv8=slim.conv2d(conv8,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')

        up9 =  upsample_and_concat( conv8, conv1, 16, 16 )
        conv9=slim.conv2d(up9,  16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
        conv9=slim.conv2d(conv9,3,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')

        conv10 = upsample_and_concat(conv9, input2, 3, 3)
        out = slim.conv2d(conv10, 3, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv10')

    return out