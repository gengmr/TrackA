import tensorflow as tf
import tensorflow.contrib.slim as slim


"""
Creates a convolutional residual block
as defined in the paper. More on
this inside model.py
x: input to pass through the residual block
channels: number of channels to compute
stride: convolution stride
"""
def resBlock(x, channels, kernel_size=[3,3], scale=1):
    tmp = slim.conv2d(x,6*channels,[1,1],activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp,channels,[1,1],activation_fn=None)
    tmp = slim.conv2d(tmp,channels,[3,3],activation_fn=None)
    tmp = scale * tmp
    return x + tmp


"""
Tensorflow log base 10.
Found here: https://github.com/tensorflow/tensorflow/issues/1666
"""
def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

from scipy import misc

def EDSR(input_image):
    with tf.variable_scope("generator"):
        input_image = input_image - tf.reshape([114.45, 111.47, 103.03], [1, 1, 1, 3])/255
        x_main = slim.conv2d(input_image, 3, [3, 3], stride=1)
        
        x_0 = slim.conv2d(input_image, 8, [3, 3], stride=1)
        x_1 = slim.conv2d(x_0, 16, [3, 3], stride=2)
        x_2 = slim.conv2d(x_1, 32, [3, 3], stride=2)
        for i in range(6):
            x_2 = resBlock(x_2, 32, [3, 3])
        W0 = weight_variable([4, 4, 16, 32], name='W0')
        up0 = tf.nn.conv2d_transpose(x_2, W0, output_shape=tf.shape(x_1), strides=[1,2,2,1])
        W1 = weight_variable([3, 3, 8, 16], name='W1')
        up1 = tf.nn.conv2d_transpose(up0, W1, output_shape=tf.shape(x_0), strides=[1, 2, 2, 1])
        W2 = weight_variable([3, 3, 8, 3], name='W2')
        x_res = tf.nn.conv2d(up1, W2, strides=[1, 1, 1, 1], padding='SAME')

        output = 0.4 * x_res +  0.1 * x_main + 0.5 * input_image
        output = output + tf.reshape([114.45, 111.47, 103.03], [1, 1, 1, 3])/255
    return output


if __name__=="__main__":

    img=tf.Variable(tf.random_normal([1,100,100,3]))

    enhanced = EDSR(img)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        output = sess.run(enhanced)
        print(sess.run([tf.shape(img),tf.shape(enhanced)]))
