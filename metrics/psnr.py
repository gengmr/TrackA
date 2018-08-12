import tensorflow as tf

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def PSNR(enhanced, dslr_, PATCH_SIZE, batch_size):
    enhanced_flat = tf.reshape(enhanced, [-1, PATCH_SIZE])
    loss_mse = tf.reduce_sum(tf.pow(dslr_ - enhanced_flat, 2)) / (PATCH_SIZE * batch_size)
    loss_psnr = 20 * log10(1.0 / tf.sqrt(loss_mse))
    return loss_psnr