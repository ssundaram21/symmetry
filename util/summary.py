import tensorflow as tf

def variable_summaries(weights, bias, opt):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if opt.extense_summary:
        mean = tf.reduce_mean(weights)
        tf.summary.scalar('mean_weights', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(weights - mean)))
        tf.summary.scalar('stddev_weights', stddev)
        tf.summary.scalar('max_weights', tf.reduce_max(weights))
        tf.summary.scalar('min_weights', tf.reduce_min(weights))
        tf.summary.histogram('stddev_weights', weights)

        mean_bias = tf.reduce_mean(bias)
        tf.summary.scalar('mean_bias', mean_bias)
        stddev_bias = tf.sqrt(tf.reduce_mean(tf.square(bias - mean_bias)))
        tf.summary.scalar('stddev_bias', stddev_bias)
        tf.summary.scalar('max_bias', tf.reduce_max(bias))
        tf.summary.scalar('min_bias', tf.reduce_min(bias))
        tf.summary.histogram('stddev_bias', bias)


def activation_summaries(responses, opt):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if opt.extense_summary:
        tf.summary.histogram('activations', responses)
        a = tf.norm(tf.cast(tf.less(responses, -0.9*tf.ones_like(responses)), tf.float32))
        b = tf.norm(tf.cast(tf.greater(responses, 0.9*tf.ones_like(responses)), tf.float32))
        c = tf.norm(tf.cast(tf.less(responses, 0.05*tf.ones_like(responses)), tf.float32))
        d = tf.norm(tf.cast(tf.greater(responses, 0.95*tf.ones_like(responses)), tf.float32))
        s = tf.cast(tf.size(responses),tf.float32)
        tf.summary.scalar('saturation_relu', tf.constant(opt.hyper.batch_size, tf.float32)*c/s)
        tf.summary.scalar('saturation_sigmoid', tf.constant(opt.hyper.batch_size, tf.float32)*(c+d)/s)
        tf.summary.scalar('saturation_tanh', tf.constant(opt.hyper.batch_size, tf.float32)*(a+b)/s)


def gradient_summaries(grad, var, opt):

    if opt.extense_summary:
        tf.summary.scalar(var.name + '/gradient_mean', tf.norm(grad))
        tf.summary.scalar(var.name + '/gradient_max', tf.reduce_max(grad))
        tf.summary.scalar(var.name + '/gradient_min', tf.reduce_min(grad))
        tf.summary.histogram(var.name + '/gradient', grad)


