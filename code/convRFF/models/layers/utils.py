import tensorflow as tf


def trapz(y, x):
    dx = x[1:] - x[:-1]
    heights = (y[..., :-1] + y[..., 1:]) / 2.0
    return tf.reduce_sum(dx * heights, axis=-1)
