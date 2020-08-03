import  tensorflow  as tf

from    tensorflow.keras.initializers   import TruncatedNormal

def tnormal(mean = 0.0, stddev = 0.01, seed = None):
    return TruncatedNormal(mean = mean, stddev = stddev, seed = seed);

def constant(value = 0.1):
    return tf.constant_initializer(value = value);
