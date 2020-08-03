import  tensorflow  as tf

from    .                       import inits
from    tensorflow.keras.layers import Layer

def conv(filters, kernel_size, strides = 1, padding = None,
         data_format = "channels_last", activation = "relu"):
    return tf.keras.layers.Conv2D \
    (
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        data_format = data_format,
        activation = activation,
        use_bias = True,
        kernel_initializer = inits.tnormal(),
        bias_initializer = inits.constant()
    );

def maxpool():
    return tf.keras.layers.MaxPool2D \
    (
        pool_size = 2,
        strides = 2,
        padding = "valid",
        data_format = "channels_last"
    );

def flatten():
    return tf.keras.layers.Flatten \
    (
        data_format = "channels_last"
    );


def dense(units, activation = "relu"):
    return tf.keras.layers.Dense \
    (
        units = units,
        activation = activation,
        use_bias = True,
        kernel_initializer = inits.tnormal(),
        bias_initializer = inits.constant()
    );

def upsample(size = 2, interpolation = "bilinear"):
    return tf.keras.layers.UpSampling2D \
    (
        size = size,
        data_format = "channels_last",
        interpolation = interpolation
    );



class Block(Layer):
    """
        Combines multiple layers into a single layer block.
    """
    def __init__(self, layer_list, **kwargs):
        super(Block, self).__init__(**kwargs);

        self.layer_list = layer_list;

    def call(self, inputs):
        outputs = inputs;
        for layer in self.layer_list:
            outputs = layer(outputs);
        return outputs;

    def compute_output_shape(self, input_shape):
        output_shape = input_shape;
        for layer in self.layer_list:
            output_shape = layer.compute_output_shape(output_shape);
        return output_shape;

class ClipValue(Layer):
    """
        Clips a tensor by value within a minimum and max range for stability.
    """
    def __init__(self, min = 0.01, max = 0.99, **kwargs):
        super(ClipValue, self).__init__(**kwargs);

        self.min = min;
        self.max = max;

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max);

    def compute_output_shape(self, input_shape):
        return input_shape;

def clipvalue(min = 0.01, max = 0.99):
    return ClipValue(min, max);
