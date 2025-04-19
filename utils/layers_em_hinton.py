import tensorflow as tf
import numpy as np


class ReLUConv(tf.keras.layers.Layer):
    def __init__(self, A=32, kernel_size=5, stride=2, **kwargs):
        super(ReLUConv, self).__init__(**kwargs)

        # Settings
        self.num_channels = A  # num_channels is more descriptive than 'A'
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = None
    
    def build(self, input_shape):
        # in_channels = input_shape[-1]  # Check for grayscale/color
        self.conv = tf.keras.layers.Conv2D(
            filters=self.num_channels, 
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='valid',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(.05)
        )
        self.built = True

    def call(self, inputs):
        return self.conv(inputs)


class PrimaryCaps(tf.keras.layers.Layer):
    def __init__(self, B=32, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)


class ConvCaps(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConvCaps, self).__init__(**kwargs)

class ClassCaps(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClassCaps, self).__init__(**kwargs)

