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
            kernel_regularizer=tf.keras.regularizers.l2(.05),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=5e-2)
        )
        self.built = True

    def call(self, inputs):
        return self.conv(inputs)


class PrimaryCaps(tf.keras.layers.Layer):
    """
    All 'magic numbers' are either from 'Matrix Capsules with EM Routing', or
    taken from the corresponding repo at google-research's github:
    https://github.com/google-research/google-research/tree/master/capsule_em
    """
    def __init__(self, B=32, out_atoms=16, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.num_capsules = B  # number of output capsule
        self.out_atoms = out_atoms  # number of values in pose matrix (16 or 9)
        self.conv = None
        self.kernel_size=1
        self.stride=1

    def build(self, input_shape):
        self.conv_dim = input_shape[-1]  # = A
        self.conv = tf.keras.layers.Conv2D(
            filters=self.num_capsules * (self.out_atoms+1),  # 1 for the activation
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='same',
            use_bias=False
        )
        # Manually construct the kernel out of two tensors with different 
        # initialization parameters
        self.conv.build(input_shape)  # build layer so that we can override kernel

        # Create the kernels
        pose_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.5)
        pose_kernel = pose_initializer(shape=[self.kernel_size, 
                                              self.kernel_size, 
                                              input_shape[-1], 
                                              self.out_atoms * self.num_capsules])

        activation_initializer = tf.keras.initializers.TruncatedNormal(stddev=3.0)
        act_kernel = activation_initializer(shape=[self.kernel_size, 
                                                   self.kernel_size, 
                                                   input_shape[-1], 
                                                   1 * self.num_capsules])
        
        kernel = tf.concat([pose_kernel, act_kernel], axis=-1)
        self.conv.kernel.assign(kernel)

        self.built = True

    def call(self, inputs):
        # Input shape: (batch_size, H, W, channels)
        conv_output = self.conv(inputs)
        # cont_output.shape = (batch_size, H, W, channels), 
        #   with channels = num_capsules*(1+num_atoms)

        # Split the pre-activation from the pose. So now there are two tensors 
        # with channel sizes 1 and num_atoms, respectively
        poses, pre_activations = tf.split(conv_output,
                                          [self.out_atoms*self.num_capsules,
                                           self.num_capsules],
                                           axis=-1)
        
        # Instead of a big out_atoms*num_capsule dimension, we reshape it into 
        # a tensor of shape (num_capsules, out_atoms) so each capsule has its 
        # own pose matrix:
        poses_shape = tf.shape(poses)
        N, H, W = poses_shape[0], poses_shape[1], poses_shape[2] 
        poses_reshaped = tf.reshape(poses, [N, H, W, 
                                            self.num_capsules, self.out_atoms])
        
        # Make sure the activation shape matches for broadcasting:
        pre_act_reshaped = tf.expand_dims(pre_activations, -1)
        # Compute activations by passing through sigmoid:
        activations_reshaped = tf.math.sigmoid(pre_act_reshaped)

        return poses_reshaped, activations_reshaped


class ConvCaps(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConvCaps, self).__init__(**kwargs)

class ClassCaps(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClassCaps, self).__init__(**kwargs)

