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
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=5e-2),
            bias_initializer=tf.keras.initializers.Constant(0.1)
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
        self.sqrt_atoms = np.sqrt(out_atoms)

    def build(self, input_shape):
        self.conv_dim = input_shape[-1]  # = A
        self.conv = tf.keras.layers.Conv2D(
            filters=self.num_capsules * (self.out_atoms+1),  # 1 for the activation
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(.0000002),
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
        #   with channels size: num_capsules*(1+num_atoms)

        # Split the pre-activation from the pose. So now there are two tensors 
        # with channel sizes 1 and num_atoms, respectively
        poses, pre_activations = tf.split(conv_output,
                                          [self.out_atoms*self.num_capsules,
                                           self.num_capsules],
                                           axis=-1)
        
        # Instead of a big out_atoms*num_capsule dimension, we reshape it into 
        # a tensor of shape (num_capsules, sqrt, sqrt) so each capsule has its 
        # own pose matrix:
        # Example (N, W, H, 512) -> (N, W, H, 32, 4, 4)
        poses_shape = tf.shape(poses)
        N, H, W = poses_shape[0], poses_shape[1], poses_shape[2] 
        poses_reshaped = tf.reshape(poses, [N, H, W, 
                                            self.num_capsules, 
                                            self.sqrt_atoms,
                                            self.sqrt_atoms])
        
        # Make sure the activation shape matches for broadcasting:
        #   (N, W, H, C) -> (N, W, H, C, 1, 1)
        pre_act_reshaped = tf.expand_dims(pre_activations, -1)
        pre_act_reshaped = tf.expand_dims(pre_act_reshaped, -1)

        # Compute activations by passing through sigmoid:
        activations_reshaped = tf.math.sigmoid(pre_act_reshaped)

        return poses_reshaped, activations_reshaped


class ConvCaps(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, num_capsules=32, stride=1, **kwargs):
        """Kernel is shaped (kernel_size, kernel_size)"""
        super(ConvCaps, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.out_capsules = num_capsules  # number of output capsules
        self.stride = stride
        self.conv = None

    def build(self, input_shape):
        # input shape = (N, H, W, capsules_in, sqrt_atom, sqrt_atom)
        #   Example: (N, H, W, 32, 4, 4) in case of 32 capsules with 4x4 matrix
        self.pose_shape_in = input_shape[0]
        self.act_shape_in = input_shape[1]

        self.in_capsules = self.pose_shape_in[3]
        self.sqrt_atom = self.pose_shape_in[-1]

        shape = (self.kernel_size, self.kernel_size, self.in_capsules,
                 self.out_capsules, self.sqrt_atom, self.sqrt_atom)
        
        self.kernel = self.add_weight(shape=(self.kernel_size,
                                             self.kernel_size,
                                             self.in_capsules,
                                             self.out_capsules,
                                             self.sqrt_atom,
                                             self.sqrt_atom),
                                      name = "kernel",)
        self.built = True

    def call(self, inputs):
        poses, activations = inputs

        # Reshape into a 4D tensor: (N, H , W, caps_in*in_atoms)
        # This has to be done becausee extract_patches() only takes 4D tensors
        _shape = tf.shape(poses)

        poses_reshaped = tf.reshape(poses, (_shape[0], _shape[1], _shape[2], 
                                            _shape[3]*_shape[4]*_shape[5]))
        # Taking patches is similar to applying convolution, We cannot use 
        # Conv2D (for example) because it does not do matrix multiplication
        pose_patches = tf.image.extract_patches(poses_reshaped, 
                                                 [1, self.kernel_size, self.kernel_size, 1],
                                                 [1, self.stride, self.stride, 1],
                                                 [1, 1, 1, 1],
                                                 'VALID')
        # 

        # Output shape based on 'valid' padding !
        out_height = (_shape[1] - self.kernel_size) // self.stride + 1
        out_width = (_shape[2] - self.kernel_size) // self.stride + 1

        # Reshape the patches back into pose matrices
        pose_patches = tf.reshape(pose_patches,
                                  (_shape[0], out_height, out_width,  # N, H, W
                                  self.kernel_size, self.kernel_size,
                                  self.in_capsules, self.sqrt_atom, self.sqrt_atom))
        
        # Hinton now transposes and reshapes the patches for optimal performance
        # TF2 fixes this behind the scenes, so we keep it in a shape that makes
        # more sense to me:)

        # Each patch must be mulitplied by the kernel
        # The kernel should matmul each pose matrix with a unique transformation matrix
        # Kernel should have the same shape as 1 patch, but with additional channels
        # for each output capsule (defined in build() method)

        # Looks terrible, but does the matrix multiplication between kernel and patches
        # Cannot repeat indices, so I use xy for kernel (instead of more intuitive kk)
        # same for pose matrix (mn instead of pp)
        #   b=batch, h=height, w=width, xy=kernel*kernel, i=in_capsules,  
        #   mn=pose_matrix (4*4)

        matrices = tf.einsum('bhwxyimn,xyiomn->bhwxyomn', pose_patches, self.kernel)
        print("Done")
        
        return matrices
        


class ClassCaps(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClassCaps, self).__init__(**kwargs)


class EMRouting(tf.keras.layers.Layer):
    def __init__(self, iterations=3, **kwargs):
        super(EMRouting, self).__init__(**kwargs)
        self.iterations = iterations

    def build(self, input_shape):
        pass

    def call(self, inputs):
        pass
