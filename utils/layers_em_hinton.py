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
        
        self.pose_kernel = self.add_weight(shape=(self.kernel_size,
                                                  self.kernel_size,
                                                  self.in_capsules,
                                                  self.out_capsules,
                                                  self.sqrt_atom,
                                                  self.sqrt_atom),
                                           name = "pose_kernel",)
        self.built = True

    def call(self, inputs):
        poses, activations = inputs

        votes_out = self.conv_caps(poses, activations=False)
        act_out = self.conv_caps(activations, activations=True)

        return votes_out, act_out
        
    
    def conv_caps(self, input, activations=True):
        """The poses and activations undergo essentially the same operations
        so to prevent duplicated code we just use call() as a wrapper to pass
        them individually through this method which does the actual work.

        activations arg is used to specificy wether activations are passed
        (True) or poses (False). Determines which kernel to use:)
        """
        # Reshape into a 4D tensor: (N, H , W, caps_in*in_atoms)
        # This has to be done because extract_patches() only takes 4D tensors
        _shape = tf.shape(input)

        poses_reshaped = tf.reshape(input, (_shape[0], _shape[1], _shape[2], 
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
                                  _shape[-3], _shape[-2], _shape[-1]))
        
        # Hinton now transposes and reshapes the patches for optimal performance
        # TF2 fixes this behind the scenes, so we keep it in a shape that makes
        # more sense to me:)

        # Each patch must be mulitplied by the kernel
        # The kernel should matmul each pose matrix with a unique transformation matrix
        # Kernel should have the same shape as 1 patch, but with additional channels
        # for each output capsule (defined in build() method

        if activations:
            # We are working on the activations - they are not multiplied
            # by the kernel
            return pose_patches
        else:
            # We are working on poses
            kernel = self.pose_kernel

        # Looks terrible, but does the matrix multiplication between kernel and patches
        # Cannot repeat indices, so I use xy for kernel (instead of more intuitive kk)
        # same for pose matrix (mn instead of pp)
        #   b=batch, h=height, w=width, xy=kernel*kernel, i=in_capsules,  
        #   mn=pose_matrix (4*4)
        matrices = tf.einsum('bhwxyimn,xyiomn->bhwxyomn', pose_patches, kernel)
        print("Done")

        # TODO: rename variables, maybe seperate the patches in a different 
        # method and apply the final part only to the poses for clarity
        
        return matrices


class ClassCaps(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClassCaps, self).__init__(**kwargs)


class EMRouting(tf.keras.layers.Layer):
    def __init__(self, iterations=3, min_var=0.0005, final_beta=1.0, **kwargs):
        super(EMRouting, self).__init__(**kwargs)
        self.iterations = iterations
        self.min_var = min_var
        self.final_beta = final_beta


    def build(self, input_shape):
        self.pose_shape_in = input_shape[0]
        self.act_shape_in = input_shape[1]

        # Initialize biases (using Hinton's setting)
        self.activation_bias = self.add_weight(
            shape=(),  # TODO: fill in shape
            initializer=tf.constant_initializer(0.5),
            name='activation_bias'
        )
        self.sigma_bias = self.add_weight(
            shape=(),  # TODO
            initializer=tf.constant_initializer(0.5),
            name='sigma_bias'
        ) 

        # Initialize empty tensors as input for 1st iter of the routing loop
        self.out_activations = tf.zeros()
        self.out_poses = tf.zeros
        self.out_mass = tf.zeros()

        # Called post in Hinton's implementation
        self.R_ij_init = tf.nn.softmax(tf.zeros(self.act_shape_in))


    def call(self, inputs):
        votes, activations = inputs

        # Ill use math notation from the paper for the variables
        R_ij = tf.nn.softmax(tf.zero(tf.shape(inputs)))

        for i in range(self.iterations):
            routing_iteration()  # TODO: add output and input to this

    def routing_iteration(self, i, posterior, activation, center, masses):
        """ The main loop of the EM routing algorithm

        i is the current iteration
        """
        # Soft warm-up for beta. Note that final_beta is never reached, it 
        # is more like a scaling factor
        beta = self.final_beta * (1 - tf.pow(0.95, (i+1)))
        
        # M-step, E-step
        pass

    def m_step(self, R_ij, a_i, V_ij, capsules_in):
        # In the paper, j is capsule in Omega_{L}, that is capsules_in

        # vote_conf in Hinton's implementation
        R_ij = R_ij * a_i  

        # V_ij are the votes, shaped:    [batch, height, width, kernel, kernel, 
        #                                 capsules, sqrt_atom, sqrt_atom]
        # R_ij is shaped like activatons:[batch, height, width, kernel, kernel,
        #                                capsules, 1, 1]
        # Each of the 16 pose values must be multuplied by the same R_ij
        # Since we were smart about the shapes they broadcast nicely

        # preactivate_unrolled in Hinton's implementation
        mu = (tf.reduce_sum(R_ij * V_ij, axis=[3,4], keepdims=True) 
              / tf.reduce_sum(R_ij, axis=[3,4], keepdims=True)
              + 0.0000001)  # This prevents numerical instability
        
        # TODO: RECHECK THIS! Waarschijnlijk nog niet helemaal de goede axis
        # Ook checken of die conv layer uberhaput wel iets met de activations
        # hoort te doen! Niet helemaal duidelijk atm. 
        

    def e_step(self):
        pass



