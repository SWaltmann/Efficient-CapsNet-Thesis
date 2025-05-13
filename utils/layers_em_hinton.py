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
        
    
    def conv_caps(self, _input, activations=True):
        """The poses and activations undergo essentially the same operations
        so to prevent duplicated code we just use call() as a wrapper to pass
        them individually through this method which does the actual work.

        activations arg is used to specificy wether activations are passed
        (True) or poses (False). Determines which kernel to use:)
        """
        # Reshape into a 4D tensor: (N, H , W, caps_in*in_atoms)
        # This has to be done because extract_patches() only takes 4D tensors
        _shape = tf.shape(_input)

        input_reshaped = tf.reshape(_input, (_shape[0], _shape[1], _shape[2], 
                                            _shape[3]*_shape[4]*_shape[5]))
        # Taking patches is similar to applying convolution, We cannot use 
        # Conv2D (for example) because it does not do matrix multiplication
        patches = tf.image.extract_patches(input_reshaped, 
                                                 [1, self.kernel_size, self.kernel_size, 1],
                                                 [1, self.stride, self.stride, 1],
                                                 [1, 1, 1, 1],
                                                 'VALID')

        # Output shape based on 'valid' padding !
        out_height = (_shape[1] - self.kernel_size) // self.stride + 1
        out_width = (_shape[2] - self.kernel_size) // self.stride + 1

        # Reshape the patches back into pose matrices
        patches = tf.reshape(patches,
                                  (_shape[0], out_height, out_width,  # N, H, W
                                  self.kernel_size, self.kernel_size,
                                  _shape[-3], _shape[-2], _shape[-1]))
        
        # Hinton now transposes and reshapes the patches for optimal performance
        # TF2 fixes this behind the scenes, so we keep it in a shape that makes
        # more sense to me:)

        if activations:
            # We are working on the activations - they are not multiplied
            # by the kernel
            return tf.expand_dims(patches, -3)  # For consistency with the poses 
     
        # Each patch must be mulitplied by the kernel
        # The kernel should matmul each pose matrix with a unique transformation matrix
        # Kernel should have the same shape as 1 patch, but with additional channels
        # for each output capsule (defined in build() method

        # Looks terrible, but does the matrix multiplication between kernel and patches
        # Cannot repeat indices, so I use xy for kernel (instead of more intuitive kk)
        # same for pose matrix (mn instead of pp)
        #   b=batch, h=height, w=width, xy=kernel*kernel, i=in_capsules, o=out_capsules  
        #   mn=pose_matrix (4*4)
        matrices = tf.einsum('bhwxyimn,xyiomn->bhwxyiomn', patches, self.pose_kernel)
        
        return matrices


class ClassCaps(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClassCaps, self).__init__(**kwargs)


class EMRouting(tf.keras.layers.Layer):
    """ Please note that although this is implemented as a seperate layer, it
    should be viewed as part of the previous layer. ConvCaps handles only the
    pose matrices and selects the corresponding activations without altering
    them. This layer takes the activations and poses and finds the new 
    activations.
    """
    def __init__(self, iterations=3, min_var=0.0005, final_beta=1.0, **kwargs):
        super(EMRouting, self).__init__(**kwargs)
        self.iterations = iterations
        self.min_var = min_var
        self.final_lambda = final_beta


    def build(self, input_shape):
        # Pose input:
        # Shape: [batch, height, width, kernel, kernel,
        #         in_caps, out_caps, sqrt_atom, sqrt_atom] 
        # Intuition: For every batch, at every grid position (height x width), 
        #   there is a kernel (kernel x kernel) consisting of in_caps number of 
        #   capsules. Each of those capsules votes for each of the out_caps number 
        #   of output capsules, and each vote is in the form of a 
        #   pose matrix (sqrt_atom x sqrt_atom)

        # Activation input:
        # Shape: [batch, height, width, kernel, kernel
        #         in_caps, 1, 1, 1]
        # activations have a similar shape, but lack the out_caps dimension, since 
        # only the lower-level capsules have an activation at this point (we are
        # calculating the higher level activations in this layer). There also is 
        # only 1 value per lower-level capsule, we keep singleton dimensions for
        # easier broadcasting later:)

        self.pose_shape_in = input_shape[0]
        self.act_shape_in = input_shape[1]

        p_shape = self.pose_shape_in

        # Initialize biases (using Hinton's setting)
        # activation_bias in Hinton's implementation
        self.beta_a = self.add_weight(
            shape=(1, 1, 1, 1, p_shape[5], 1, 1, 1, 1),  # Each higher-level capsule has its own activation cost
            initializer=tf.constant_initializer(0.5),
            name='activation_bias'
        )
        # sigma_bias in Hinton's implementation
        self.beta_u = self.add_weight(
            shape=(1, 1, 1, 1, p_shape[5], 1, 1, 1, 1),  # Each higher-level capsule has its own activation cost
            initializer=tf.constant_initializer(0.5),
            name='sigma_bias'
        ) 

        
    def call(self, inputs):
        votes, activations = inputs

        # Initialize empty tensors as input for 1st iter of the routing loop
        self.out_activations = tf.zeros(tf.shape(activations))
        self.out_poses = tf.zeros(tf.shape(votes))
        # post in Hinton's implementation
        self.R_ij = tf.nn.softmax(tf.zeros(tf.shape(activations)))


        # Perform routing
        for i in range(self.iterations - 1):
            self.m_step(activations, votes, i)
            self.e_step()

        # Last routing iteration only requires the m-step
        self.m_step(activations, votes, self.iterations)

        return self.out_poses, self.out_activations



    def m_step(self, a_i, V_ij, i):
        # Hinton names the variables differently in his code than 
        # in the paper. We stick to the paper, but this is the translation:
        #   my var - Hinton's var
        #   R_ij - posterior
        #   a_i - activation
        #   V_ij - wx

        #   the 'masses' arg in Hintons code is unnecessary (it gets redefined)

        # I know that _j, _j, _h are just indices to a specific element. I kept
        # them in the variable names so that it is easier to match to the pseudo 
        # code in the paper.

        R_ij = self.R_ij

        # vote_conf in Hinton's implementation
        R_ij = R_ij * a_i  

        # We need Sum_i(R_ij) multiple times so we'll store it:
        # masses in Hinton's implementation
        sum_R_ij = tf.reduce_sum(R_ij, axis=[3,4,5], keepdims=True)

        # V_ij are the votes, shaped:    [batch, height, width, kernel, kernel, 
        #                                 caps_in, caps_out, sqrt_atom, sqrt_atom]
        # R_ij is shaped like activatons:[batch, height, width, kernel, kernel,
        #                                 caps_in, 1, 1, 1]

        # Each of the 16 pose values must be multiplied by the same R_ij
        # Since we were smart about the shapes they broadcast nicely

        # The summation should be done over ALL lower-level capsules. For
        # each higher-level capsule there are caps_in lower-level votes
        # for each position in the kernel. So we must sum over dimensions
        # kernel, kernel AND caps_in

        # It should result in a mu for each value of the pose matrix, ie shape:
        #   [batch, heigth, width, 1, 1, 1, caps_out, sqrt_atom, sqrt_atom]
        # This is kind of a weighed average of the votes - but it is averaged
        # in a weird way (averaging each value seperately instead of treating 
        # the pose matrix as 3D thing). Will not result in a valid pose mat.

        # So now mu_jh[:, :, :, :, :, : j, h1, h2] is mu_j^h from the paper
        # BTW, the paper treats the pose matrix as a vector with length 16,
        # which is why there is only 1 index for the value in the matrix 

        # preactivate_unrolled in Hinton's implementation. Hinton then
        # combines this with the old value to get 'center'. But we skip
        # that step since it is not mentioned in the paper.
        mu_jh = (tf.reduce_sum(R_ij * V_ij, axis=[3,4,5], keepdims=True) 
                / sum_R_ij + 0.0000001)  # e-7 from Hinton's implementation
                # e-7 prevents numerical instability (Gritzman, 2019)
        
        # variance in Hinton's implementation
        sigma_jh_sq = (tf.reduce_sum(R_ij * tf.pow((V_ij - mu_jh), 2), axis=[3,4,5], keepdims=True)
                       / sum_R_ij) + self.min_var 

        # This happens in the paper, but not in Hinton's code
        sigma_jh = tf.math.sqrt(sigma_jh_sq)

        # Completely lost how Hinton's code relates to their paper at this point
        # Good luck figuring that out    
        cost_h = (self.beta_u + tf.math.log(sigma_jh)) * sum_R_ij

        # beta in Hinton's implementation
        inverse_temp = self.final_lambda*(1-tf.pow(0.95, i+1))

        # activation_update in Hinton's implementation (I THINK, shit's a maze imo)
        # Maybe logit is actually closer but yout guess is as good as mine
        a_j = tf.math.sigmoid(
            inverse_temp*(self.beta_a - tf.reduce_sum(cost_h, axis=[-1,-2], keepdims=True))  # Sum over values in pose matrix
              )

        # Assign everythin to the corresponding attributes 
        # Could have done that immediately but wanted to follow paper's notation
        self.out_activations = a_j
        self.out_poses = mu_jh
        self.R_ij = R_ij

    def e_step(self):
        pass
