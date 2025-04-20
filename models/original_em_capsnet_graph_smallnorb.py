
import numpy as np
import tensorflow as tf
from utils.layers_em_hinton import ReLUConv, PrimaryCaps, ConvCaps, ClassCaps

def em_capsnet_graph(input_shape):
    """ Architecture of EM CapsNet, as described in: 'Matrix Capsules with EM Routing '

    Each layer is named after what their output represents
    """
    inputs = tf.keras.Input(input_shape)
    # Other models in this repo need the y_true tensor for the reconstruction 
    # regularizer. We do not use y_true, but accept the input so that we can 
    # use the same Dataset object for training/testing
    y_true = tf.keras.layers.Input(shape=(5,))

    relu_conv1 = ReLUConv(A=32)(inputs)
    prim_caps1 = PrimaryCaps()(relu_conv1)
    # conv_caps1 = ConvCaps()(prim_caps1)
    # conv_caps2 = ConvCaps()(conv_caps1)

    # capsules = ClassCaps()(conv_caps2)  

    return tf.keras.Model(inputs=[inputs, y_true],outputs=prim_caps1, name='EM_CapsNet')

if __name__ == '__main__':
    # For testing I will just run it from this
    pass
