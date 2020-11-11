import tensorflow as tf 
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Conv1D, BatchNormalization, \
    Activation, Input, UpSampling2D, MaxPooling2D, MaxPooling1D, SpatialDropout2D, Lambda
import numpy as np 
from tensorflow.keras import layers

def tomogan_disc(input_shape):
    inputs = Input(shape=input_shape)
    _tmp = Conv2D(filters=64, kernel_size=3, padding='same', \
                  activation='relu')(inputs)
    _tmp = Conv2D(filters=64, kernel_size=3, padding='same', strides=(2,2),\
                  activation='relu')(_tmp)
    
    _tmp = Conv2D(filters=128, kernel_size=3, padding='same', \
                  activation='relu')(_tmp)
    _tmp = Conv2D(filters=128, kernel_size=3, padding='same', strides=(2,2),\
                  activation='relu')(_tmp)
    
    _tmp = Conv2D(filters=128, kernel_size=3, padding='same', \
                  activation='relu')(_tmp)
    _tmp = Conv2D(filters=4, kernel_size=3, padding='same', strides=(2,2),\
                  activation='relu')(_tmp)

    _tmp = layers.Flatten()(_tmp)
    _tmp = layers.Dense(units=64, activation='relu')(_tmp)
    _tmp = layers.Dense(units=1)(_tmp)
    
    return tf.keras.models.Model(inputs, _tmp)

def unet_conv_block(inputs, nch):
    _tmp = Conv2D(filters=nch, kernel_size=3, padding='same', activation='relu')(inputs)
    _tmp = Conv2D(filters=nch, kernel_size=3, padding='same', activation='relu')(_tmp)
    return _tmp

def unet(input_shape, use_cnnt=False, nlayers=3):
    inputs = Input(shape=input_shape)
    ly_outs= [inputs, ]
    label2idx = {'input': 0}
    
    _tmp = Conv2D(filters=8, kernel_size=1, padding='valid', activation='relu')(ly_outs[-1])
    ly_outs.append(_tmp)
#     label2idx['ch_stack'] = len(ly_outs)-1

    _tmp = unet_conv_block(ly_outs[-1], 32)
    ly_outs.append(_tmp)
    label2idx['box1_out'] = len(ly_outs)-1
    for ly in range(2, nlayers+1):
        _tmp = MaxPooling2D(pool_size=2, strides=2, padding='same')(ly_outs[-1])        
        _tmp = unet_conv_block(_tmp, 2*ly_outs[-1].shape[-1].value)
        ly_outs.append(_tmp)
        label2idx['box%d_out' % (ly)] = len(ly_outs)-1
        
    # intermediate layers
    _tmp = MaxPooling2D(pool_size=2, strides=2, padding='same')(ly_outs[-1])
    _tmp = Conv2D(filters=ly_outs[-1].shape[-1].value, kernel_size=3, \
                  padding='same', activation='relu')(_tmp)
    ly_outs.append(_tmp)
    
    for ly, nch in zip(range(1, nlayers+1), (64, 32, 32)):
        if use_cnnt:
            _tmp = Conv2DTranspose(filters=ly_outs[-1].shape[-1].value, activation='relu', \
                                   kernel_size=4, strides=(2, 2), padding='same')(ly_outs[-1]) 
        else: 
            _tmp = UpSampling2D(size=(2, 2), interpolation='bilinear')(ly_outs[-1]) 
        _tmp = tf.keras.layers.concatenate([ly_outs[label2idx['box%d_out' % (nlayers-ly+1)]], _tmp])
        _tmp = unet_conv_block(_tmp, nch)
        ly_outs.append(_tmp)
    
    _tmp = Conv2D(filters=16, kernel_size=1, padding='valid', 
                  activation='relu')(ly_outs[-1])

    _tmp = Conv2D(filters=1, kernel_size=1, padding='valid', \
                  activation=None)(_tmp)
    
    return tf.keras.models.Model(inputs, _tmp)

