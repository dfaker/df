from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Dropout, Add,Concatenate, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.initializers import RandomNormal
from keras.optimizers import Adam

from pixel_shuffler import PixelShuffler

import tensorflow as tf

from keras_contrib.losses import DSSIMObjective
from keras import losses
import time
from keras.utils import multi_gpu_model


class penalized_loss(object):

  def __init__(self,mask,lossFunc,maskProp= 1.0):
    self.mask = mask
    self.lossFunc=lossFunc
    self.maskProp = maskProp
    self.maskaskinvProp = 1-maskProp  

  def __call__(self,y_true, y_pred):

    tro, tgo, tbo = tf.split(y_true,3, 3 )
    pro, pgo, pbo = tf.split(y_pred,3, 3 )

    tr = tro
    tg = tgo
    tb = tbo

    pr = pro
    pg = pgo
    pb = pbo
    m  = self.mask 

    m   = m*self.maskProp
    m  += self.maskaskinvProp
    tr *= m
    tg *= m
    tb *= m

    pr *= m
    pg *= m
    pb *= m


    y = tf.concat([tr, tg, tb],3)
    p = tf.concat([pr, pg, pb],3)

    #yo = tf.stack([tro,tgo,tbo],3)
    #po = tf.stack([pro,pgo,pbo],3)

    return self.lossFunc(y,p)


optimizer = Adam( lr=5e-5, beta_1=0.5, beta_2=0.999 )


IMAGE_SHAPE = (64,64,3)

ENCODER_DIM = 1024
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k

def upscale_ps(filters, use_norm=True):
    def block(x):
        x = Conv2D(filters*4, kernel_size=3, use_bias=False, kernel_initializer=RandomNormal(0, 0.02), padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

def res_block(input_tensor, f):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = Add()([x, input_tensor])
    x = LeakyReLU(alpha=0.2)(x)
    return x

def conv( filters ):
    def block(x):
        x = Conv2D( filters, kernel_size=5, strides=2, padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        return x
    return block

def upscale( filters ):
    def block(x):
        x = Conv2D( filters*4, kernel_size=3, padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

def Encoder():
    input_ = Input( shape=IMAGE_SHAPE )
    x = conv( 128)(input_)
    x = conv( 256)(x)
    x = conv( 512)(x)

    x = conv(1024)(x)
    x = Dense( ENCODER_DIM )( Flatten()(x) )
    x = Dense(4*4*1024)(x)
    x = Reshape((4,4,1024))(x)
    x = upscale(512)(x)
    return Model( input_, [x] )

def Decoder(name):
    input_ = Input( shape=(8,8,512) )
    skip_in = Input( shape=(8,8,512) )

    x = input_
    x = upscale(512)(x)
    x = res_block(x, 512)
    x = upscale(256)(x)
    x = res_block(x, 256)
    x = upscale(128)(x)
    x = res_block(x, 128)
    x = upscale(64)(x)
    x = Conv2D( 3, kernel_size=5, padding='same', activation='sigmoid' )(x)

    y = input_
    y = upscale(512)(y)
    y = upscale(256)(y)
    y = upscale(128)(y)
    y = upscale(64)(y)
    y = Conv2D( 1, kernel_size=5, padding='same', activation='sigmoid' )(y)

    return Model( [input_], outputs=[x,y] )

encoder = Encoder()
decoder_A = Decoder('MA')
decoder_B = Decoder('MB')

print(encoder.summary()) 
print(decoder_A.summary())

x1 = Input( shape=IMAGE_SHAPE )
x2 = Input( shape=IMAGE_SHAPE )
m1 = Input( shape=(64*2,64*2,1) )
m2 = Input( shape=(64*2,64*2,1) )

autoencoder_A = Model( [x1,m1], decoder_A( encoder(x1) ) )
#autoencoder_A = multi_gpu_model( autoencoder_A ,2)

autoencoder_B = Model( [x2,m2], decoder_B( encoder(x2) ) )
#autoencoder_B = multi_gpu_model( autoencoder_B ,2)

o1,om1  = decoder_A( encoder(x1))
o2,om2  = decoder_B( encoder(x2))

DSSIM = DSSIMObjective()
autoencoder_A.compile( optimizer=optimizer, loss=[ penalized_loss(m1, DSSIM),'mse'] )
autoencoder_B.compile( optimizer=optimizer, loss=[ penalized_loss(m2, DSSIM),'mse'] )
