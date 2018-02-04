

from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data
import random
import numpy
import cv2

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, concatenate, Add, add, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.initializers import RandomNormal
from keras.optimizers import Adam

from keras.utils import conv_utils
from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf
import time
from keras_contrib.losses import DSSIMObjective

images_A = get_image_paths( "data/A" )
images_B = get_image_paths( "data/B" )
images_C = get_image_paths( "data/C" )
images_D = get_image_paths( "data/D" )

minImages = max(len(images_A),len(images_B),len(images_C),len(images_D))

random.shuffle(images_A)
random.shuffle(images_B)
random.shuffle(images_C)
random.shuffle(images_D)

images_A,landmarks_A = load_images( images_A[:minImages] ) 
images_B,landmarks_B = load_images( images_B[:minImages] )
images_C,landmarks_C = load_images( images_C[:minImages] )
images_D,landmarks_D = load_images( images_D[:minImages] )

images_A = images_A/255.0
images_B = images_B/255.0
images_C = images_C/255.0
images_D = images_D/255.0


random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.0,
    }

conv_init = RandomNormal(0, 0.02)


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

def random_transform( image, seed, rotation_range, zoom_range, shift_range, random_flip ):
    h,w = image.shape[0:2]
    numpy.random.seed( seed )
    rotation = numpy.random.uniform( -rotation_range, rotation_range )
    scale = numpy.random.uniform( 1 - zoom_range, 1 + zoom_range )
    tx = numpy.random.uniform( -shift_range, shift_range ) * w
    ty = numpy.random.uniform( -shift_range, shift_range ) * h
    mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale )
    mat[:,2] += (tx,ty)
    result = cv2.warpAffine( image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE )
    if numpy.random.random() < random_flip:
        result = result[:,::-1]
    return result


def random_warp(image):
  range_ = numpy.linspace(0, 256, 20)
  mapx = numpy.broadcast_to(range_, (20, 20))
  mapy = mapx.T
  numpy.random.seed( int(time.time()) )
  mapx = mapx + numpy.random.normal(size=(20, 20), scale=5)
  mapy = mapy + numpy.random.normal(size=(20, 20), scale=5)

  interp_mapx = cv2.resize(mapx, (256, 256)).astype('float32')
  interp_mapy = cv2.resize(mapy, (256, 256)).astype('float32')


  return cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)


def getOpposingBestMatches(image_A,landmark_A,images_B,landmarks_B):
  closest = ( numpy.mean(numpy.square(landmark_A-landmarks_B),axis=(1,2)) ).argsort()[0:20]
  closest = numpy.random.choice(closest, 6, replace=False)
  source = cv2.resize( image_A[:,:,:3], (64,64))
  closestMerged = numpy.dstack([  
                                  cv2.resize( images_B[closest[0]][:,:,:3], (64,64)), 
                                  cv2.resize( images_B[closest[1]][:,:,:3], (64,64)), 
                                  cv2.resize( images_B[closest[2]][:,:,:3], (64,64)),
                                  cv2.resize( images_B[closest[3]][:,:,:3], (64,64)),
                                  cv2.resize( images_B[closest[4]][:,:,:3], (64,64)),
                                  cv2.resize( images_B[closest[5]][:,:,:3], (64,64)),
                                ])  
  return source,closestMerged


def get_training_data( images,landmarks,batch_size):
  while 1:
    indices = numpy.random.choice(range(0,images.shape[0]),size=batch_size,replace=True)
    for i,index in enumerate(indices):
      image = images[index]
      seed  = int(time.time())
      image = random_transform( image, seed, **random_transform_args )
      closest = ( numpy.mean(numpy.square(landmarks[index]-landmarks),axis=(1,2)) ).argsort()[1:20]
      closest = numpy.random.choice(closest, 6, replace=False)
      closestMerged = numpy.dstack([  
                                      cv2.resize( random_transform( images[closest[0]][:,:,:3] ,seed, **random_transform_args) , (64,64)), 
                                      cv2.resize( random_transform( images[closest[1]][:,:,:3] ,seed, **random_transform_args) , (64,64)), 
                                      cv2.resize( random_transform( images[closest[2]][:,:,:3] ,seed, **random_transform_args) , (64,64)),
                                      cv2.resize( random_transform( images[closest[3]][:,:,:3] ,seed, **random_transform_args) , (64,64)),
                                      cv2.resize( random_transform( images[closest[4]][:,:,:3] ,seed, **random_transform_args) , (64,64)),
                                      cv2.resize( random_transform( images[closest[5]][:,:,:3] ,seed, **random_transform_args) , (64,64)),
                                    ])

      if i == 0:
          warped_images  = numpy.empty( (batch_size,)  + (64,64,3),   image.dtype )
          example_images = numpy.empty( (batch_size,)  + (64,64,18),  image.dtype )
          target_images  = numpy.empty( (batch_size,)  + (128,128,3), image.dtype )
          mask_images    = numpy.empty( (batch_size,)  + (128,128,1), image.dtype )

      warped_image =  random_warp( image[:,:,:3] )

      warped_image =  cv2.GaussianBlur( warped_image,(91,91),0 )

      image_mask = image[:,:,3].reshape((image.shape[0],image.shape[1],1)) * numpy.ones((image.shape[0],image.shape[1],3)).astype(float)


      foreground = cv2.multiply(image_mask, warped_image.astype(float))
      background = cv2.multiply(1.0 - image_mask, image[:,:,:3].astype(float))

      warped_image = numpy.add(background,foreground)

      warped_image = cv2.resize(warped_image,(64,64))

      warped_images[i]  = warped_image
      example_images[i] = closestMerged
      target_images[i]  = cv2.resize( image[:,:,:3], (128,128) )
      mask_images[i]    = cv2.resize( image[:,:,3], (128,128) ).reshape((128,128,1))
    yield warped_images,example_images,target_images,mask_images



batch_size = 8



class PixelShuffler(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs):

        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            batch_size, c, h, w = input_shape
            if batch_size is None:
                batch_size = -1
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, rh, rw, oc, h, w))
            out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
            out = K.reshape(out, (batch_size, oc, oh, ow))
            return out

        elif self.data_format == 'channels_last':
            batch_size, h, w, c = input_shape
            if batch_size is None:
                batch_size = -1
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
            out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
            out = K.reshape(out, (batch_size, oh, ow, oc))
            return out

    def compute_output_shape(self, input_shape):

        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
            width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
            channels = input_shape[1] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[1]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    channels,
                    height,
                    width)

        elif self.data_format == 'channels_last':
            height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
            width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
            channels = input_shape[3] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[3]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    height,
                    width,
                    channels)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


optimizer = Adam( lr=5e-5, beta_1=0.5, beta_2=0.999 )

import numpy as np

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


input_warped   = Input( shape=(64,64,3) )
input_examples = Input( shape=(64,64,18) )


x = conv( 32)(input_warped)
x = conv( 64)(x)
x = conv( 128)(x)
x = conv( 256)(x)
x = res_block(x, 256)


e = conv( 32)(input_examples)
e = conv( 64)(e)
e = conv( 128)(e)
e = conv( 256)(e)
e = res_block(e, 256)

x = Dense( 2048 )( Flatten()(x) )
x = Dense(4*4*1024)(x)
x = Reshape((4,4,1024))(x)
x = upscale(512)(x)

e = Dense( 2048 )( Flatten()(e) )
e = Dense(4*4*1024)(e)
e = Reshape((4,4,1024))(e)
e = upscale(512)(e)

x = add([e,x])
#x = concatenate([e,x])

e = upscale(512)(e)
x = res_block(x, 512)
x = upscale(256)(x)
x = res_block(x, 256)
x = upscale(128)(x)
x = res_block(x, 128)
x = upscale(64)(x)
x = res_block(x, 64)
x = upscale(32)(x)
#x = res_block(x, 32)
#x = upscale(16)(x)


x = Conv2D( 3, kernel_size=5, padding='same', activation='sigmoid' )(x)
m1 = Input( shape=(128,128,1) )
sumModel  = Model( [input_warped,input_examples,m1], [x] )
print(sumModel.summary())
try:
  sumModel.load_weights('weights.dat')
except:
  print('Weights not found')

DSSIM = DSSIMObjective()

sumModel.compile( optimizer=optimizer, loss=['mae'] )

from keras_contrib.losses import DSSIMObjective


for n in range(900000):
  imaegsAGen = get_training_data(images_A, landmarks_A, batch_size)
  imaegsBGen = get_training_data(images_B, landmarks_B, batch_size)
  imaegsCGen = get_training_data(images_C, landmarks_C, batch_size)
  imaegsDGen = get_training_data(images_D, landmarks_D, batch_size)

  xa,xae,ya,ma = next(imaegsAGen)
  xb,xbe,yb,mb = next(imaegsBGen)
  xc,xce,yc,mc = next(imaegsCGen)
  xd,xde,yd,md = next(imaegsDGen)



  xl = numpy.concatenate((xa, xb, xc, xd), axis=0)
  xel = numpy.concatenate((xae, xbe, xce, xde), axis=0)
  yl = numpy.concatenate((ya, yb, yc, yd), axis=0)
  ml = numpy.concatenate((ma, mb, mc, md), axis=0)



  indices = numpy.random.choice(range(0,xl.shape[0]),size=xl.shape[0],replace=False)



  print(sumModel.train_on_batch([xl[indices],xel[indices],ml[indices]], yl[indices]))

  numpy.random.seed( int(time.time()) )
  if n%100 == 0:
    sumModel.save_weights('weights.dat')
    print("saving weights")
  
  if n%5 == 0:
    testindexA = numpy.random.choice(range(0,images_A.shape[0]))
    testindexB = numpy.random.choice(range(0,images_B.shape[0]))
    testindexC = numpy.random.choice(range(0,images_C.shape[0]))
    testindexD = numpy.random.choice(range(0,images_D.shape[0]))



  zmask = numpy.zeros((1,128,128,1),float)

  genSet = [
    ("A->B",testindexA,images_A,images_B,landmarks_A,landmarks_B),
    ("B->A",testindexB,images_B,images_A,landmarks_B,landmarks_A),
    ("C->D",testindexC,images_C,images_D,landmarks_C,landmarks_D),
    ("D->C",testindexD,images_D,images_C,landmarks_D,landmarks_C),
    ("A->C",testindexA,images_A,images_C,landmarks_A,landmarks_C),
    ("B->D",testindexB,images_B,images_D,landmarks_B,landmarks_D),
  ]


  figList = []

  for name,index,imagesSrc,imagesDst,landmarksSrc,landmarksDst in genSet:
    testImage = imagesSrc[index]
    landmarkSrc  = landmarksSrc[index]
    src,examp = getOpposingBestMatches(testImage, landmarkSrc, imagesDst, landmarksDst)
    pred = sumModel.predict( [numpy.array([src]),numpy.array([examp]),zmask] )
    figList.append(
      numpy.concatenate( [
            cv2.resize( testImage[:,:,:3],  (128,128)),
            cv2.resize( pred[0],  (128,128)),
            cv2.resize( examp[:,:,:3],  (128,128)),

                        ] , axis=1)



    )


  stack = numpy.concatenate(figList,axis=0)

  stack = numpy.clip( stack * 255, 0, 255 ).astype('uint8')
  stack = stack_images( stack )

  cv2.imshow("p", stack)


  if cv2.waitKey(1) == ord('q'):
    exit()
