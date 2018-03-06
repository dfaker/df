import cv2
import numpy

from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data
import glob

import random

from scipy.stats import linregress
from tqdm import tqdm

if __name__ == '__main__':

  print('running')

  images_A = get_image_paths( "data/A" )
  images_B = get_image_paths( "data/B"  )

  minImages = 2000#min(len(images_A),len(images_B))*20

  random.shuffle(images_A)
  random.shuffle(images_B)

  images_A,landmarks_A = load_images( images_A[:minImages] ) 
  images_B,landmarks_B = load_images( images_B[:minImages] )

  print('Images A', images_A.shape)
  print('Images B', images_B.shape)

  images_A = images_A/255.0
  images_B = images_B/255.0

  images_A[:,:,:,:3] += images_B[:,:,:,:3].mean( axis=(0,1,2) ) - images_A[:,:,:,:3].mean( axis=(0,1,2) )

  print( "press 'q' to stop training and save model" )

  batch_size = int(32)

  warped_A, target_A, mask_A = get_training_data( images_A,  landmarks_A,landmarks_B, batch_size )
  warped_B, target_B, mask_B  = get_training_data( images_B, landmarks_B,landmarks_A, batch_size )


  print(warped_A.shape, target_A.shape, mask_A.shape)

  figWarped = numpy.stack([warped_A[:6],warped_B[:6]],axis=0 )
  figWarped = numpy.clip( figWarped * 255, 0, 255 ).astype('uint8')
  figWarped = stack_images( figWarped )
  cv2.imshow( "w", figWarped )

  print(warped_A.shape)
  print(target_A.shape)

  from model import autoencoder_A
  from model import autoencoder_B


  from model import encoder, decoder_A, decoder_B


  try:
      encoder  .load_weights( "models/encoder.h5"   )
      decoder_A.load_weights( "models/decoder_A.h5" )
      decoder_B.load_weights( "models/decoder_B.h5" )
  except:
      pass

  def save_model_weights():
      encoder  .save_weights( "models/encoder.h5"   )
      decoder_A.save_weights( "models/decoder_A.h5" )
      decoder_B.save_weights( "models/decoder_B.h5" ) 
      print( "\nSave model weights" )



  while 1:
    pbar = tqdm(range(1000000))
    for epoch in pbar:

      
        warped_A, target_A, mask_A = get_training_data( images_A,  landmarks_A,landmarks_B, batch_size )
        warped_B, target_B, mask_B = get_training_data( images_B, landmarks_B,landmarks_A, batch_size )

      
        omask = numpy.ones((target_A.shape[0],64,64,1),float)


        loss_A = autoencoder_A.train_on_batch([warped_A,mask_A], [target_A,mask_A])
        loss_B = autoencoder_B.train_on_batch([warped_B,mask_B], [target_B,mask_B])

 
        pbar.set_description("Loss A [{}] Loss B [{}]".format(loss_A,loss_B))


        if epoch % 100 == 0:
          save_model_weights()
          test_A = target_A[0:8,:,:,:3]
          test_B = target_B[0:8,:,:,:3]

          test_A_i = []
          test_B_i = []
          
          for i in test_A:
            test_A_i.append(cv2.resize(i,(64,64),cv2.INTER_AREA))
          test_A_i = numpy.array(test_A_i).reshape((-1,64,64,3))

          for i in test_B:
            test_B_i.append(cv2.resize(i,(64,64),cv2.INTER_AREA))
          test_B_i = numpy.array(test_B_i).reshape((-1,64,64,3))






        figWarped = numpy.stack([warped_A[:6],warped_B[:6]],axis=0 )
        figWarped = numpy.clip( figWarped * 255, 0, 255 ).astype('uint8')
        figWarped = stack_images( figWarped )
        cv2.imshow( "w", figWarped )

        
        zmask = numpy.zeros((test_A.shape[0],128,128,1),float)

        pred_a_a,pred_a_a_m = autoencoder_A.predict([test_A_i,zmask])
        pred_b_a,pred_b_a_m = autoencoder_B.predict([test_A_i,zmask])

        pred_a_b,pred_a_b_m = autoencoder_A.predict([test_B_i,zmask])
        pred_b_b,pred_b_b_m = autoencoder_B.predict([test_B_i,zmask])

        pred_a_a = pred_a_a[0:18,:,:,:3]
        pred_a_b = pred_a_b[0:18,:,:,:3]
        pred_b_a = pred_b_a[0:18,:,:,:3]
        pred_b_b = pred_b_b[0:18,:,:,:3]

        figure_A = numpy.stack([
            test_A,
            pred_a_a,
            pred_b_a,
            ], axis=1 )
        figure_B = numpy.stack([
            test_B,
            pred_b_b,
            pred_a_b,
            ], axis=1 )


        figure = numpy.concatenate( [ figure_A, figure_B ], axis=0 )
        figure = figure.reshape( (4,4) + figure.shape[1:] )
        figure = stack_images( figure )

        figure = numpy.clip( figure * 255, 0, 255 ).astype('uint8')


        cv2.imshow( "p", figure )
    
        key = cv2.waitKey(1)
        if key == ord('q'):
            save_model_weights()
            exit()

