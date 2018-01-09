import numpy
import queue
import threading
from image_augmentation import random_transform
from image_augmentation import random_warp,random_warp_src_dest
import cv2


random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.0,
    }



def async_get_training_data( images, srcPoints, dstPoints, batch_size,workerqueue ):


    while 1:
      indices = numpy.random.choice(range(0,images.shape[0]),size=batch_size,replace=False)
      for i,index in enumerate(indices):
          image = images[index]
          image = random_transform( image, **random_transform_args )

          closest = ( numpy.mean(numpy.square(srcPoints[index]-dstPoints),axis=(1,2)) ).argsort()[:10]
          closest = numpy.random.choice(closest)
          warped_img, target_img, mask_image = random_warp_src_dest( image,srcPoints[index],dstPoints[ closest ] )
 
          if numpy.random.random() < 0.5:
            warped_img = warped_img[:,::-1]
            target_img = target_img[:,::-1]
            mask_image = mask_image[:,::-1]


          if i == 0:
              warped_images = numpy.empty( (batch_size,) + warped_img.shape, warped_img.dtype )
              target_images = numpy.empty( (batch_size,) + target_img.shape, warped_img.dtype )
              mask_images = numpy.empty( (batch_size,)   + mask_image.shape, mask_image.dtype )

          warped_images[i] = warped_img
          target_images[i] = target_img
          mask_images[i]   = mask_image


      workerqueue.put( (warped_images, target_images , mask_images ) )


queues = {
  
}
t=None
q=None
cycle=0
def get_training_data( images , srcPoints, dstPoints, batch_size ):
  global cycle
  cycle+=1
  key = (id(images),batch_size,cycle%3)
  if key in queues:
    t,q = queues.get(key)
  else:
    print('New Image thread start',key)
    q = queue.Queue(maxsize=20)
    t = threading.Thread(target=async_get_training_data,args=(images, srcPoints, dstPoints, batch_size,q))
    t.daemon = True
    t.start()
    queues[key]=(t,q)
  return q.get()
