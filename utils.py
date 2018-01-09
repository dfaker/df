import cv2
import numpy
import os
import json

def get_image_paths( directory ):
    return [ x.path for x in os.scandir( directory ) if x.name.endswith(".jpg") or x.name.endswith(".png") ]

from tqdm import tqdm

guideShades = numpy.linspace(20,250,68)

def load_images_masked(image_paths, convert=None,blurSize=35):
  basePath = os.path.split(image_paths[0])[0]
  alignments = os.path.join(basePath,'alignments.json')
  alignments = json.loads( open(alignments).read() )

  all_images    = []
  landmarks     = []


  pbar = tqdm(alignments)
  for original,cropped,mat,points in pbar:
    pbar.set_description('loading '+basePath)
    cropped = os.path.split(cropped)[1]
    cropped = os.path.join(basePath,cropped)
    if cropped in image_paths and os.path.exists(cropped):
      cropped = cv2.imread(cropped).astype(float)

      mat = numpy.array(mat).reshape(2,3)
      points = numpy.array(points).reshape((-1,2))

      mat = mat*160
      mat[:,2] += 42

      facepoints = numpy.array( points ).reshape((-1,2))

      mask = numpy.zeros_like(cropped,dtype=numpy.uint8)

      hull = cv2.convexHull( facepoints.astype(int) )
      hull = cv2.transform( hull.reshape(1,-1,2) , mat).reshape(-1,2).astype(int)

      cv2.fillConvexPoly( mask,hull,(255,255,255) )

      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

      mask = cv2.dilate(mask,kernel,iterations = 1,borderType=cv2.BORDER_REFLECT )

      facepoints = cv2.transform( numpy.array( points ).reshape((1,-1,2)) , mat).reshape(-1,2).astype(int)

      mask = mask[:,:,0]

      all_images.append(  numpy.dstack([cropped,mask]).astype(numpy.uint8) )
      landmarks.append( facepoints )

  return numpy.array(all_images),numpy.array(landmarks)


def load_images_std( image_paths, convert=None ):
    iter_all_images = ( cv2.imread(fn)  for fn in image_paths )
    if convert:
        iter_all_images = ( convert(img) for img in iter_all_images )
    for i,image in enumerate( iter_all_images ):
        if i == 0:
            all_images = numpy.empty( ( len(image_paths), ) + image.shape, dtype=image.dtype )
        all_images[i] = image
    return all_images


load_images = load_images_masked

def get_transpose_axes( n ):
    if n % 2 == 0:
        y_axes = list( range( 1, n-1, 2 ) )
        x_axes = list( range( 0, n-1, 2 ) )
    else:
        y_axes = list( range( 0, n-1, 2 ) )
        x_axes = list( range( 1, n-1, 2 ) )
    return y_axes, x_axes, [n-1]

def stack_images( images ):
    images_shape = numpy.array( images.shape )
    new_axes = get_transpose_axes( len( images_shape ) )
    new_shape = [ numpy.prod( images_shape[x] ) for x in new_axes ]
    return numpy.transpose(
        images,
        axes = numpy.concatenate( new_axes )
        ).reshape( new_shape )
