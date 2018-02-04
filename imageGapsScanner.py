

import numpy
import argparse
import os
import json
from tqdm import tqdm
import cv2

def main(args):
  


  if not os.path.exists(args.input_dir_a[0]):
    raise Exception("Folder A {} does not exist".format(args.input_dir_a[0]))
  if not os.path.exists(args.input_dir_b[0]):
    raise Exception("Folder B {} does not exist".format(args.input_dir_b[0]))

  dira = args.input_dir_a[0]
  dirb = args.input_dir_b[0]
  alinmentsFileName = 'alignments.json'

  alignmentsa = os.path.join( dira,alinmentsFileName )
  alignmentsb = os.path.join( dirb,alinmentsFileName )

  if not os.path.exists(alignmentsa):
    raise Exception("Folder A {} does not contain an alignments.json file".format(args.input_dir_a[0]))
  if not os.path.exists(alignmentsb):
    raise Exception("Folder B {} does not contain an alignments.json file".format(args.input_dir_b[0]))

  alignmentsa = json.loads( open(alignmentsa).read() )
  pointsa = []
  filenamesa = []
  pbar = tqdm(alignmentsa)
  for original,cropped,mat,points in pbar:
    cropped = os.path.split(cropped)[1]
    cropped = os.path.join(dira,cropped)

    croppedjpg = cropped.replace('.png','.jpg')
    croppedpng = cropped.replace('.jpg','.png')

    if (not os.path.exists(croppedjpg)) and os.path.exists(croppedpng):
      cropped = croppedpng
    if (not os.path.exists(croppedpng)) and os.path.exists(croppedjpg):
      cropped = croppedjpg

    if os.path.exists(cropped):
      mat = numpy.array(mat).reshape(2,3)
      facepoints = cv2.transform( numpy.array( points ).reshape((1,-1,2)) , mat).reshape(-1,2).astype(int)
      pointsa.append( numpy.array(facepoints).reshape((-1,2)) )
      filenamesa.append(cropped)

  alignmentsb = json.loads( open(alignmentsb).read() )
  pointsb = []
  filenamesb = []
  pbar = tqdm(alignmentsb)
  for original,cropped,mat,points in pbar:
    cropped = os.path.split(cropped)[1]
    cropped = os.path.join(dirb,cropped)

    croppedjpg = cropped.replace('.png','.jpg')
    croppedpng = cropped.replace('.jpg','.png')

    if (not os.path.exists(croppedjpg)) and os.path.exists(croppedpng):
      cropped = croppedpng
    if (not os.path.exists(croppedpng)) and os.path.exists(croppedjpg):
      cropped = croppedjpg

    if os.path.exists(cropped):
      mat = numpy.array(mat).reshape(2,3)
      facepoints = cv2.transform( numpy.array( points ).reshape((1,-1,2)) , mat).reshape(-1,2).astype(int)
      pointsb.append( numpy.array(facepoints).reshape((-1,2)) )
      filenamesb.append(cropped)

  if len(filenamesa)<9:
    raise Exception("Folder A {} must contain at least 9 images, {} found".format(args.input_dir_a[0],len(filenamesa)))

  if len(filenamesb)<9:
    raise Exception("Folder B {} must contain at least 9 images, {} found".format(args.input_dir_b[0],len(filenamesb)))

  pointsa = numpy.array(pointsa)
  pointsb = numpy.array(pointsb)

  distancesa = []
  for ind in range(0,pointsa.shape[0]):
    closest = ( numpy.mean(numpy.square(pointsa[ind]-pointsb),axis=(1,2)) ).min()
    distancesa.append((closest,ind))

  distancesb = []
  for ind in range(0,pointsb.shape[0]):
    closest = ( numpy.mean(numpy.square(pointsb[ind]-pointsa),axis=(1,2)) ).min()
    distancesb.append((closest,ind))

  worsta = numpy.concatenate( [ cv2.imread(filenamesa[x[1]]) for x in sorted(distancesa,reverse=True)[:9]], axis=1)
  worstb = numpy.concatenate( [ cv2.imread(filenamesb[x[1]]) for x in sorted(distancesb,reverse=True)[:9]], axis=1)

  worst  = numpy.concatenate((worsta, worstb), axis=0)

  cv2.imshow("",worst)
  cv2.waitKey(0)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Compares two folders containing images and an alinments.json')
  parser.add_argument( "input_dir_a", type=str, nargs=1, help='First folder for comparison of contents')
  parser.add_argument( "input_dir_b", type=str, nargs=1, help='Second folder for comparison of contents' )
  main( parser.parse_args() )