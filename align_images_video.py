import argparse
import cv2
import json
import numpy
from pathlib import Path
from tqdm import tqdm
from umeyama import umeyama

import queue
import threading

import av

from face_alignment import FaceAlignment, LandmarksType

FACE_ALIGNMENT = FaceAlignment( LandmarksType._2D, enable_cuda=True, flip_input=False )

mean_face_x = numpy.array([
0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
0.553364, 0.490127, 0.42689 ])

mean_face_y = numpy.array([
0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
0.784792, 0.824182, 0.831803, 0.824182 ])

landmarks_2D = numpy.stack( [ mean_face_x, mean_face_y ], axis=1 )

def transform( image, mat, size, padding=0 ):
    mat = mat * size
    mat[:,2] += padding
    new_size = int( size + padding * 2 )
    return cv2.warpAffine( image, mat, ( new_size, new_size ) )

def prepare_image(input_dir, seekstart, durationtime, fps, workerqueue):
    container = av.open(input_dir)
    stream = container.streams.video[0]
    container.seek(seekstart*1000000)
    
    frame = next(f for f in container.decode(video=0))
    videostart = frame.pts * stream.time_base
    
    endtime = seekstart + durationtime
    d = min(endtime, stream.duration * stream.time_base) - videostart
    pbar = tqdm(total=(int(d*min(stream.average_rate, fps))+1), ascii=True)

    image = frame.to_nd_array(format='rgb24')
    workerqueue.put((frame.pts, image))
    pbar.update(1)
    
    fps = 1/fps
    timenext = videostart + fps
    for frame in container.decode(video=0):
        timenow = frame.pts * stream.time_base
        if timenow > endtime:
            break
        if (timenow >= timenext):
            timenext += fps
            image = frame.to_nd_array(format='rgb24')
            workerqueue.put((frame.pts, image))
            pbar.update(1)
        
    workerqueue.put(None)
    pbar.close()

def find_face(inputqueue, outputqueue):
    while True:
        item = inputqueue.get()
        if item is None:
            break
            
        idx, image = item
        faces = FACE_ALIGNMENT.get_landmarks( image )
        if faces is None: continue
        if len(faces) == 0: continue
        
        outputqueue.put((idx, image, faces))
        
    outputqueue.put(None)

def align_face(output_dir, workerqueue):
    while True:
        item = workerqueue.get()
        if item is None:
            break

        idx, image, faces = item
        
        image = image[:, :, ::-1]
        
        for i,points in enumerate(faces):
            alignment = umeyama( points[17:], landmarks_2D, True )[0:2]
            aligned_image = transform( image, alignment, 80, 24 )
            # aligned_image = transform( image, alignment, 160, 48 )
        
            if len(faces) == 1:
                out_fn = "{:0>10}.jpg".format( idx )
            else:
                out_fn = "{:0>10}_{}.jpg".format( idx, i )
        
            out_fn = output_dir / out_fn
            cv2.imwrite( str(out_fn), aligned_image )
        
            yield idx, str(out_fn.relative_to(output_dir)), list( alignment.ravel() ), list(points.flatten().astype(float))

def main( args ):
    output_dir = Path(args.output_dir)
    output_dir.mkdir( parents=True, exist_ok=True )

    output_file = output_dir / args.alignments
    
    queue_prepare_image = queue.Queue(maxsize=1)
    thread_prepare_image = threading.Thread(target=prepare_image, args=(args.input_dir, args.seekstart, args.durationtime, args.fps, queue_prepare_image))
    thread_prepare_image.start()

    queue_find_face = queue.Queue(maxsize=1)
    thread_find_face = threading.Thread(target=find_face, args=(queue_prepare_image, queue_find_face))
    thread_find_face.start()
    
    face_alignments = list( align_face(output_dir, queue_find_face) )

    with output_file.open('w') as f:
        results = json.dumps( face_alignments, ensure_ascii=False )
        f.write( results )

    print( "Save face alignments to output file:", output_file )    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir',
                        dest="input_dir",
                        type=str,
                        default="input.mp4",
                        help="Input video")
    parser.add_argument('-o', '--output-dir',
                        dest="output_dir",
                        type=str,
                        default="aligned",
                        help="Output directory. This is where the aligned images will \
                        be stored. Defaults to 'aligned'")
    parser.add_argument('-a', '--alignments',
                        dest="alignments",
                        type=str,
                        default='alignments.json')
    parser.add_argument('-ss', '--seekstart',
                        dest="seekstart",
                        type=int,
                        default=0)
    parser.add_argument('-t', '--durationtime',
                        dest="durationtime",
                        type=int,
                        default=100000)                        
    parser.add_argument('-vf', '--fps',
                        dest="fps",
                        type=float,
                        default=1000.0)                        
    main( parser.parse_args() )
