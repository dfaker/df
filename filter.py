import argparse
import skimage
import json
import numpy
from pathlib import Path
from tqdm import tqdm

import queue
import threading
from multiprocessing import Process, Queue

import av

from face_alignment import FaceAlignment, LandmarksType
import dlib
import face_recognition_models

rect = dlib.rectangle(0, 0, 0, 0)

def filter_face(input_dir, output_dir, processn, alignments, fps, filter_encodings, workerqueue):
    face_recognition_model = face_recognition_models.face_recognition_model_location()
    face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
    filter_threshold = 0.56
    
    def encode_one_face_local(image, faces):
        for points in faces:
            parts = []
            for p in points:
                parts.append(dlib.point(p[0], p[1]))
            raw_landmark_set = dlib.full_object_detection(rect, parts)
            yield numpy.array(face_encoder.compute_face_descriptor(image, raw_landmark_set, 1))
            
    container = av.open(input_dir)
    stream = container.streams.video[0]
    container.seek(int(alignments[0][0] * stream.time_base * 1000000))
    fps = 1/fps
    
    idx = 0
    length = len(alignments)
    
    frame = next(f for f in container.decode(video=0))
    timenext = frame.pts * stream.time_base + fps
    
    def process_frame():
        nonlocal idx
        while (idx < length) and (alignments[idx][0] < frame.pts):
            idx += 1
        face_files = []
        faces = []
        while (idx < length) and (frame.pts == alignments[idx][0]):
            face_file = output_dir / alignments[idx][1]
            if face_file.exists():
                face_files.append(face_file)
                faces.append(numpy.array( alignments[idx][3] ).reshape((-1,2)).astype(int))
            idx += 1
        if len(face_files) > 0:
            scores = []
            encodings = list(encode_one_face_local(frame.to_nd_array(format='rgb24'), faces))
            min = 1
            for (face_file, encoding) in zip(face_files, encodings):
                score = numpy.linalg.norm(filter_encodings - encoding, axis=1)
                scores.append(score)
                t = score.min()
                if t < min:
                     min = t
                if t > filter_threshold:
                    face_file.replace(output_dir / 'filter1' / face_file.name)
            if len(face_files) > 1:
                for i, face_file in enumerate(face_files):
                    if face_file.exists() and scores[i].min() > min:
                        face_file.replace(output_dir / 'filter2' / face_file.name)
            workerqueue.put((processn, idx, face_files, scores))
            
    process_frame()

    for frame in container.decode(video=0):
        timenow = frame.pts * stream.time_base
        if (timenow >= timenext):
            timenext += fps
            process_frame()
            if idx >= length:
                break
        
    workerqueue.put(None)

def stat_update(output_dir, process_number, align_length, workerqueue):
    
    face_distance_log = open(str(output_dir / 'face_distance_log.txt'), 'w')
    stat = []
    pbar = tqdm(total=align_length, ascii=True)
    
    for i in range(process_number):
        stat.append(0)

    for i in range(process_number):
        while True:
            item = workerqueue.get()
            if item is None:
                break
            
            processn, idx, face_files, scores = item
            pbar.update(idx - stat[processn])
            stat[processn] = idx
            
            for (face_file, score) in zip(face_files, scores):
                face_distance_log.write('{}\t{}\n'.format(face_file.stem, score))
           
    face_distance_log.close()
    pbar.close()

def main( args ):
    def encode_filter(filter_files):
        images = []
        faces = []
        
        FACE_ALIGNMENT = FaceAlignment( LandmarksType._2D, enable_cuda=True, flip_input=False )
        for i, filter_file in enumerate(filter_files):
            images.append(skimage.io.imread( str(filter_file) ))
            faces.append(FACE_ALIGNMENT.get_landmarks( images[i] ))
        FACE_ALIGNMENT = None
        
        face_recognition_model = face_recognition_models.face_recognition_model_location()
        face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
        for i, face in enumerate(faces):
            if face is None:
                print('Warning: {} has no face.'.format(filter_files[i]))
                continue
            if len(face) > 1:
                print('Warning: {} has more than one face.'.format(filter_files[i]))
                
            parts = []
            for p in face[0]:
                parts.append(dlib.point(p[0], p[1]))
            raw_landmark_set = dlib.full_object_detection(rect, parts)
            yield numpy.array(face_encoder.compute_face_descriptor(images[i], raw_landmark_set, 1))

    output_dir = Path(args.output_dir)
    filter_dir = Path(args.filter_dir)
    assert filter_dir.is_dir()
    
    filter_files = list( filter_dir.glob( "*.png" ) ) + list( filter_dir.glob( "*.jpg" ) )
    assert len( filter_files ) > 0, "Can't find filter files"
    
    (output_dir / 'filter1').mkdir( parents=True, exist_ok=True ) # filter1 for faces under filter_threshold
    (output_dir / 'filter2').mkdir( parents=True, exist_ok=True ) # filter2 for faces in the frame which has more than one face

    alignments = output_dir / args.alignments
    with alignments.open() as f:
        alignments = json.load(f)
    container = av.open(args.input_dir)
    stream = container.streams.video[0]
    align_start = 0
    while (alignments[align_start][0] * stream.time_base) < args.seekstart:
        align_start += 1
    align_end = len(alignments) - 1
    while (alignments[align_end][0] * stream.time_base) > args.seekstart + args.durationtime:
        align_end -= 1
    align_length = align_end-align_start+1
    align_d = (align_length // args.process_number) + 1
    align_m = align_length % args.process_number
    
    filter_encodings = list(encode_filter(filter_files))

    workerqueue = Queue(maxsize=1)
    process_list = []
    for i in range(args.process_number):
        if i == align_m:
            align_d -= 1
        t = align_start + align_d
        process_list.append(Process(target=filter_face, args=(args.input_dir, output_dir, i, alignments[align_start:t], args.fps, filter_encodings, workerqueue)))
        process_list[i].start()
        align_start = t

    stat_update(output_dir, args.process_number, align_length, workerqueue)

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
    parser.add_argument('-f', '--filter_dir',
                        dest="filter_dir",
                        type=str,
                        default='filter')                        
    parser.add_argument('-pn', '--process_number',
                        dest="process_number",
                        type=int,
                        default=4)                        
    main( parser.parse_args() )
