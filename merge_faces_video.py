import time
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import queue
import threading
from multiprocessing import Process, Queue

import av
import cv2
import numpy

zmask = numpy.zeros((1,128, 128,1),float)

def adjust_avg_color(img_old,img_new):
    w,h,c = img_new.shape
    for i in range(img_new.shape[-1]):
        old_avg = img_old[:, :, i].mean()
        new_avg = img_new[:, :, i].mean()
        diff_int = (int)(old_avg - new_avg)
        for m in range(img_new.shape[0]):
            for n in range(img_new.shape[1]):
                temp = (img_new[m,n,i] + diff_int)
                if temp < 0:
                    img_new[m,n,i] = 0
                elif temp > 255:
                    img_new[m,n,i] = 255
                else:
                    img_new[m,n,i] = temp

def decoder(alignment_dir, alignments, seekstart, durationtime, facequeue, framequeue):
    global input, output, invstream, inastream, outastream, vbasepts
    
    idx = 0
    length = len(alignments)
    
    videopackets = []
    audiopackets = []
    for inpacket in input.demux():
        if inpacket.stream.type == "video":
            videopackets.append(inpacket)
            if len(audiopackets) > 0:
                break
        elif inpacket.stream.type == "audio":
            audiopackets.append(inpacket)
            if len(videopackets) > 0:
                break
    videostart = videopackets[0].pts * invstream.time_base
    audiostart = audiopackets[0].pts * inastream.time_base            
    if videostart <= audiostart:
        vbasepts = videopackets[0].pts
        abasepts = round(videostart / inastream.time_base)
    else:
        abasepts = audiopackets[0].pts
        vbasepts = round(audiostart / invstream.time_base)
        
    endtime = seekstart + durationtime
    d = min(endtime, invstream.duration * invstream.time_base) - videostart
    pbar = tqdm(total=(int(d*invstream.average_rate)+1), ascii=True)
    def video_process(inpacket):
        nonlocal idx
        for frame in inpacket.decode():
            processed = False
            while (idx < length) and (alignments[idx][0] < frame.pts):
                idx += 1
            while (idx < length) and (frame.pts == alignments[idx][0]):
                face_file = alignment_dir / alignments[idx][1]
                if face_file.exists():
                    image = frame.to_nd_array(format='bgr24')
                    mat = numpy.array(alignments[idx][2]).reshape(2,3)
                    
                    sourceMat = mat.copy()
                    
                    # sourceMat = sourceMat*(240+(16*2))
                    # sourceMat[:,2] += 48
                    # face = cv2.warpAffine( image, sourceMat, (240+(48+16)*2,240+(48+16)*2) )
                    
                    sourceMat = sourceMat*160
                    sourceMat[:,2] += 40
                    face = cv2.warpAffine( image, sourceMat, (240,240) )
                    
                    sourceFace = face.copy()
                    
                    face = cv2.resize(face,(64,64),cv2.INTER_AREA)
                    face = numpy.expand_dims( face, 0 )
                    
                    facequeue.put((image, mat, sourceFace, face / 255.0, alignments[idx][0]))
                    processed = True
                    idx += 1
                    break
                    
                idx += 1
                
            if not processed:
                # framequeue.put((frame.pts, frame))
                framequeue.put((frame.pts, av.VideoFrame.from_ndarray(frame.to_nd_array(format='bgr24'), format='bgr24')))
            else:
                framequeue.put((frame.pts, None))
                
            pbar.update(1)
    
    def audio_mux_one(inpacket):
        inpacket.stream = outastream
        inpacket.pts -= abasepts
        inpacket.dts -= abasepts
        output.mux(inpacket)
    
    for inpacket in videopackets:
        video_process(inpacket)
    for inpacket in audiopackets:
        audio_mux_one(inpacket)
        
    for inpacket in input.demux():
        if inpacket.stream.type == "video":
            video_process(inpacket)
        elif (inpacket.stream.type == "audio") and inpacket.pts:
            if (inpacket.pts * inastream.time_base) > endtime:
                break
            audio_mux_one(inpacket)
        
    facequeue.put(None)
    framequeue.put(None)
    pbar.close()
    
def convert_face(inputqueue, outputqueue, swap_model, thread_number, double_pass):
    from model import autoencoder_A
    from model import autoencoder_B
    from model import encoder, decoder_A, decoder_B
    
    encoder  .load_weights( "models/encoder.h5"   )
    decoder_A.load_weights( "models/decoder_A.h5" )
    decoder_B.load_weights( "models/decoder_B.h5" )
    
    if swap_model: autoencoder,otherautoencoder = autoencoder_A,autoencoder_B
    else:          autoencoder,otherautoencoder = autoencoder_B,autoencoder_A
    
    while True:
        item = inputqueue.get()
        if item is None:
            break
        image, mat, sourceFace, face, framepts = item
 
        new_face_rgb, new_face_m = autoencoder.predict( [face, zmask] )
        if double_pass:
          #feed the original prediction back into the network for a second round.
          new_face_rgb = new_face_rgb.reshape((128, 128, 3))
          new_face_rgb = cv2.resize( new_face_rgb , (64,64))
          new_face_rgb = numpy.expand_dims( new_face_rgb, 0 )
          new_face_rgb,_ = autoencoder.predict( [new_face_rgb, zmask] )
        
        _,other_face_m = otherautoencoder.predict( [face, zmask] )
 
        outputqueue.put( (image, mat, sourceFace, new_face_rgb, new_face_m, other_face_m, framepts) )
        
    for i in range(thread_number):
        outputqueue.put(None)

def merge_face(inputqueue, erosion_kernel, blur_size, seamless_clone, outputqueue):
    while True:
        item = inputqueue.get()
        if item is None:
            break
        image, mat, sourceFace, new_face_rgb, new_face_m, other_face_m, framepts = item
        
        image_size = image.shape[1], image.shape[0]
        
        mat = mat*160*128/240
        transmat = mat.copy()
        transmat[:,2] += 36.25*128/240
        
        # mat =  mat * (64-16) *2
        # mat[::,2] += 8*2
        
        new_face_rgb = numpy.clip( new_face_rgb[0] * 255, 0, 255 ).astype( image.dtype )
        new_face_rgb = new_face_rgb[2:126, 2:126, :]
        
        base_image = numpy.copy( image )
        new_image = numpy.copy( image )
        
        if not seamless_clone:
            sourceFace = cv2.resize(sourceFace,(128,128),cv2.INTER_AREA)
            adjust_avg_color(sourceFace[2:126, 2:126, :],new_face_rgb)

        cv2.warpAffine( new_face_rgb, transmat, image_size, new_image, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
        
        mat[:,2] += 34.375*128/240
        
        new_face_m = numpy.maximum(new_face_m, other_face_m)
        new_face_m   = numpy.clip( new_face_m[0]  , 0, 1 ).astype( float ) * numpy.ones((new_face_m.shape[0],new_face_m.shape[1],3))
        image_mask = numpy.zeros_like(new_image, dtype=float)
        cv2.warpAffine( new_face_m[3:125, 3:125, :], mat, image_size, image_mask, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
        
        if erosion_kernel is not None:
          image_mask = cv2.erode(image_mask, erosion_kernel, iterations = 1)
        
        #slightly enlarge the mask area
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        # image_mask = cv2.dilate(image_mask,kernel,iterations = 1)
        
        if seamless_clone:
        
            unitMask = numpy.clip( image_mask * 365, 0, 255 ).astype(numpy.uint8)
            # unitMask = (image_mask*255).astype(numpy.uint8)
            
            maxregion = numpy.argwhere(unitMask==255)
            
            if maxregion.size > 0:
              miny,minx = maxregion.min(axis=0)[:2]
              maxy,maxx = maxregion.max(axis=0)[:2]
              lenx = maxx - minx;
              leny = maxy - miny;
              masky = int(minx+(lenx//2))
              maskx = int(miny+(leny//2))
              
              # new_image = cv2.seamlessClone(new_image.astype(numpy.uint8),base_image.astype(numpy.uint8),unitMask,(masky,maskx) , cv2.NORMAL_CLONE)
              outputqueue.put((framepts, cv2.seamlessClone(new_image.astype(numpy.uint8),base_image.astype(numpy.uint8),unitMask,(masky,maskx) , cv2.NORMAL_CLONE)))
              continue
        
        # image_mask = cv2.GaussianBlur(image_mask,(63,63),0)

        if blur_size!=0:
          image_mask = cv2.GaussianBlur(image_mask,(blur_size,blur_size),0)
        
        foreground = cv2.multiply(image_mask, new_image.astype(float))
        background = cv2.multiply(1.0 - image_mask, base_image.astype(float))
        
        outputqueue.put((framepts, numpy.add(background,foreground).astype(numpy.uint8)))

def encoder(thread_number, framequeue, queue_list):
    global output, invstream, outvstream, vbasepts
    
    faceitems = []
    for i in range(thread_number):
        faceitems.append(None)
        
    while True:
        frameitem = framequeue.get()
        if frameitem is None:
            break
            
        framepts, frame = frameitem
        while frame is None:
            time.sleep(0.3) # You can tune it small for faster speed if you have more powerful machine.
            for i in range(thread_number):
                if faceitems[i] is None:
                    try:
                        faceitems[i] = queue_list[i].get_nowait()
                    except:
                        continue
                pts, img = faceitems[i]
                if pts == framepts:
                    frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                    faceitems[i] = None
                    break
        
        frame.pts = framepts - vbasepts
        frame.time_base = invstream.time_base
        for outpacket in outvstream.encode(frame):
            output.mux(outpacket)
            
    for outpacket in outvstream.encode(None):
        output.mux(outpacket)
    
    output.close()

def main(args):
    global input, output, invstream, outvstream, inastream, outastream
    
    input = av.open(args.input_file)
    invstream = input.streams.video[0]
    inastream = input.streams.audio[0]
    
    frame = next(input.decode(video=0))
    
    output = av.open(args.output_file, 'w')
    outvstream = output.add_stream(args.codec, invstream.rate)
    outvstream.pix_fmt = invstream.pix_fmt
    outvstream.height = invstream.height
    outvstream.width = invstream.width
    outvstream.options = {"preset":"medium","tune":"film","crf":"22"}
    outastream = output.add_stream(template=inastream)
    outastream.options = {}
    
    input.seek(args.seekstart*1000000)

    alignment_dir = Path(args.alignment_dir)
    alignments = alignment_dir / 'alignments.json'
    with alignments.open() as f:
        alignments = json.load(f)
    # alignments.sort()

    # if args.seamless_clone and args.blur_size != 0:
      # print('Setting blur size to zero for seamless_clone')
      # args.blur_size = 0
    if args.seamless_clone != 0 and args.seamless_clone %2 == 0:
      args.blur_size += 1
      
    if args.erosion_kernel_size>0:
      erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.erosion_kernel_size, args.erosion_kernel_size))
    else:
      erosion_kernel = None

    for e in alignments:    
      if len(e)<4:
        raise LookupError('This script expects new format json files with face points included.')
    
    if args.thread_number < 1:
      args.thread_number = 1
      print('Warning: thread_number < 1, set to 1.')
        
    framequeue = queue.Queue(maxsize=8)
    facequeue = queue.Queue(maxsize=8)
    thread_decoder = threading.Thread(target=decoder, args=(alignment_dir, alignments, args.seekstart, args.durationtime, facequeue, framequeue))
    thread_decoder.start()

    queue_convert_face = Queue(maxsize=8)
    thread_convert_face = threading.Thread(target=convert_face, args=(facequeue, queue_convert_face, args.swap_model, args.thread_number, args.double_pass))
    thread_convert_face.start()

    queue_list = []
    thread_list = []
    for i in range(args.thread_number):
        queue_list.append(Queue(maxsize=4))
        thread_list.append(Process(target=merge_face, args=(queue_convert_face, erosion_kernel, args.blur_size, args.seamless_clone, queue_list[i])))
        thread_list[i].start()
        
    encoder(args.thread_number, framequeue, queue_list)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file',
                        dest="input_file",
                        type=str,
                        default="input.mp4",
                        help="Input video")
    parser.add_argument('-o', '--output-file',
                        dest="output_file",
                        type=str,
                        default="output.mp4",
                        help="Output video")
    parser.add_argument('-a', '--alignment-dir',
                        dest="alignment_dir",
                        type=str,
                        default='aligned')
    parser.add_argument('-s', '--swap-model',
                        action="store_true",
                        dest="swap_model",
                        default=False,
                        help="Swap the model. Instead of A -> B, swap B -> A.")
    parser.add_argument('-b', '--blur-size',
                        dest="blur_size",
                        type=int,
                        default=63,
                        help="Blur size. (Masked converter only)")
    parser.add_argument('-S', '--seamless',
                        action="store_true",
                        dest="seamless_clone",
                        default=False,
                        help="Seamless mode. (Masked converter only)")
    parser.add_argument('-M', '--mask-type',
                        type=str.lower, #lowercase this, because its just a string later on.
                        dest="mask_type",
                        choices=["rect", "facehull", "facehullandrect"],
                        default="facehullandrect",
                        help="Mask to use to replace faces. (Masked converter only)")
    parser.add_argument('-e', '--erosion-kernel-size',
                        dest="erosion_kernel_size",
                        type=int,
                        default=10,
                        help="Erosion kernel size. (Masked converter only)")
    parser.add_argument('--double-pass',
                        action="store_true",
                        dest="double_pass",
                        default=False,
                        help="Pass the original prediction output back through for a second pass.")
    parser.add_argument('-pn', '--thread_number',
                        dest="thread_number",
                        type=int,
                        default=3)
    parser.add_argument('-ss', '--seekstart',
                        dest="seekstart",
                        type=int,
                        default=0)
    parser.add_argument('-t', '--durationtime',
                        dest="durationtime",
                        type=int,
                        default=100000)                        
    parser.add_argument('-f', '--format', default='yuv420p')
    parser.add_argument('-c', '--codec', default='libx264')
    main(parser.parse_args())
