from src import detect_faces
from PIL import Image, ImageDraw

import argparse
import cv2
import numpy as np
import os
import torch
import thread

from matlab_cp2tform import get_similarity_transform_for_cv2

def alignment(src_img, src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5, 2)
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)
    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

def process(input_dir, output_dir):
    for i in os.listdir(input_dir):
        file = os.path.join(input_dir, i)
        if os.path.isfile(file):
            try:
                image = Image.open(file)
            except:
                pass
            else:
                print(' -  Detecting image ' + file + ' ... ', end='', flush=True)
                boxes, marks = detect_faces(image)
                if len(boxes) != 1:
                    print('failed .', flush=True)
                    continue
                np_arr = np.fromfile(file, np.uint8)
                landmark0 = [[marks[0][i], marks[0][i + 5]] for i in range(5)]
                aligned_image = alignment(cv2.imdecode(np_arr, 1), landmark0)
                cv2.imwrite(os.path.join(output_dir, i), aligned_image)
                print('done !', flush=True)
    return

parser = argparse.ArgumentParser(description='Face landmark detector, Tsinghua University, ASC 19')
parser.add_argument('--output', type=str, default='', help='Mark landmarks on images, option for output directory')
parser.add_argument('--input', type=str, default='', help='Input directory')

options = parser.parse_args()
print('[!] Processing images ... ', flush=False)
process(options.input, options.output)
print('[!] done !')
            
    