# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:42:15 2016

Visualize original bounding boxes

@author: Juanyan Li
"""

import os, sys
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN trainset demo')
    parser.add_argument('--db', dest='imdb',
                        help='Dataset to be visualized {pascal, ilsvrc}',
                        default=None, type=str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def vis_bbox(im, boxes, image_name):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    for ix, _ in enumerate(boxes):
        bbox = boxes[ix, :]
        ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                       bbox[2] - bbox[0],
                       bbox[3] - bbox[1], fill=False,
                       edgecolor='red', linewidth=3.5))
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

if __name__ == '__main__':
    
    args = parse_args()
    
    bbox_suffix = "xml"    
    
    if args.imdb == "pascal":
        devkit_path = "./data/VOCdevkit2007/VOC2007"
        db = os.path.join(devkit_path, "JPEGImages")
        at = os.path.join(devkit_path, "Annotations")
        ref = os.path.join(devkit_path, "ImageSets", "Main", "val.txt") # image names
        im_suffix = "jpg"
        parser = (lambda c, suf: ".".join([c.strip(), suf]))
        
    elif args.imdb == "ilsvrc":
        devkit_path = "./data/ILSVRCdevkit2013"
        db = os.path.join(devkit_path, "ILSVRC2013_DET_val")
        at = os.path.join(devkit_path, "Annotations")
        ref = os.path.join(devkit_path, "data", "det_lists", "val.txt") # image names
        im_suffix = "JPEG"
        parser = (lambda c, suf: ".".join([c.split(" ")[0], suf]))

    assert os.path.isfile(ref)  
    
    with open(ref, 'r') as f:
        content = f.readlines()
        im_files = np.array([parser(c, im_suffix) for c in content])
        bbox_files = np.array([parser(c, bbox_suffix) for c in content])
    
    # num_samples = 10
    # choices = np.random.choice(len(im_files), num_samples)
    # im_samples = im_files[choices]
    # bbox_samples = bbox_files[choices]
    num_samples = 1
    im_samples = ["000005.jpg"]
    bbox_samples = ["000005.xml"]


    for i in xrange(num_samples):
        # images
        im = cv2.imread(os.path.join(db, im_samples[i]))
        # bounding boxes        
        tree = ET.parse(os.path.join(at, bbox_samples[i]))
        objs = tree.findall('object')
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            boxes[ix, :] = [x1, y1, x2, y2]
        vis_bbox(im, boxes, im_samples[i])
    plt.show()