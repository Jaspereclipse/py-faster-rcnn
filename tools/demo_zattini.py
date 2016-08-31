#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',)
# synsets = sio.loadmat(os.path.join('data/ILSVRCdevkit2013', 'data', 'meta_det.mat'))
# for i in xrange(200):
#     CLASSES = CLASSES + (synsets['synsets'][0][i][2][0],)

with open(os.path.join('data', 'REIMUdevkit2016', 'descriptions','notes.txt'), 'r') as f:
    synsets = f.readlines() # (wnid, class)
synsets = [(l.split(" - ")[0], l.strip().split(" - ")[1]) for l in synsets]
for _, class_ in synsets:
    CLASSES = CLASSES + (class_,)

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def vis_detections(im, dets_classes, image_name, thresh=0.5):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    save = False # True if at least one class has bbox >= thres
    for class_name, dets in dets_classes.iteritems():
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue
        save = True
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

    ax.set_title(('Detections-Classification with '
                  'thres >= {:.1f}').format(thresh), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    
    # Save
    if not save:
        return 
    output_path = os.path.join("output", "frame_analysis", "reimu_2016_VGG16_70000_v4")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fig.savefig(os.path.join(output_path, image_name))

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'zattini_frames', 'zattini_4', image_name)
    assert os.path.isfile(im_file)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    det_cls_dict = {}
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        det_cls_dict[cls] = dets
    vis_detections(im, det_cls_dict, image_name, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo for zattini')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models/reimu/VGG16/faster_rcnn_end2end/test.prototxt')
    caffemodel = './output/faster_rcnn_end2end/reimu_2016_train/vgg16_faster_rcnn_iter_70000.caffemodel'
    
    data_path = "./data/zattini_frames/zattini_4"
    
    assert os.path.exists(data_path)
    assert os.path.isfile(caffemodel)

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
        
    im_names = [name for name in os.listdir(data_path) if name.endswith(".jpg")]
    
#    im_names = ['demo_01.jpg', 'demo_02.jpg', 'demo_03.jpg', '000456.jpg', '000542.jpg', '001150.jpg', '001763.jpg', '004545.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        demo(net, im_name)

