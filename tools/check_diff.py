#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN Forked on ImageNet Dataset
# Written by Juanyan Li
# --------------------------------------------------------

import os


image_names = [f.split('.')[0] for f in os.listdir('./data/ILSVRCdevkit2013/ILSVRC2013_DET_val')]

with open('./data/ILSVRCdevkit2013/data/det_lists/val.txt', 'r') as f:
	val_names = f.readlines()
	val_names = [name.split(' ')[0] for name in val_names]

diff = list(set(image_names) - set(val_names))
print diff