# Training Faster RCNN on ILSVRC data

1. This is a forked version of [Faster RCNN](https://github.com/rbgirshick/py-faster-rcnn). Please refer to the original [README.md](https://github.com/rbgirshick/py-faster-rcnn/blob/master/README.md) file for more information.

2. The implementation of the code was an attempt to reproduce [andrewliao11](https://github.com/andrewliao11/py-faster-rcnn-imagenet)'s work on ImageNet's dataset.

3. The code was run by AWS's [g2.2xlarge](https://aws.amazon.com/ec2/instance-types/) instance. 

## Prepare Data
### Download
I'm using the ILSVRC 2013 Validation set (~2.7 GB, you can download images data, annotations and devkit [here](http://image-net.org/challenges/LSVRC/2014/download-images-5jj5.php)).

My organization of the database has the following structure (I saved it to ~/):
```
ILSVRC13 
└─── LSVRC2013_DET_val
    │   *.JPEG (e.g. ILSVRC2012_val_00000001.JPEG)
└─── data
    │   meta_det.mat
    └─── det_lists
             │  *.txt (e.g. val.txt)
└─── Annotations
    │   *.xml (e.g. ILSVRC2012_val_00000001.xml)
```

It is convenient to create a symbolic link for the code to refer to the data (assuming $FRCN_ROOT is directory of the repo for convention, e.g. blablabla/py-faster-rcnn):

```
cd $FRCN_ROOT
ln -s ~/ILSVRC13 ./data/ILSVRCdevkit2013
```
### Subset
Next I wrote a small python script ([sklearn](http://scikit-learn.org/stable/) needed) to shuffle split the ```val.txt``` into ```val_train.txt``` and ```val_test.txt``` (test size is 0.25), which reside in the same directory as ```val.txt```.

```
cd $FRCN_ROOT
python ./tools/shuffle_split.py --des ./data/ILSVRC13/data/det_lists/val.txt
```
## Modify Scripts
Most of the changes are related to data import (creating roidb for RPN). Some have to deal with prototxt files.
Please refer to the corresponding file(s) for details.

1. Add a new path for ILSVRC in ```faster_rcnn_end2end.sh```
2. Edit ```./lib/datasets/factory.py``` to pass the correct arguments into ```ilsvrc``` object
3. Create a new class ```ilsvrc.py``` resembling ```pascal_voc.py```
    * ```__init__()```
        * read class from meta_det.mat and index them (see [this](https://github.com/andrewliao11/py-faster-rcnn-imagenet/blob/master/README.md))
        * change suffix to .JPEG
        * comment 'use_diff' (no such thing in ImageNet annotations)
    * ```_get_default_path()```
        * change to your devkit path (symbolic link)
    * ```_load_image_set_index()```
        * change path to your val_train.txt/val_test.txt
    * ```_load_ilsvrc_annotation()``` (changed from ```_load_pascal_annotation()```)
        * point it to the annotations folder
        * comment out the 'use_diff' part
        * change pixel index (see [this](https://github.com/andrewliao11/py-faster-rcnn-imagenet/blob/master/README.md))
