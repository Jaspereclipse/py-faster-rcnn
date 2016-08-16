# Training Faster RCNN on ILSVRC data

1. This is a forked version of [Faster RCNN](https://github.com/rbgirshick/py-faster-rcnn). Please refer to the original [README.md](https://github.com/rbgirshick/py-faster-rcnn/blob/master/README.md) file for more information.

2. The implementation of the code was an attempt to reproduce [andrewliao11](https://github.com/andrewliao11/py-faster-rcnn-imagenet)'s work on ImageNet's dataset.

3. The code was run by AWS's [g2.2xlarge](https://aws.amazon.com/ec2/instance-types/) instance. 

## Preparing Data
### Downloading
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
### Subsetting
Next I wrote a small python script (require sklearn) to shuffle split the ```val.txt``` into ```val_train.txt``` and ```val_test.txt``` (test size is 0.25), which reside in the same directory as ```val.txt```.

```
cd $FRCN_ROOT
python ./tools/shuffle_split.py --des ./data/ILSVRC13/data/det_lists/val.txt
```
