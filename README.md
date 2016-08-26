# Training Faster RCNN on ILSVRC data

1. This is a forked version of [Faster RCNN](https://github.com/rbgirshick/py-faster-rcnn). Please refer to the original [README.md](https://github.com/rbgirshick/py-faster-rcnn/blob/master/README.md) file for more information.

2. The implementation of the code was an attempt to reproduce [andrewliao11](https://github.com/andrewliao11/py-faster-rcnn-imagenet)'s work on ImageNet's dataset.

3. The code ran on a AWS's [g2.2xlarge](https://aws.amazon.com/ec2/instance-types/) instance (~2.8s/iter) and later configured to run locally on a desktop with GTX 1070 (CUDA 8.0, cuDNNv5, ~0.45s/iter)

## Configuration for GTX 1070 on Ubuntu 16.04
It took some time to get it working on my desktop (GTX 1070, CUDA 8.0, cuDNNv5, gcc 5.4). Some key steps:

* Install newest NVIDIA driver: I find this [post](http://www.allaboutlinux.eu/remove-nouveau-and-install-nvidia-driver-in-ubuntu-15-04/) to be rather useful. 
* Install CUDA 8.0: run with ```--override``` to suppress compiler error (gcc); [official documentation](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz4ITZNE9gb)

```
./cuda_8.0.27_linux.run --override
```
* Install cuDNNv5: this [post](http://askubuntu.com/questions/767269/how-can-i-install-cudnn-on-ubuntu-16-04) might help
* Suppress compiler error (again): simply a hack
```
vi /usr/local/cuda/include/host_config.h
```
comment the following line (L115):
```
//#error -- unsupported GNU version! gcc versions later than 5.3 are not supported!
```
* Modify Caffe: [#237](https://github.com/rbgirshick/py-faster-rcnn/issues/237)
* Other issues: some of the required libraries like ```protobuf``` are compiled in older version of ```gcc``` when you use ```apt-get install``` to install them. This might cause errors later. To get around with this, download the corresponding repo from github and compile it from src.

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

## Modify Prototxt
Here I will use the orginal implementation of [Faster RCNN](https://github.com/rbgirshick/py-faster-rcnn) to illustrate the changes.

* ```solver.prototxt```
    * Change [this](https://github.com/rbgirshick/py-faster-rcnn) to the correct ```train.prototxt``` directory
* ```train.prototxt```
    * Change [this](https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt#L11), [this](https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt#L530) and [this](https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt#L620) to the correct number of classes (200 + 1 = 201 in my case)
    * Change [this](https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt#L643) to correct number of bboxes ((200 + 1) * 4 = 804 in my case)
* ```test.prototxt```
    * Similarly, change [this](https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt#L567) to 201 and [this](https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt#L592) to 804

## Train/test the model
To run the end-to-end training:

```
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 ilsvrc --set RNG_SEED 42
```

## Use Caffe's snapshot
After running ~28,000 iterations on AWS, the problem stopped somehow (maybe I did something). And the implemented snapshot method in Python has no way to restore the training state, meaning I have to run it again. To prevent similar things from happening again, I decided to use the Caffe's original snapshot.

The fix refers to the solution to [Issue#35](http://stackoverflow.com/questions/8773299/how-to-cut-an-entire-line-in-vim-and-paste-it). 

* ```$FRCN_ROOT/tools/train_net.py```
    * [Modification #1](https://github.com/Jaspereclipse/py-faster-rcnn/blob/master/tools/train_net.py#L40-L42)
    * [Modification #2](https://github.com/Jaspereclipse/py-faster-rcnn/blob/master/tools/train_net.py#L113-L116)
*  ```$FRCN_ROOT/lib/fast_rcnn/train.py```
    * [Modification #3](https://github.com/Jaspereclipse/py-faster-rcnn/blob/master/lib/fast_rcnn/train.py#L26-L27)
    * [Modification #4](https://github.com/Jaspereclipse/py-faster-rcnn/blob/master/lib/fast_rcnn/train.py#L44-L51)
    * [Modification #5](https://github.com/Jaspereclipse/py-faster-rcnn/blob/master/lib/fast_rcnn/train.py#L155-L161)
*  ```$FRCN_ROOT/models/ilsvrc/VGG16/faster_rcnn_end2end/solver.prototxt```
    * [Modification #6](https://github.com/Jaspereclipse/py-faster-rcnn/blob/master/models/ilsvrc/VGG16/faster_rcnn_end2end/solver.prototxt#L13)
*  ```$FRCN_ROOT/experiments/scripts/faster_rcnn_end2end.sh```
    * [Modification #7](https://github.com/Jaspereclipse/py-faster-rcnn/blob/master/experiments/scripts/faster_rcnn_end2end.sh#L64) (comment the ```--weights``` and uncomment this line if you want to restore previous state)

**Warning:** Don't use the ```.caffemodel``` ported directly by Caffe to test the results. Use the one output by [snapshot()](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/train.py#L55) (the one stored in ./output/faster_rcnn_end2end/ilsvrc_2013_det_val_train)

## Results
To be continued...
