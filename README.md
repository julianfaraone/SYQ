### SYQ Training
This repository represents training examples for the CVPR 2018 paper "SYQ:Learning Symmetric Quantization For Efficient Deep Neural Networks"

##Tested Platform - Dependencies
Python 2.7 or 3
Python bindings for OpenCV
Tensorflow == 0.12.1 (pip install tensorflow-gpu==0.12.1)

#Add tensorpack to python path:
`export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack``

If caffe.proto error occurs:
wget ''

## Usage

# To train a binarized AlexNet with 8-bit actiavtions, use:
    
`python syq-imagenet-alexnet.py --data <enter path to data> --num-epochs 56 84 120 --learning-rate 1e-4 2e-5 4e-6 --load <enter checkpoint or npy path>`
