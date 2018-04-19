# SYQ Training
This repository represents training examples for the CVPR 2018 paper "SYQ:Learning Symmetric Quantization For Efficient Deep Neural Networks"

## Tested Platform - Dependencies
Python 2.7 or 3
Python bindings for OpenCV
Tensorflow == 0.12.1 (pip install tensorflow-gpu==0.12.1)

### Add tensorpack to python path:
`export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack``

## Usage

### Command to train a binarized AlexNet with 8-bit actiavtions, use:
    
`python syq-imagenet-alexnet.py --data <enter path to data> --num-epochs 56 84 120 --learning-rate 1e-4 2e-5 4e-6 --load <enter checkpoint or npy path> --eta 0.0`

### Continue training from latest epoch:
    
`python syq-imagenet-alexnet.py --data <enter path to data> --num-epochs 56 84 120 --learning-rate 1e-4 2e-5 4e-6 --load <enter checkpoint or npy path> --eta 0.0 --load <PATH>/checkpoint`

## Parameters

--gpu >Set which gpu's you want to instantiate (example: --gpu 0,1) <br />
--load >Load either npy or checkpoint file as a pretrained model by entering its path <br />
--data >Path to training and validation data <br />
--run >Enter image files along with a pretrained models to compute inference <br />
--eta >Quantization Threshold value (eta=0 for binary networks and defaults to 0.05 for ternary) <br />
--learning-rate >enter a list of learning rate factors at each step <br />
--num-epochs >Enter the epochs when the learning rate changes, the last value is the total epochs <br />
--in-epochs >Enter epochs which will compute the validation error. It automatically compute all epochs after the last <br />                learning rate change. Default is computing validation error for all epochs <br />
--eval >Evaluate the model on test or validation set <br />
--name >Sets the name of the folder for storing training results <br />
