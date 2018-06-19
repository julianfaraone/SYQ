# SYQ Training
This repository represents training examples for the CVPR 2018 paper "SYQ: Learning Symmetric Quantization For Efficient Deep Neural Networks"

## Tested Platform - Dependencies
Python 2.7 or 3 <br />
Python bindings for OpenCV <br />
Tensorflow == 0.12.1 (pip install tensorflow-gpu==0.12.1) 

### Add tensorpack to python path:
export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack`

## Usage

### Command to train a binarized AlexNet with 8-bit actiavtions, use:
    
`python syq-alexnet.py --data <enter path to data> --num-epochs 112 168 240 --learning-rate 1e-4 2e-5 4e-6 --load <npy path> --eta 0.0 --gpu 0`

### Continue training from latest epoch:
    
`python syq-alexnet.py --data <enter path to data> --num-epochs 112 168 240 --learning-rate 1e-4 2e-5 4e-6 --eta 0.0 --load <PATH>/checkpoint --gpu 0`

Ensure INITIAL = False and weights = None in syq-alexnet.py

## Parameters

--gpu >Set which gpu's you want to instantiate (example: --gpu 0,1) <br />
--load >Load either npy or checkpoint file as a pretrained model by entering its path <br />
--data >Path to training (train)  and validation (val) data folders  <br />
--run >Enter image files along with a pretrained models to compute inference <br />
--eta >Quantization Threshold value (eta=0 for binary networks and defaults to 0.05 for ternary) <br />
--learning-rate >enter a list of learning rate factors at each step <br />
--num-epochs >Enter the epochs when the learning rate changes, the last value is the total epochs <br />
--in-epochs >Enter epochs which will compute the validation error. It automatically compute all epochs after the last <br />                learning rate change. Default is computing validation error for all epochs <br />
--eval >Evaluate the model on test or validation set <br />
--name >Sets the name of the folder for storing training results <br />

## Initial Training

For faster training (less required epochs), we recommend using the pre-trained floating point weights for AlexNet which can be found at <br />
<br />
https://drive.google.com/open?id=1Saa9kADmWhS5f_91aW83r8UwHQf_UY6Z <br />
Download this model and set the variable PATH_float in syq-alexnet.py to its path <br />

Ensure INITIAL = True in syq-alexnet.py

## Training Results

Set path of training results storage as PATH in syq-alexnet <br />
<br />
Enter directory name which training data will be stored in using the --name parameter 
