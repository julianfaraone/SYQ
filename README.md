# SYQ
This repository represents training examples for the CVPR 2018 paper "SYQ:Learning Symmetric Quantization For Efficient Deep Neural Networks"

Must firstly install Opencv bindings for python
Tensorflow >= 1.4

To train a binarized AlexNet with 8-bit actiavtions, use:

python syq-imagenet-alexnet.py --data <enter path to data> --num-epochs 56 84 120 --learning-rate 1e-4 2e-5 4e-6 --load <enter checkpoint or npy path>
