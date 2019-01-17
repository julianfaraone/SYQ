#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: syq-alexnet.py
# Author: Julian Faraone (julian.faraone@sydney.edu.au)

import cv2
import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
import msgpack
import os, sys

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from quantize import *
from tensorpack.utils.stats import RatioCounter

TOTAL_BATCH_SIZE = 32
INITIAL = True

BITA = 8
FRAC = 4
PATH = ''

#Enter Path to floatingpoint_alexnet.npy downloaded from Google Drive
PATH_float = 'floatingpoint_alexnet.npy'

if INITIAL:
    d = np.load(PATH_float).item()

    weights = {}

    #calculate initialization for scaling coefficients
    for i in d.keys():
        if '/W:' in i and 'conv' in i:
            mean = np.mean(np.absolute(d[i]), axis = (2,3))
            weights[i] = mean
        elif '/W:' in i and 'fc' in i:
            mean = np.mean(np.absolute(d[i]))
            weights[i] = mean
else:
    weights = None

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 224, 224, 3], 'input'),
                InputVar(tf.int32, [None], 'label') ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 255.0

        # monkey-patch tf.get_variable to apply fw
        old_get_variable = tf.get_variable
        def new_get_variable(name, shape=None, **kwargs):
            v = old_get_variable(name, shape, **kwargs)
        
            # don't binarize first and last layer
            if name != 'W' or 'conv0' in v.op.name or 'fct' in v.op.name:
                return v
            else:
                return fine_grained_quant(v, args.eta, v.op.name, INITIAL, weights)
        tf.get_variable = new_get_variable

        def activate(x):
            x = tf.nn.relu(x)
            x = tf.clip_by_value(x,0,1)
            x = quantize(x, BITA, None)
            return x

        with argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity):
            logits = (LinearWrap(image)
                .Conv2D('conv0', 96, 12, stride=4, padding='VALID')
                .apply(activate)
                .Conv2D('conv1', 256, 5, padding='SAME', split=2)
                .BatchNorm('bn1')
                .MaxPooling('pool1', 3, 2, padding='SAME')
                .apply(activate)

                .Conv2D('conv2', 384, 3)
                .BatchNorm('bn2')
                .MaxPooling('pool2', 3, 2, padding='SAME')
                .apply(activate)

                .Conv2D('conv3', 384, 3, split=2)
                .BatchNorm('bn3')
                .apply(activate)

                .Conv2D('conv4', 256, 3, split=2)
                .BatchNorm('bn4')
                .MaxPooling('pool4', 3, 2, padding='VALID')
                .apply(activate)

                .FullyConnected('fc0', 4096)
                .BatchNorm('bnfc0')
                .apply(activate)

                .FullyConnected('fc1', 4096)
                .BatchNorm('bnfc1')
                .apply(activate)
                .FullyConnected('fct', 1000, use_bias=True)())
        tf.get_variable = old_get_variable

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))
        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(5e-6))
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram', 'rms'])])
        self.cost = tf.add_n([cost, wd_cost], name='cost')

def get_data(dataset_name):
    isTrain = dataset_name == 'train'
    ds = dataset.ILSVRC12(args.data, dataset_name, shuffle=isTrain)

    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16,16:-16,:]

    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            def __init__(self):
                self._init(locals())
            def _augment(self, img, _):
                h, w = img.shape[:2]
                size = 224
                scale = self.rng.randint(size, 308) * 1.0 / min(h, w)
                scaleX = scale * self.rng.uniform(0.85, 1.15)
                scaleY = scale * self.rng.uniform(0.85, 1.15)
                desSize = map(int, (max(size, min(w, scaleX * w)),\
                    max(size, min(h, scaleY * h))))
                dst = cv2.resize(img, tuple(desSize),
                     interpolation=cv2.INTER_CUBIC)
                return dst

        augmentors = [
            Resize(),
            imgaug.RandomCrop((224, 224)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    else:
        def resize_func(im):
            h, w = im.shape[:2]
            scale = 256.0 / min(h, w)
            desSize = map(int, (max(224, min(w, scale * w)),\
                                max(224, min(h, scale * h))))
            im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
            return im
        augmentors = [
            imgaug.MapImage(resize_func),
            imgaug.CenterCrop((224, 224)),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(12, multiprocessing.cpu_count()))
    return ds

def get_config(learning_rate, num_epochs, inf_epochs):
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__)
    logdir = os.path.join(PATH, '{}'.format(args.name))

    logger.set_logger_dir(logdir)

    import shutil

    shutil.copy(mod.__file__, logger.LOG_DIR)

    # prepare dataset
    data_train = get_data('train')
    data_test = get_data('val')

    lr = get_scalar_var('learning_rate', learning_rate[0], summary=True)

   # lr = tf.Variable(4e-6, trainable=False, name='learning_rate')
    #tf.scalar_summary('learning_rate', lr)

    total_epochs = np.arange(1, (num_epochs[-1] + 1))
    do_epochs = np.append(inf_epochs, total_epochs[num_epochs[-2]:])

    return TrainConfig(
        dataset=data_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-5),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            #HumanHyperParamSetter('learning_rate'),
            ScheduledHyperParamSetter(
                'learning_rate', zip(num_epochs[:-1], learning_rate[1:])),
            InferenceRunner(data_test,
                [ScalarStats('cost'),
                 ClassificationError('wrong-top1', 'val-error-top1'),
                 ClassificationError('wrong-top5', 'val-error-top5')], do_epochs)
        ]),
        model=Model(),
        step_per_epoch=10000,
        max_epoch=num_epochs[-1],
    )

def run_image(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        session_config=get_default_sess_config(0.9),
        input_var_names=['input'],
        output_var_names=['output']
    )
    predict_func = get_predict_func(pred_config)
    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16,16:-16,:]
    words = meta.get_synset_words_1000()

    def resize_func(im):
        h, w = im.shape[:2]
        scale = 256.0 / min(h, w)
        desSize = map(int, (max(224, min(w, scale * w)),\
                            max(224, min(h, scale * h))))
        im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
        return im
    transformers = imgaug.AugmentorList([
        imgaug.MapImage(resize_func),
        imgaug.CenterCrop((224, 224)),
        imgaug.MapImage(lambda x: x - pp_mean_224),
    ])
    for f in inputs:
        assert os.path.isfile(f)
        img = cv2.imread(f).astype('float32')
        assert img is not None

        img = transformers.augment(img)[np.newaxis, :,:,:]
        outputs = predict_func([img])[0]
        prob = outputs[0]
        ret = prob.argsort()[-10:][::-1]

        names = [words[i] for i in ret]
        print(f + ":")
        print(list(zip(names, prob[ret])))

def eval_on_ILSVRC12(model_path, data_dir, ds_type):
    ds = get_data(ds_type)
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for o in pred.get_result():  
        batch_size = o[0].shape[0]
        acc1.feed(o[0].sum(), batch_size)
        acc5.feed(o[1].sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load a checkpoint, or a npy (given as the pretrained model)')
    parser.add_argument('--data', help='ILSVRC dataset dir', default='/ssd/dataset/imagenet')
    parser.add_argument('--run', help='run on a list of images with the pretrained model', nargs='*')
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--learning-rate', type=float, nargs='+', metavar='LR', default=[1e-4, 2e-5, 4e-6],
            help='Learning rates to use during training, first value is the initial learning rate (default: %(default)s). Must have the same number of args as --num-epochs')
    parser.add_argument('--num-epochs', type=int, nargs='+', metavar='E', default=[56, 64, 100],
            help='Epochs to change the learning rate, last value is the maximum number of epochs (default: %(default)s). Must have the same number of args as --learning-rate')
    parser.add_argument('--inf-epochs', type=int, nargs='+', metavar='I', default=list(np.arange(1,121)))
    parser.add_argument('--eval', type=str, default=None, choices=['val', 'test'],
            help='evaluate the model on the test of validation set')
    parser.add_argument('--name', default='train1')

    args = parser.parse_args()

    if args.eval != None:
        eval_on_ILSVRC12(args.load, args.data, args.eval)
        sys.exit()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.run:
        assert args.load.endswith('.npy')
        run_image(Model(), ParamRestore(np.load(args.load, encoding='latin1').item()), args.run)
        sys.exit()

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU

    assert len(args.num_epochs) == len(args.learning_rate)
    config = get_config(args.learning_rate, args.num_epochs, args.inf_epochs)

    if args.load:
        if args.load.endswith('.npy'):
            config.session_init = ParamRestore(np.load(args.load, encoding='latin1').item())
        else:
            config.session_init = SaverRestore(args.load)

    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
