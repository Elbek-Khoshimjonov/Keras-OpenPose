import tensorflow as tf
import cv2
import numpy as np
import re
import math

import os
from os import path

from utils import *
from model import *

from keras.applications.vgg19 import VGG19
from keras.layers import Conv2D
import keras.backend as K

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, TerminateOnNaN

from argparse import ArgumentParser

from tqdm import tqdm

from glob import glob


parser = ArgumentParser()


parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

parser.add_argument("--base_lr", type=float, default=2e-5, help="Keypoints count")

parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")

parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay property")

parser.add_argument("--lr_policy", type=str, default="step")

parser.add_argument("--gamma", type=float, default=0.333)

parser.add_argument("--stepsize", type=int, default=120052*17) # 120052 and step change is on 17th epoch

parser.add_argument("--max_iter", type=int, default=200)

parser.add_argument("--use_multiple_gpus", type=int, default=4, help="Use multiple gpus 1, None for 1 gpu")

parser.add_argument("--training_log", type=str, default="weights/v1/training.csv", help="Training log csv file")

parser.add_argument("--weight_dir", type=str, default="weights/v1", help="Folder to save weights")

parser.add_argument("--weight_save", type=str, default="weights.{epoch:04d}.h5", help="Weights file template")

parser.add_argument("--log_dir", type=str, default="log/v1", help="Tensorboard logdir")

parser.add_argument("--train_dataset", type=str, default="dataset/train_dataset_2017.h5", help="Train Dataset path")

parser.add_argument("--val_dataset", type=str, default="dataset/val_dataset_2017.h5", help="Validation Dataset path")


args = parser.parse_args()


# Init model
model = get_training_model(args.weight_decay)

## Make log_dir
if not path.exists(args.log_dir):
    os.makedirs(args.log_dir)

## Get last epoch and weights 
def get_last_epoch_and_weights_file():
    os.makedirs(args.weight_dir, exist_ok=True)
    # os.makedirs(args.weight_dir)
    files = [file for file in glob(args.weight_dir + '/weights.*.h5')]
    files = [file.split('/')[-1] for file in files]
    epochs = [file.split('.')[1] for file in files if file]
    epochs = [int(epoch) for epoch in epochs if epoch.isdigit() ]
    if len(epochs) == 0:
        if 'weights.best.h5' in files:
            return -1, args.weight_dir + '/weights.best.h5'
    else:
        ep = max([int(epoch) for epoch in epochs])
        return ep, args.weight_dir + '/' + args.weight_save.format(epoch=ep)
    return None, None



from_vgg = dict()
from_vgg['conv1_1'] = 'block1_conv1'
from_vgg['conv1_2'] = 'block1_conv2'
from_vgg['conv2_1'] = 'block2_conv1'
from_vgg['conv2_2'] = 'block2_conv2'
from_vgg['conv3_1'] = 'block3_conv1'
from_vgg['conv3_2'] = 'block3_conv2'
from_vgg['conv3_3'] = 'block3_conv3'
from_vgg['conv3_4'] = 'block3_conv4'
from_vgg['conv4_1'] = 'block4_conv1'
from_vgg['conv4_2'] = 'block4_conv2'

# load previous weights or vgg19 if this is the first run
last_epoch, wfile = get_last_epoch_and_weights_file()
if wfile is not None:
    print("Loading %s ..." % wfile)

    model.load_weights(wfile)
    last_epoch = last_epoch + 1

else:
    print("Loading vgg19 weights...")

    vgg_model = VGG19(include_top=False, weights='imagenet')

    for layer in model.layers:
        if layer.name in from_vgg:
            vgg_layer_name = from_vgg[layer.name]
            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
            print("Loaded VGG19 layer: " + vgg_layer_name)

    last_epoch = 0



# setup lr multipliers for conv layers
lr_mult=dict()
for layer in model.layers:

    if isinstance(layer, Conv2D):

        # stage = 1
        if re.match("Mconv\d_stage1.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 1
            lr_mult[bias_name] = 2

        # stage > 1
        elif re.match("Mconv\d_stage.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 4
            lr_mult[bias_name] = 8

        # vgg
        else:
           kernel_name = layer.weights[0].name
           bias_name = layer.weights[1].name
           lr_mult[kernel_name] = 1
           lr_mult[bias_name] = 2

# configure loss functions

# euclidean loss as implemented in caffe https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
def eucl_loss(x, y):
    l = K.sum(K.square(x - y)) / args.batch_size / 2
    return l

# prepare generators
train_client = DataIterator(args.train_dataset, shuffle=True, augment=True, batch_size=args.batch_size)
val_client = DataIterator(args.val_dataset, shuffle=False, augment=False, batch_size=args.batch_size)

train_di = train_client.gen()
train_samples = 120052
val_di = val_client.gen()
val_samples = 4798


# learning rate schedule - equivalent of caffe lr_policy =  "step"
iterations_per_epoch = train_samples // args.batch_size

def step_decay(epoch):
    steps = epoch * iterations_per_epoch * args.batch_size
    lrate = args.base_lr * math.pow(args.gamma, math.floor(steps/args.stepsize))
    print("Epoch:", epoch, "Learning rate:", lrate)
    return lrate

print("Weight decay policy...")
for i in range(1,100,5): step_decay(i)

# configure callbacks
lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(args.weight_dir + '/' + args.weight_save, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=1)
csv_logger = CSVLogger(args.training_log, append=True)
tb = TensorBoard(log_dir=args.log_dir, histogram_freq=0, write_graph=True, write_images=False)
tnan = TerminateOnNaN()

callbacks_list = [lrate, checkpoint, csv_logger, tb, tnan]

# sgd optimizer with lr multipliers
multisgd = MultiSGD(lr=args.base_lr, momentum=args.momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)

# start training

if args.use_multiple_gpus is not None:
    from keras.utils import multi_gpu_model
    model = multi_gpu_model(model, gpus=args.use_multiple_gpus)

model.compile(loss=eucl_loss, optimizer=multisgd)


model.fit_generator(train_di,
                    steps_per_epoch=iterations_per_epoch,
                    epochs=args.max_iter,
                    callbacks=callbacks_list,
                    validation_data=val_di,
                    validation_steps=val_samples // args.batch_size,
                    use_multiprocessing=False,
                    initial_epoch=last_epoch
                    )
