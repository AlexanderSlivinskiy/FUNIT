"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import os
import sys
import argparse
import shutil
from torchsummary import summary

from tensorboardX import SummaryWriter

from utils import get_config, get_train_loaders, make_result_folders, get_train_loaders_custom
from utils import write_loss, write_html, write_1images, Timer, make_log_folder
from trainer import Trainer
from globalConstants import GlobalConstants
from blocks import AdaptiveInstanceNorm2d
from torch.nn import BatchNorm1d, BatchNorm2d

import torch.backends.cudnn as cudnn
os.environ['CUDA_VISIBLE_DEVICES'] = '1' #"0,1" for the hard stuff, "2" for your everyday bread and butter
# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True

#USE FOLLOWING COMMAND TO EXECUTE: 
#python train.py --config configs/funit_confs_custom.yaml --output_path ../../../scratch/slivinskiy/new_datasets/outputs/GPU0

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='configs/funit_animals.yaml',
                    help='configuration file for training and testing')
parser.add_argument('--output_path',
                    type=str,
                    default='.',
                    help="outputs path")
parser.add_argument('--multigpus',
                    action="store_true")
parser.add_argument('--batch_size',
                    type=int,
                    default=0)
parser.add_argument('--test_batch_size',
                    type=int,
                    default=4)
parser.add_argument("--resume",
                    type = str,
                    default="")

opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
# Override the batch size if specified.
if opts.batch_size != 0:
    config['batch_size'] = opts.batch_size

GlobalConstants.setPrecision(config['precision'])
GlobalConstants.setInputOutputChannels(config['gen']['input_nc'], config['gen']['output_nc'])
GlobalConstants.setOptimizer(config['optimizer'])

trainer = Trainer(config)
trainer.cuda()
if opts.multigpus:
    ngpus = torch.cuda.device_count()
    config['gpus'] = ngpus
    print("Number of GPUs: %d" % ngpus)
    trainer.model = torch.nn.DataParallel(
        trainer.model, device_ids=range(ngpus))
else:
    config['gpus'] = 1

loaders = get_train_loaders_custom(config)
train_content_loader = loaders[0]
train_class_loader = loaders[1]
test_content_loader = loaders[2]
test_class_loader = loaders[3]

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
logs_path = make_log_folder("./")
train_writer = SummaryWriter(
    os.path.join(logs_path, model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
GlobalConstants.setOutputPath(output_directory)
checkpoint_directory, image_directory = make_result_folders(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

resume_directory = opts.resume

iterations = trainer.resume(resume_directory,
                            hp=config,
                            multigpus=opts.multigpus) if opts.resume != "" else 0


if (GlobalConstants.getPrecision() == torch.float16 and (not GlobalConstants.usingApex)):
    trainer.model.half()  # convert to half precision
    for layer in trainer.model.modules():
        if isinstance(layer, AdaptiveInstanceNorm2d):
            layer.float()
        elif isinstance(layer, BatchNorm2d):
            layer.float()


#trainer.summary(None)
while True:
    for it, (co_data, cl_data) in enumerate(
            zip(train_content_loader, train_class_loader)):
        with Timer("Elapsed time in update: %f"):
            #torch.autograd.set_detect_anomaly(True)
            d_acc = trainer.dis_update(co_data, cl_data, config, it)
            g_acc = trainer.gen_update(co_data, cl_data, config,
                                       opts.multigpus, it)
            torch.cuda.synchronize()
            print('D acc: %.4f\t G acc: %.4f' % (d_acc, g_acc))

        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        if ((iterations + 1) % config['image_save_iter'] == 0 or (
                iterations + 1) % config['image_display_iter'] == 0):
            if (iterations + 1) % config['image_save_iter'] == 0:
                key_str = '%08d' % (iterations + 1)
                write_html(output_directory + "/index.html", iterations + 1,
                           config['image_save_iter'], 'images')
            else:
                key_str = 'current'
            with torch.no_grad():
                for t, (val_co_data, val_cl_data) in enumerate(
                        zip(train_content_loader, train_class_loader)):
                    if t >= opts.test_batch_size:
                        break
                    val_image_outputs = trainer.test(val_co_data, val_cl_data,
                                                     opts.multigpus)
                    write_1images(val_image_outputs, image_directory,
                                  'train_%s_%02d' % (key_str, t))
                for t, (test_co_data, test_cl_data) in enumerate(
                            zip(test_content_loader, test_class_loader)):
                    if t >= opts.test_batch_size:
                        break
                    test_image_outputs = trainer.test(test_co_data,
                                                      test_cl_data,
                                                      opts.multigpus)
                    write_1images(test_image_outputs, image_directory,
                                  'test_%s_%02d' % (key_str, t))

        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations, opts.multigpus)
            print('Saved model at iteration %d' % (iterations + 1))

        iterations += 1
        if iterations >= max_iter:
            print("Finish Training")
            sys.exit(0)
