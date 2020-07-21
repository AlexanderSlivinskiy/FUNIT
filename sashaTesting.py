import torch
import os
import sys
import argparse
import shutil

from tensorboardX import SummaryWriter

from utils import get_config, get_train_loaders, make_result_folders
from utils import write_loss, write_html, write_1images, Timer
from trainer import Trainer
from data import ImageLabelFilelist,ImageLabelFilelistCustom
from PIL import Image

import torch.utils.data as data

import torch.backends.cudnn as cudnn
from glob import glob


directory = '../../../scratch/bunk/cell2cell/train/'
path='../../../scratch/bunk/cell2cell/train/A/*.tif'
pathX='../../../scratch/bunk/cell2cell/train/A/*.tif'
imlist = glob('../../../scratch/bunk/cell2cell/train/A/*.tif')
classes = sorted(list(set([path.split('/')[-2] for path in imlist])))
#print(path.split('/')[-2])
dirs = next(os.walk(directory))[1]
print("DIRS: ")
imlist = []
class_to_idx = {dirs[i]: i for i in range(len(dirs))}
for d in dirs:
    print("d: ",d)
    path = os.path.join(directory, d)
    path = os.path.join(path, "*.tif")
    #print(path)
    print("P_X: ",pathX)
    print("P_N: ",path)
    #print(glob(pathX))
    imlist += glob(path)


class_to_idx = {dirs[i]: i for i in range(len(dirs))}



print([(im_path, class_to_idx[im_path.split('/')[-2]]) for im_path in imlist])
#imgs = [(im_path, class_to_idx[im_path.split('/')[0]]) for
#                     im_path in self.im_list]
#print(class_to_idx)
#print(imlist)
#print(imlist)
#print("\n\n=====================\n\n",classes)
#print(dirs)
#print(next(os.walk(directory))[1])



dataset = ImageLabelFilelistCustom(
    path="../../../scratch/bunk/cell2cell/train/",
    return_paths=True
)
print("LENGTH: ",len(dataset))
print(dataset[0])
#loader = DataLoader(dataset,
#                    batch_size,
#                    shuffle=shuffle,
#                    drop_last=drop_last,
#                    num_workers=num_workers)