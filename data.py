"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os.path
from PIL import Image

import torch.utils.data as data
from glob import glob
from skimage.io import imread
import skimage.color as color
from skimage.util import invert
import numpy as np
from globalConstants import GlobalConstants
from imgaug import augmenters as iaa


def default_loader(path):
    pic = Image.open(path).convert('RGB')
    return pic

def default_loader_custom(path):
    pic = imread(path)
    class_name = get_class(path)
    if class_name == "malaria":
        pic = color.rgb2grey(pic)
        pic = invert(pic)
    elif class_name == "Human_HT29_Colon_Cancer_DNA":
        pic = color.rgb2grey(pic)
    elif class_name == "dp":
        pic = color.rgba2rgb(pic)
        pic = color.rgb2grey(pic)
        
    #if (pic.dtype == 'uint16'):
        #print("anything else than double!!")
    #    if (pic.max()<32768):
    #        pic = pic.astype('int16')
    #    else:
    #        print("Converting to int32")
    #        pic = pic.astype('int32')
    if (GlobalConstants.usingApex):
        pic = pic.astype('float32')

    #if (pic.dtype  == 'int32'):
    #    print("Converting to uint32")
    #    pic = pic.astype('uint32')

    #if (pic.dtype == 'float64'):
    #    print("LOADING FLOAT64 IMAGE")

    if (GlobalConstants.getInputChannels()==3):
        if (len(pic.shape)==2):
            pic = pic.reshape((pic.shape[0], pic.shape[1],1))
            pic = np.repeat(pic, 3, axis=-1)
        if (pic.shape[0]==3):
            #print("**************3 IS BACK: ",pic.shape)
            #pic = pic.transpose((2,0,1)) #Not sure this is correct to get from (y,x,3) to (3,y,x)
            pass
    elif (GlobalConstants.getInputChannels()==1):
        if (len(pic.shape)==3):
            pic = color.rgb2grey(pic)
            print("Had to grayscale")
        elif (len(pic.shape)>3):
            print("ENCOUNTERED AN INPUT WITH MORE THAN 3 CHANNELS. THIS IS LIKELY TO CAUSE CRASHES. NUM OF CHANNELS: ",len(pic.shape))



    #=============SCALING======================
    shorter_side = min(pic.shape[0], pic.shape[1])
    if (shorter_side < 256):
        print("PIC VERY SMALL: ", shorter_side)
        shorter_side = shorter_side * 4
    if (class_name == "Hela"):
        shorter_side = shorter_side//8
    if (class_name == "mSar"):
        shorter_side = shorter_side//6
    if (class_name == "malaria"):
        shorter_side = shorter_side//4
    if (class_name == "Human_Hepatocyte_Murine_Fibroblast"):
        shorter_side = int(shorter_side/2)
    scale = iaa.Resize({"shorter-side":shorter_side, "longer-side":"keep-aspect-ratio"}).augment_image
    pic = scale(pic)

    return pic

def get_class(path):
    return path.split('/')[-2]

def default_filelist_reader(filelist):
    im_list = []
    with open(filelist, 'r') as rf:
        for line in rf.readlines():
            im_path = line.strip()
            im_list.append(im_path)
    return im_list


class ImageLabelFilelist(data.Dataset):
    def __init__(self,
                 root,
                 filelist,
                 transform=None,
                 filelist_reader=default_filelist_reader,
                 loader=default_loader,
                 return_paths=False):
        self.root = root
        self.im_list = filelist_reader(os.path.join(filelist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(
            list(set([path.split('/')[0] for path in self.im_list])))
        self.class_to_idx = {self.classes[i]: i for i in
                             range(len(self.classes))}
        self.imgs = [(im_path, self.class_to_idx[im_path.split('/')[0]]) for
                     im_path in self.im_list]
        self.return_paths = return_paths
        print('Data loader')
        print("\tRoot: %s" % root)
        print("\tList: %s" % filelist)
        print("\tNumber of classes: %d" % (len(self.classes)))

    def __getitem__(self, index):
        im_path, label = self.imgs[index]
        path = os.path.join(self.root, im_path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, label, path
        else:
            return img, label

    def __len__(self):
        return len(self.imgs)

class ImageLabelFilelistCustom(data.Dataset):
    """
    ToDo: If we want to load content and target
    Params:
        path:   Leads from executing script to the path with the subfolders of classes.
                The class is labeled after it's folders name.
    """

    def __init__(self,
                 root=".",
                 path="",
                 transform=None,
                 loader=default_loader_custom,
                 num_classes = None,
                 return_paths=False):

        print("PATH: ",path)        
        self.classes = next(os.walk(path))[1]
        self.imlist = []
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        for d in self.classes:
                impath = os.path.join(path, d)
                self.imlist += self.getImgs(impath, "*.tif") + self.getImgs(impath, "*.TIF") + self.getImgs(impath, "*.png") + self.getImgs(impath, "*.jpg")
                
        self.imgs = [(im_path, self.class_to_idx[im_path.split('/')[-2]]) for im_path in self.imlist]

        self.root = root #Do I need this?

        self.transform = transform
        self.loader = loader
        self.return_paths = return_paths
        print('Data loader')
        print("\tRoot: %s" % root)
        print("\tNumber of images: %d" % (len(self.imgs)))
        print("\tClasses: ",self.classes)
        print("\tNumber of classes: %d" % (len(self.classes)))
        if ((num_classes != None) and (num_classes != len(self.classes))):
            print("------------------WARNING----------------")
            print("It seems you have specified to have %d classes in the conf. file but %d classes were read" % (num_classes, len(self.classes)))

    def __getitem__(self, index):
        im_path, label = self.imgs[index]
        path = os.path.join(self.root, im_path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, label, path
        else:
            return img, label

    def __len__(self):
        return len(self.imgs)

    def getImgs(self, pth, dataType):
        return glob(os.path.join(pth, dataType))