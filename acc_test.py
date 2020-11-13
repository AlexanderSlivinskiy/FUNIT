"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import numpy as np
from skimage.io import imread
from skimage.io import imsave
from glob import glob
from skimage.util import invert
import skimage.color as color
from skimage.filters import threshold_otsu
from skimage import data
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from data import default_loader_custom
import sys
import random
import csv

from PIL import Image

import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils import get_config
from trainer import Trainer
from globalConstants import GlobalConstants
import customTransforms

from debugUtils import DebugNet

import argparse

from skimage.io import imsave

def load_pic(path):
    pic = imread(path)
    class_name = get_class(path)
    if class_name == "malaria":
        pic = color.rgb2grey(pic)
        pic = invert(pic)
    elif class_name == "Human_HT29_colon-cancer" or class_name == "dna":
        #pic = color.rgb2grey(pic)
        pass
    elif class_name == "dp":
        pic = color.rgba2rgb(pic)
        pic = color.rgb2grey(pic)
        
    if (pic.dtype == 'uint16'):
        #print("anything else than double!!")
        if (pic.max()<32768):
            pic = pic.astype('int16')
        else:
            pic = pic.astype('int32')
    if (False):
        pic = pic.astype('float')


    #if (len(pic.shape)==2):
        #pic = pic.reshape((pic.shape[0], pic.shape[1],1))
        #pic = np.repeat(pic, 3, axis=-1)
    if (pic.shape[0]==3):
        #print("**************3 IS BACK: ",pic.shape)
        pic = pic.transpose() #Not sure this is correct to get from (y,x,3) to (3,y,x)
    
    #==========RESHAPING=============
    shorter_side = min(pic.shape[0], pic.shape[1])
    if (class_name == "Hela"):
        shorter_side = shorter_side//8
    if (class_name == "mSar"):
        shorter_side = shorter_side//6
    if (class_name == "malaria"):
        shorter_side = shorter_side//4
    if (class_name == "Human_Hepatocyte_Murine_Fibroblast"):
        shorter_side = int(shorter_side/2)
    else:
        #shorter_side = 256
        pass
    scale = iaa.Resize({"shorter-side":shorter_side, "longer-side":"keep-aspect-ratio"}).augment_image
    pic = scale(pic)
         
    #===========CROP==================    
    crop = iaa.CropToFixedSize(width=256, height=256, position = 'center', seed = 0).augment_image
    pic  = crop(pic)
    #==========TO [0,1]=============
    pic = pic/pic.max()
    
    return pic

def get_class(path):
    return path.split('/')[-2]

def getImgs(pth, dataType):
    return glob(os.path.join(pth, dataType))

def cleanTestFolder():
    test_path = "../../../scratch/slivinskiy/new_datasets/Tests"
    to = "../../../scratch/slivinskiy/new_datasets/LostAndFound"
    imgs = getImgs(test_path, "*.tif")
    for im in imgs:
        os.system("mv "+im+" "+to)
        
def cleanLostAndFound():
    from_path = "../../../scratch/slivinskiy/new_datasets/LostAndFound"
    to_path = "../../../scratch/slivinskiy/new_datasets/high_conc/Train"
    csv_path = '../../../scratch/slivinskiy/new_datasets/conf/BBBC022_v1_image.csv'
    imgs = getImgs(from_path, "*.tif")
    
    with open('../../../scratch/slivinskiy/new_datasets/conf/BBBC022_v1_image.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',', quotechar = '"')
        classes = ["OrigER", "OrigHoechst", "OrigMito", "OrigPh_golgi", "OrigSyto"]
        index = 0
        for row in readCSV:
            for i in range(1,6):
                reference_name = row[i]
                for lost_img in imgs:
                    n = lost_img.split("/")[-1]
                    if (n == reference_name):
                        target_folder = os.path.join(to_path, classes[i-1])
                        os.system("mv "+lost_img+" "+target_folder)

def getRandomPicFromClass(count, path):
    img_pths = getImgs(path, "*.tif")
    img_paths = []
    fr = 0
    to = len(img_pths) -1
    for i in range(count):
        picIndex = random.randint(fr, to)
        img_paths.append(img_pths[picIndex])
        
    return img_paths

def getRandomInputPics_Paths(count, content_dir_path):
    content_class = classes[random.randint(0, len(classes)-1)]
    content_cls_path = os.path.join(content_dir_path, content_class)
    return getRandomPicFromClass(count, content_cls_path)

def getRandomClassImg(take_from, put_to, content_dir_path, class_path = ""):
    if (class_path != ""):
        pth = getRandomPicFromClass(1, os.path.join(take_from,class_path))[0]
    else:
        pth = getRandomInputPics_Paths(1, content_dir_path)[0]
    os.system("mv "+pth+" "+put_to)
    return pth

def putBack(test_path, original_path):
    img_pths = getImgs(test_path, "*.tif")
    img_pth = img_pths[0]
    os.system("mv "+img_pth+" "+original_path)

def findCorrespondingHoechstImg(imgName):
    with open('../../../scratch/slivinskiy/new_datasets/conf/BBBC022_v1_image.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',', quotechar = '"')
        classes = ["OrigER", "OrigHoechst", "OrigMito", "OrigPh_golgi", "OrigSyto"]
        for row in readCSV:
            for i in range(1,6):
                reference_img_name = row[i]
                if (reference_img_name == imgName):
                    #print("Found image in the table!")
                    return row[2]
    return ""


def initFUNIT(conf, ckpt):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='configs/funit_animals.yaml')
    parser.add_argument('--ckpt',
                        type=str,
                        default='pretrained/animal119_gen_00200000.pt')
    parser.add_argument('--class_image_folder',
                        type=str,
                        default='images/n02138411')
    parser.add_argument('--input',
                        type=str,
                        default='images/input_content.jpg')
    parser.add_argument('--output',
                        type=str,
                        default='images/output.jpg')
    opts = parser.parse_args()
    cudnn.benchmark = True
    opts.vis = True

    config = get_config(conf)
    config['batch_size'] = 1
    config['gpus'] = 1

    DebugNet.safeImgSwitch = True

    GlobalConstants.setPrecision(config['precision'])
    GlobalConstants.setInputOutputChannels(config['gen']['input_nc'], config['gen']['output_nc'])
    GlobalConstants.setOptimizer(config['optimizer'])
    desired_size = config['desired_size']

    trainer = Trainer(config)
    trainer.cuda()
    trainer.load_ckpt(ckpt)
    trainer.eval()
    #resume_directory = opts.ckpt
    #trainer.resume(resume_directory,hp=config,multigpus=False)

    transform = transforms.Compose([
            iaa.Sequential([
                #iaa.Resize({"shorter-side":256, "longer-side":"keep-aspect-ratio"}), #Don't forget to exclude this later
                iaa.CropToFixedSize(width=desired_size, height=desired_size, position = 'center', seed = 0),
                #iaa.HorizontalFlip(p=0.5),
                #iaa.VerticalFlip(p=0.5),
            ]).augment_image,
            customTransforms.ToTensor(),
            customTransforms.RescaleToOneOne()
        ])
    return (trainer, transform)

def useFUNIT(class_img_folder, content_img_pth, output_path, trainer, transform):
    classPath = class_img_folder
    imgPths = []
    imgNames = next(os.walk(classPath))[2]
    for imgName in imgNames:
        imgpath = os.path.join(classPath, imgName)
        imgPths.append(imgpath)

    final_class_code = trainer.model.gen_test.enc_class_model(transform(default_loader_custom(imgPths[0])).unsqueeze(0).cuda())
    DebugNet.setName("input")

    image = default_loader_custom(content_img_pth)
    content_img = transform(image)
    #DebugNet.safeImage(content_img)
    content_img = content_img.unsqueeze(0)

    with torch.no_grad():
        output_image = trainer.model.translate_simple(content_img, final_class_code)
        image = output_image.detach().cpu().squeeze().numpy()
        image = ((image + 1) * 0.5 * 255.0)
        if (len(image.shape) == 3):
            image = np.transpose(image, (1, 2, 0))
        imsave(output_path, image)
        print('Save output to %s' % output_path)


accs = []
accs_classes = []
img_counter = 0
img_seen = []

dir_path = "../../../scratch/slivinskiy/SOFI/test"
classes = next(os.walk(dir_path))[1]
pth_list = []

for cls in classes:
    cls_pth = os.path.join(dir_path, cls)
    cls_imgs = getImgs(cls_pth, "*.TIF") + getImgs(cls_pth, "*.png") + getImgs(cls_pth, "*.jpg") + getImgs(cls_pth, "*.tif")
    pth_list.append(cls_imgs)

print(len(pth_list))


content_dir_path = "../../../scratch/slivinskiy/new_datasets/run2/Test"
cls_path = "../../../scratch/slivinskiy/new_datasets/Tests"
class_source_path = "../../../scratch/slivinskiy/new_datasets/high_conc/Train"

#trained_model_path = "../../../scratch/slivinskiy/new_datasets/GPU1/outputs/funit_B022/"
trained_model_path = "../../../scratch/slivinskiy/new_datasets/GPU1/outputs_further/outputs/config/"
checkpoint_path = trained_model_path+"checkpoints/gen_00470000.pt"
config_path = trained_model_path+"config.yaml"
funit_dir = "FUNIT_GPU1"

classes = next(os.walk(content_dir_path))[1]

(trainer, transform) = initFUNIT(config_path, checkpoint_path)

while(img_counter<1500):
    print("img_counter: ",img_counter)
    moved_class_img_path = getRandomClassImg(class_source_path, cls_path, content_dir_path, class_path = "OrigHoechst")
    try:
        inp_num = 1

        class_img_pths = []

        cls_imgs = getImgs(cls_path, "*.TIF") + getImgs(cls_path, "*.png") + getImgs(cls_path, "*.jpg") + getImgs(cls_path, "*.tif")
        class_img_pths.append(cls_imgs)

        class_img_pths = class_img_pths[0]
        
        columns = max(inp_num, len(class_img_pths))
        rows = 5
        columns = 2


        pic_paths = getRandomInputPics_Paths(inp_num, content_dir_path)

        #print("INPUT IMAGE SHAPE: ", imread(pic_paths[0]).shape)

        content_class = pic_paths[0].split("/")[-2]
        content_img_name = pic_paths[0].split("/")[-1]
        if (content_class == "OrigHoechst"):
            putBack(cls_path, moved_class_img_path)
            continue
        if content_img_name in img_seen:
            putBack(cls_path, moved_class_img_path)
            continue
        else:
            img_seen.append(content_img_name)
        corresponding_hoechst_name = findCorrespondingHoechstImg(content_img_name)
        corresponding_hoechst_image_path = ""

        goIntoFUNIT_MIX = "cd ../"+funit_dir+" ;"


        options = ["Test", "Val", "Train"]
        base_path = "../../../scratch/slivinskiy/new_datasets/classes"
        for opt in options:
            base_pth = os.path.join(base_path, opt)
            base_pth = os.path.join(base_pth, "OrigHoechst")
            #print("Search in: "+base_pth)
            base_img_name = os.path.join(base_pth, corresponding_hoechst_name)
            #print("PATH: ",base_img_name)
            if (os.path.exists(base_img_name)):
                #print("Hoechst Image found")
                corresponding_hoechst_image_path = base_img_name
                break



        #print("Exists? ",os.path.exists(content_dir_path+"/OrigHoechst/"+corresponding_hoechst_name))

        style = class_img_pths[0].split("/")[-1].split("_")[0]
        output_pic_paths = []
        fullpath = "../"+funit_dir+"/"
        for i in range(len(pic_paths)):
            pic_path = pic_paths[i]
            outputName = "images/test_"+content_class+"_"+((str)(i))+"_to_"+style+"_"+((str)(len(class_img_pths)))+".png"
            os.system(goIntoFUNIT_MIX + " rm -rf pics; mkdir pics")
            useFUNIT(cls_path, pic_path, outputName , trainer, transform)
            #os.system(goIntoFUNIT_MIX + " python test_k_shot.py --config "+config_path+" --ckpt "+checkpoint_path+" --input "+pic_path+" --class_image_folder "+cls_path+" --output "+outputName)
            output_pic_paths.append(fullpath+outputName)

        output_pic = load_pic("../"+funit_dir+"/"+outputName)
        #plt.imshow(output_pic, cmap="gray")


        fake = load_pic(output_pic_paths[0])
        thresh = threshold_otsu(fake)
        fake = fake > thresh

        real = load_pic(corresponding_hoechst_image_path)
        thresh = threshold_otsu(real)
        real = real > thresh

        #print("Shape Fake: ",fake.shape,"Shape Real: ",real.shape)
        #print("Entries:",(fake.shape[0] * fake.shape[1]))
        #print("True Fake: ",np.sum(fake),", True Real: ",np.sum(real))
        intersect = np.logical_and(fake, real)
        union = np.logical_or(fake, real)



        putBack(cls_path, moved_class_img_path)

        print("====DONE====")
        #print("Class: ", content_class, " to :",moved_class_img_path.split('/')[-2])
        acc = (np.sum(intersect)/np.sum(union))*100
        accs.append(acc)
        accs_classes.append(content_class)
        print("ACCURRACY: ",acc,"%")
        img_counter+=1
    except Exception as e:
        print(e)
        putBack(cls_path, moved_class_img_path)

    
print("Number of random tests: ",len(accs))
minimum = 100
avg = 0
maximum = 0
for i in range(len(accs)):
    minimum = min(accs[i], minimum)
    maximum = max(accs[i], maximum)
    avg+=accs[i]
avg /= len(accs)
print("average: ",avg)
print("minimum percentage: ",minimum)
print("maximum percentage: ",maximum)
