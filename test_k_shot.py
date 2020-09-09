"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils import get_config
from trainer import Trainer
from imgaug import augmenters as iaa
from globalConstants import GlobalConstants
import customTransforms
from data import default_loader_custom

import argparse

from skimage.io import imsave

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

config = get_config(opts.config)
config['batch_size'] = 1
config['gpus'] = 1

GlobalConstants.setPrecision(config['precision'])
GlobalConstants.setInputOutputChannels(config['gen']['input_nc'], config['gen']['output_nc'])
GlobalConstants.setOptimizer(config['optimizer'])
desired_size = config['desired_size']

#python test_k_shot.py --config outputs7_RMSprop/config.yaml --ckpt outputs7_RMSprop/checkpoints/gen_00075000.pt --input ../../../scratch/bunk/cell2cell/test/A/malaria/ac3358f1-ef9a-4ccc-b66c-da5e47e352e0.png --class_image_folder ../../../scratch/bunk/cell2cell/test/A/Human_HT29_Colon_Cancer_DNA/00733-DNA.tif --output images/output.jpg 

trainer = Trainer(config)
trainer.cuda()
trainer.load_ckpt(opts.ckpt)
trainer.eval()

"""
transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.Resize((128, 128))] + transform_list
transform = transforms.Compose(transform_list)
"""

transform = transforms.Compose([
        iaa.Sequential([
            #iaa.Resize({"shorter-side":resize_shorter_side, "longer-side":"keep-aspect-ratio"}),
            iaa.CropToFixedSize(width=desired_size, height=desired_size),
            #iaa.HorizontalFlip(p=0.5),
            #iaa.VerticalFlip(p=0.5),
        ]).augment_image,
        customTransforms.ToTensor(),
        customTransforms.RescaleToOneOne()
    ])

print('Compute average class codes for images in %s' % opts.class_image_folder)


classPath = opts.class_image_folder
imgPths = []
imgNames = next(os.walk(classPath))[2]
for imgName in imgNames:
    imgpath = os.path.join(classPath, imgName)
    imgPths.append(imgpath)


for i, f in enumerate(imgPths):
    img = default_loader_custom(f)
    img_tensor = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        class_code = trainer.model.compute_k_style(img_tensor, 1)
        if i == 0:
            new_class_code = class_code
        else:
            new_class_code += class_code
final_class_code = new_class_code / len(imgPths)


#final_class_code = trainer.model.gen_test.enc_class_model(transform(default_loader_custom(imgPths[0])).unsqueeze(0).cuda())
print("Shape: ",final_class_code.shape)
image = default_loader_custom(opts.input)
content_img = transform(image).unsqueeze(0)

print('Compute translation for %s' % opts.input)
with torch.no_grad():
    output_image = trainer.model.translate_simple(content_img, final_class_code)
    image = output_image.detach().cpu().squeeze().numpy()
    print("Image has shape: ", image.shape)
    #image = np.transpose(image, (1, 2, 0))
    image = ((image + 1) * 0.5 * 255.0)
    image = np.transpose(image)
    print("Image has shape now: ",image.shape)
    imsave(opts.output, image)
    #output_img = Image.fromarray(np.uint8(image))
    #output_img.save(opts.output, 'JPEG', quality=99)
    print('Save output to %s' % opts.output)
