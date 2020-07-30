"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import yaml
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils

from data import ImageLabelFilelist, ImageLabelFilelistCustom
import customTransforms
from imgaug import augmenters as iaa
from glob import glob
import torch.nn.functional as F

# Finds biggest 2^x such that 2^x < size
def find_next_crop_size(size):
    x = 0
    while (size >= (2 ** (x+1))):
        x+=1
    return 2**x

# Scales so much down that picture is still bigger than desired size so that crop can crop it exactly
# But resizing tries to save the context
# Probably 
def resize_correctly(current_size, desired_size):
    x = 1
    while (desired_size <= current_size/(x+1)):
        x+=1
    return x


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


def create_loader(root, path, rescale_size_a, rescale_size_b, batch_size, num_classes=None,
    num_workers=4, desired_size=None, resize_shorter_side=None, shuffle=True, return_paths=False, drop_last=True):

    crop_size = find_next_crop_size(rescale_size_a)
    cut = rescale_size_a - crop_size
    print("RESCALE_SIZE: ",rescale_size_a)
    transforms_ = transforms.Compose([
        iaa.Sequential([
            iaa.Resize({"shorter-side":resize_shorter_side, "longer-side":"keep-aspect-ratio"}),
            iaa.CropToFixedSize(width=desired_size, height=desired_size),
            iaa.HorizontalFlip(p=0.5),
            iaa.VerticalFlip(p=0.5)
        ]).augment_image,
        customTransforms.ToTensor(),
        customTransforms.RescaleToOneOne()
    ])
    dataset = ImageLabelFilelistCustom(root=root, path=path, transform=transforms_, return_paths=return_paths, num_classes=num_classes)
    print(dataset[0][0].shape)
    loader = DataLoader(dataset,
                        batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=num_workers)
    return loader


def loader_from_list(
        root,
        file_list,
        batch_size,
        new_size=None,
        height=128,
        width=128,
        crop=True,
        num_workers=4,
        shuffle=True,
        center_crop=False,
        return_paths=False,
        drop_last=True):
    transform_list = [customTransforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if center_crop:
        transform_list = [transforms.CenterCrop((height, width))] + \
                         transform_list if crop else transform_list
    else:
        transform_list = [transforms.RandomCrop((height, width))] + \
                         transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list \
        if new_size is not None else transform_list
    if not center_crop:
        transform_list = [transforms.RandomHorizontalFlip()] + transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageLabelFilelist(root,
                                 file_list,
                                 transform,
                                 return_paths=return_paths)
    loader = DataLoader(dataset,
                        batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=num_workers)
    return loader


def get_evaluation_loaders(conf, shuffle_content=False):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size = conf['new_size']
    width = conf['crop_image_width']
    height = conf['crop_image_height']
    content_loader = loader_from_list(
            root=conf['data_folder_train'],
            file_list=conf['data_list_train'],
            batch_size=batch_size,
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=num_workers,
            shuffle=shuffle_content,
            center_crop=True,
            return_paths=True,
            drop_last=False)

    class_loader = loader_from_list(
            root=conf['data_folder_test'],
            file_list=conf['data_list_test'],
            batch_size=batch_size * conf['k_shot'],
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=1,
            shuffle=False,
            center_crop=True,
            return_paths=True,
            drop_last=False)
    return content_loader, class_loader

def get_train_loaders_custom(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    root = "."
    pathBasis = conf["data_folder_train"]
    scalar = conf["scalar"]
    rescale_size_a = (conf["size_a"]//scalar)
    rescale_size_b = (conf["size_b"]//scalar)

    train_content_loader = create_loader(
        ".",
        os.path.join(pathBasis, "A"),
        rescale_size_a,
        rescale_size_b,
        desired_size = conf["desired_size"],
        resize_shorter_side = conf["resize_shorter_side"],
        num_classes=conf['dis']['num_classes'],
        batch_size=batch_size,
        num_workers=num_workers
    )
    train_class_loader = create_loader(
        ".",
        os.path.join(pathBasis, "A"),
        rescale_size_a,
        rescale_size_b,
        desired_size = conf["desired_size"],
        resize_shorter_side = conf["resize_shorter_side"],
        num_classes=conf['dis']['num_classes'],
        batch_size=batch_size,
        num_workers=num_workers
    )
    return (train_content_loader, train_class_loader, train_content_loader, train_class_loader)

def get_train_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size = conf['new_size']
    width = conf['crop_image_width']
    height = conf['crop_image_height']
    train_content_loader = loader_from_list(
            root=conf['data_folder_train'],
            file_list=conf['data_list_train'],
            batch_size=batch_size,
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=num_workers)
    train_class_loader = loader_from_list(
            root=conf['data_folder_train'],
            file_list=conf['data_list_train'],
            batch_size=batch_size,
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=num_workers)
    test_content_loader = loader_from_list(
            root=conf['data_folder_test'],
            file_list=conf['data_list_test'],
            batch_size=batch_size,
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=1)
    test_class_loader = loader_from_list(
            root=conf['data_folder_test'],
            file_list=conf['data_list_test'],
            batch_size=batch_size,
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=1)

    return (train_content_loader, train_class_loader, test_content_loader,
            test_class_loader)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def make_log_folder(basePath):
    logs_path = os.path.join(basePath, "logs")
    exists = True
    newest_index = 1
    if not os.path.exists(logs_path):
        print("Creating directory: {}".format(logs_path))
        os.makedirs(logs_path)
        exists = False
    if exists:
        #Expected strucutre "run_1, run_2, SOME_COOL_NAME_3 ..."
        dirs = glob(os.path.join(logs_path, "*"))
        for direc in dirs:
            direc = direc.split("/")[-1]
            index = int(direc.split("_")[-1])
            newest_index = max(newest_index, index)
        newest_index += 1
    new_dir = os.path.join(logs_path, "run_"+str(newest_index))
    os.makedirs(new_dir)
    return new_dir

    

def make_result_folders(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def __write_images(im_outs, dis_img_n, file_name):
    im_outs = [images.expand(-1, 3, -1, -1) for images in im_outs]
    if im_outs[0].shape != im_outs[1].shape:
        in_dim = im_outs[0].shape[2]
        out_dim = im_outs[1].shape[2]
        diff = int((out_dim - in_dim)/2)
        diff_tup = (diff, diff, diff, diff)
        im_outs[0] = F.pad(input=im_outs[0], pad=diff_tup, mode='constant', value=0)
        im_outs[3] = F.pad(input=im_outs[3], pad=diff_tup, mode='constant', value=0)
    image_tensor = torch.cat([images[:dis_img_n] for images in im_outs], 0)
    image_grid = vutils.make_grid(image_tensor.data,
                                  nrow=dis_img_n, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_1images(image_outputs, image_directory, postfix):
    display_image_num = image_outputs[0].size(0)
    __write_images(image_outputs, display_image_num,
                   '%s/gen_%s.jpg' % (image_directory, postfix))


def _write_row(html_file, it, fn, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (it, fn.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (fn, fn, all_size))
    return


def write_html(filename, it, img_save_it, img_dir, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    _write_row(html_file, it, '%s/gen_train_current.jpg' % img_dir, all_size)
    for j in range(it, img_save_it - 1, -1):
        _write_row(html_file, j, '%s/gen_train_%08d.jpg' % (img_dir, j),
                   all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr))]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
