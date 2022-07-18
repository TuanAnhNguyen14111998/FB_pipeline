# -*- coding: utf-8 -*-
"""
Created on 3/09/2020 8:51 pm

@author: Soan Duong, UOW
"""
# Standard library imports
import numpy as np
from numpy import loadtxt
from collections import namedtuple

# Third party imports
import cv2
import kornia as K
from PIL import Image
from albumentations import *
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset

# Local application imports
from utils import augmentations
from utils.augmentations import ToTensor, XRayResizer, XrayRandomHorizontalFlip, Squeeze


class XRayClassDataset(Dataset):
    def __init__(self, root_dir, txt_files, cfg, mode='train', labels=None, n_cutoff_imgs=None):
        """
        Initialize the class
        Args:
            root_dir: root directory to the dataset folder
            cfg: dictionary of the model configuration
            txt_files: filenames of text files that contains filename of x-ray images and ground-truth
                       format of each line in the text file: <image_name>, <ground_truth>
            n_cutoff_imgs: maximum number of used images in each text file.
                       This param is used for testing the dataset class
        """
        super(XRayClassDataset, self).__init__()
        self.txt_files = txt_files
        self.root_dir = root_dir
        self.cfg = cfg
        self.mode = mode
        self.img_list = None
        self.ground_truths = None
        self.labels = labels

        if (not isinstance(n_cutoff_imgs, int)) or (n_cutoff_imgs <= 0):
            n_cutoff_imgs = None
        if isinstance(txt_files, str):
            txt_files = [txt_files]

        # Get filename of the training images stored in txt_files
        img_list = []
        for filename in txt_files:
            imgs = loadtxt(filename, delimiter=', ', dtype=np.str)[:n_cutoff_imgs]
            #print(imgs)
            #print(imgs.shape)
            if len(img_list) == 0:
                img_list = imgs
            else:
                img_list = np.concatenate((img_list, imgs))

        # Set the training_images and ground_truths
        if len(img_list.shape) == 1:
            self.img_list = img_list
        else:
            # Set the ground-truth for
            self.ground_truths = list(map(int, img_list[:, 1]))
            # self.ground_truths = np.asarray(list(map(int, img_list[:, 1])))
            # Set the training image
            self.img_list = img_list[:, 0]

        # Append the root_dir to get the full path of the images
        self.img_list = [self.root_dir + img for img in self.img_list]
        #print(self.ground_truths)

    def __len__(self):
        """
        Returns: the size of the dataset, i.t. the number of input images
        """
        return len(self.img_list)

    def __getitem__(self, index):
        """
        Read an image from the training images
        Args:
            index: index of the image in the list of training images
        Returns: image and ground-truth images
                + x: input image of size (1, H, W)
                + y_lbl: label of the image
                + y_oht: one-hot coding label of the image
                + img_path: file path of the image
        """
        img_path = self.img_list[index]
        #print(img_path)
        # Read the input image
        img = cv2.imread(img_path)                  # of size (H, W, 3), color order: BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # of size (H, W, 3), color order: RGB

        # Apply the preprocessing the the img
        # x = pre_process(img, self.cfg, self.mode)   # Tensor of size (H, W, 1)
        # x = x.repeat((3, 1, 1))                         # Tensor of size (H, W, 1)
        x = matching_templates(img, self.cfg, self.mode)
        # print(x.shape)

        # Get the ground-truth of the input image
        if self.ground_truths is None:
            return {'input': x, 'file_path': img_path}
        else:
            y_lbl = self.ground_truths[index]
            if len(self.labels) == 2: # for binary classification
                y_lbl = torch.as_tensor(y_lbl, dtype=torch.float32)
            else:                     # for multi-class classification
                y_lbl = torch.as_tensor(y_lbl, dtype=torch.long)
                
            return {'input': x, 'ground_truth_label': y_lbl, 'file_path': img_path}


# ------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------
def pre_process(img, cfg, mode='train'):
    """
    Preprocess the input img with the given configuration cfg.
    The preprocessing pipeline includes:
    + Normalize the intensity into the range [-1024, 1024]
    + Apply the transformation (resizing and cropping)
    Args:
        img: input image, 3D array of size (H, W, 3)
        cfg: dictionary of the model configuration
        mode: mode of the dataloader (train/val/test)

    Returns: transformed image
    """
    def _norm_intensity(sample, maxval=255):
        """
        Normalize the intensity of the input sample into the range [-1024, 1024]
        Args:
            sample:
            maxval:

        Returns:

        """
        sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
        return sample

    # Assert if the given img is an image
    assert len(img.shape) >= 2, "Given img variable is not an image"

    # Get the first channel of the img
    if len(img.shape) > 2:
        img = img[:, :, 0]                  # of size (H, W)

    # Normalize the intensity of the image
    img = _norm_intensity(img)              # of size (H, W)

    # Add a dimension for the number of channel
    img = img[np.newaxis, ...]              # of size (1, H, W)

    if mode == 'train':
        transform = nn.Sequential(ToTensor(), XRayResizer(cfg['img_size']),
                                  K.augmentation.CenterCrop(size=cfg['crop_size']),
                                  XrayRandomHorizontalFlip(p=0.5),
                                  Squeeze(), )
    else:
        transform = nn.Sequential(ToTensor(), XRayResizer(cfg['img_size']),
                                  K.augmentation.CenterCrop(size=cfg['crop_size']),
                                  Squeeze())

    return transform(img)


def matching_templates(org_img, cfg, mode='train'):
    """
    Preprocessing pipeline:
    + Resize
    + Center crop
    + trans = transforms.Compose(
                [transforms.Resize(size=cfg.img_size), transforms.CenterCrop(size=cfg.crop_size),
                 transforms.RandomRotation(degrees=5), transforms.ColorJitter(),
                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    Args:
        org_img:
        cfg:
        mode:

    Returns:

    """
    # Convert cfg from dictionary to a class object
    cfg = namedtuple('cfg', cfg.keys())(*cfg.values())

    img = Image.fromarray(org_img)

    if cfg.imagenet:
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(
            [cfg.pixel_mean / 256, cfg.pixel_mean / 256, cfg.pixel_mean / 256],
            [cfg.pixel_std / 256, cfg.pixel_std / 256, cfg.pixel_std / 256])

    if mode == 'train':
        if cfg.n_crops == 10:
            trans = transforms.Compose(
                [transforms.RandomResizedCrop(size=cfg.img_size, scale=(0.8, 1.0)),
                 transforms.RandomRotation(degrees=5), transforms.ColorJitter(),
                 transforms.TenCrop(size=cfg.crop_size), transforms.Lambda(
                    lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                 transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])
        elif cfg.n_crops == 5:
            trans = transforms.Compose(
                [transforms.RandomResizedCrop(size=cfg.img_size, scale=(0.8, 1.0)),
                 transforms.RandomRotation(degrees=5), transforms.ColorJitter(),
                 transforms.RandomHorizontalFlip(), transforms.FiveCrop(size=cfg.crop_size),
                 transforms.Lambda(
                     lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                 transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])
        elif cfg.n_crops == -1:
            trans = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            trans = transforms.Compose(
                [transforms.Resize(size=cfg.img_size), transforms.CenterCrop(size=cfg.crop_size),
                 transforms.RandomRotation(degrees=5), transforms.ColorJitter(),
                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    elif mode == 'test' and cfg.n_crops > 0:
        trans = transforms.Compose(
            [transforms.Resize(size=cfg.img_size), transforms.FiveCrop(size=cfg.crop_size),
             transforms.Lambda(
                 lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
             transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])
    else:
        trans = transforms.Compose(
            [transforms.Resize(size=cfg.img_size), transforms.CenterCrop(size=cfg.crop_size),
             transforms.ToTensor(), normalize])

    if mode == 'train' and cfg.n_crops == 0 and cfg.augmix:
        if cfg.no_jsd:
            return augment_and_mix(img, trans, cfg)
        else:
            return trans(img), augment_and_mix(img, trans, cfg), augment_and_mix(img, trans, cfg)
    else:
        if mode == 'train' and cfg.n_crops == -1:
            aug = strong_aug(cfg.crop_size, p=1)
            img = Image.fromarray(augment(aug, org_img))

        return trans(img)


def augment_and_mix(image, trans, cfg):
    """Perform AugMix augmentations and compute mixture.
      Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
      Returns:
        mixed: Augmented and mixed image.
      """
    aug_list = augmentations.augmentations
    if cfg.all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([cfg.aug_prob_coeff] * cfg.mixture_width))
    m = np.float32(np.random.beta(cfg.aug_prob_coeff, cfg.aug_prob_coeff))

    mix = torch.zeros_like(trans(image))
    for i in range(cfg.mixture_width):
        image_aug = image.copy()
        depth = cfg.mixture_depth if cfg.mixture_depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, cfg.aug_severity)

        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * trans(image_aug)

    mixed = (1 - m) * trans(image) + m * mix
    return mixed


def strong_aug(img_size, p=1):
    return Compose([Resize(img_size, img_size, always_apply=True), Flip(), Transpose(),
                    OneOf([IAAAdditiveGaussianNoise(), GaussNoise(), ], p=0.5),
                    OneOf([MotionBlur(p=.2), MedianBlur(blur_limit=3, p=.1), Blur(blur_limit=3, p=.1), ],
                          p=0.5), ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45),
                    OneOf([CLAHE(clip_limit=2), IAASharpen(), IAAEmboss(), RandomBrightnessContrast(), ],
                          p=0.5), Cutout(max_h_size=8, max_w_size=8), ], p=p)


def augment(aug, image):
    return aug(image=image)['image']



if __name__ == "__main__":
    root_dir = '../../00-data/chest_xray_pneumonia_children/train'
    txtfile = 'data/train_pneumonia.txt'
    cfg = {'img_size': 224, 'crop_size': 224, 'batch_size': 1}
    # img_list = loadtxt(txtfile, delimiter=',', dtype=np.str)
    #
    # img_list = [root_dir + img for img in img_list[:, 0]]
    #
    # img_path = img_list[0]
    # img_bgr = cv2.imread(img_path)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    train_dataset = XRayClassDataset(root_dir, txtfile, cfg)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg['batch_size'],
                                                   pin_memory=True)

    from tqdm import tqdm
    pbar = tqdm(train_dataloader, ncols=80, desc='Training')
    for step, minibatch in enumerate(pbar):
        print(minibatch['input'].shape)
    # import glob
    # # Set the dataset
    # dataset_dir = '../../00-data/chest_xray_pneumonia_children/train'
    # img_ext = 'jpeg'
    # files = glob.glob('%s/NORMAL/*%s' % (dataset_dir, img_ext))
    # list_imgs = [file.replace(dataset_dir, '') + ', 0' for file in files]
    #
    # files = glob.glob('%s/PNEUMONIA/*%s' % (dataset_dir, img_ext))
    # list_imgs = np.concatenate((list_imgs, [file.replace(dataset_dir, '') + ', 1'
    #                                         for file in files]))
    #
    # # Save the list into a text file
    # with open('data/train_pneumonia.txt', 'w') as filehandle:
    #     for listitem in list_imgs:
    #         filehandle.write('%s\n' % listitem)

    # root_dir = 'ccc'
    # for k in range(len(imgs)):
    #     imgs[k, 0] = root_dir + '/' + imgs[k, 0]
    #     imgs[k, 1] = int(imgs[k, 1])
