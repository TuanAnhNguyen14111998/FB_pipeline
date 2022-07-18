import numpy as np
import cv2
import torch
from torchvision.transforms import ToTensor
import albumentations as albu
from lib.bone_segmentation.se_enet import ENet as SE_ENet

class BoneSegmentation:
    def __init__(self, 
                 weight_path,
                 img_size=320,
                 device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        self.core = SE_ENet(num_classes=1)
        self.core.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.core.to(self.device)
        self.core.eval()

        self.img_size = img_size
        self.aug = albu.Compose([
            albu.Resize(self.img_size, self.img_size, always_apply=True),
            albu.CLAHE(clip_limit=(2.0,2.0), tile_grid_size=(4, 4), always_apply=True)
        ])

    def predict(self, img_array):
        x = self.preprocess(img_array)
        x = self.core(x)
        return self.postprocess(x)

    def preprocess(self, x):
        self.org_w, self.org_h, _ = x.shape
        return ToTensor()(self.aug(image=x)['image']).unsqueeze(0).to(self.device)

    def postprocess(self, x):
        x = x[0].detach().cpu()
        x = torch.sigmoid(x).numpy()[0]
        x = (x > 0.5).astype(np.uint8)
        x = cv2.resize(x, (self.org_h, self.org_w))
        return x

    @staticmethod
    def visualize(img_array, mask_pred):
        img = img_array/255
        msk = mask_pred
        mask = msk[...,None]
        color_mask = np.array([0.2*msk, 0.5*msk, 0.85*msk])
        color_mask = np.transpose(color_mask, (1,2,0))
        blend = 0.3*color_mask + 0.7*img*mask + (1 - mask)*img
        return blend
