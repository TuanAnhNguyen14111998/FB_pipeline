import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from lib.Lung_segment.model.enet import ENet
from lib.Lung_segment.model.attention_block import SENet, CBAM, SelfAttentionBlock
import torch
from torchvision import transforms
import albumentations
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
from skimage.measure import label

class LungSegmentation:
    def __init__(self, checkpoint_path,
                       image_size = 320,
                       device='cuda:0' if torch.cuda.is_available() else 'cpu',
                       threshold = 0.2):
        self.device = device
        self.model = ENet(num_classes = 1,
                          attention_block= SENet,
                          ignore_attention=[])
        self.model.to(device)
        self.model.eval()
        self.load_checkpoints(checkpoint_path)

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.540,0.540,0.540), std = (0.264,0.264,0.264)),
            transforms.Resize((image_size, image_size))
        ])
        self.image_size = image_size
        self.threshold = threshold
    
    def predict_batch(self, org_h, org_w, img_tensor, threshold = None):
        if threshold is None:
            threshold = self.threshold
        img_tensor = img_tensor.to(self.device)
        outputs = self.model(img_tensor)
        if isinstance(outputs, list):
            outputs = outputs[-1]
        
        list_pred = []
        with torch.no_grad():
            predicts = torch.sigmoid(outputs)
            predicts = predicts.cpu().numpy().transpose(0, 2, 3, 1)
            predicts = (predicts > threshold).astype(np.uint8)
            for i in range(predicts.shape[0]):
                predict = predicts[i][..., 0]
                predict = revert_origin_size(predict, int(org_h[i]), int(org_w[i]))   
                # predict = getLargestCC(predict)
                list_pred.append(predict)
        
        return list_pred

    def predict(self, img_array):
        assert img_array is not None
        with torch.no_grad():
            img_tensor = self.preprocess(img_array)
            outputs = self.model(img_tensor)
        predicts = self.posprocess(outputs, self.threshold)

        return predicts

    def preprocess(self, img):
        self.org_h, self.org_w = img.shape[:2]
        if len(img.shape) == 2:
            img = np.dstack([img,]*3)
        elif len(img.shape) == 3:
            img = img[:,:,:3]
        img = center_crop(img)
        img_tensor = self.transform_test(img)
        img_tensor = img_tensor.to(self.device)
        img_tensor = img_tensor[None, :, :, :]

        return img_tensor

    def posprocess(self, outputs, threshold = 0.2):
        if isinstance(outputs, list):
            outputs = outputs[-1]
        predicts = torch.sigmoid(outputs)[0]
        predicts = predicts.cpu().numpy().transpose(1, 2, 0)[:,:,0]
        predicts = (predicts > threshold).astype(np.uint8)
        predicts = revert_origin_size(predicts, self.org_h, self.org_w)
        
        return predicts

    def visualize(self, img_array, msk):
        if len(img_array.shape) == 2:
            img_array = np.dstack([img_array,]*3)
        elif len(img_array.shape) == 3:
            img_array = img_array[:,:,:3]
        if img_array.dtype == 'uint8':
            img = img_array/255
        else:
            img = img_array
        mask = msk[...,None]
        color_mask = np.array([0.2*msk, 0.5*msk, 0.85*msk])
        color_mask = np.transpose(color_mask, (1,2,0))
        blend = 0.3*color_mask + 0.7*img*mask + (1 - mask)*img
#         plt.figure(figsize=(10,10))
#         plt.imshow(blend)
#         plt.show()
        return blend
    def load_checkpoints(self, checkpoint_path):
        assert self.model is not None
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict["net"])


def init_model(model_name):
    if model_name == "ENet":
        model = ENet( num_classes  = 1)
    elif model_name == "ENet_SE":
        model = ENet(num_classes = 1,
                            attention_block= SENet)
    elif model_name == "ENet_CBAM":
        model = ENet(num_classes = 1,
                            attention_block= CBAM)
    elif model_name == "ENet_CBAM":
        model = ENet(num_classes = 1, attention_block= SelfAttentionBlock,
                    ignore_attention=[1, 2, 5, 6])
    else:
        model = None
    
    return model


def center_crop(image):
    h_origin, w_origin = image.shape[:2]
    if w_origin > h_origin:
        w_1 = int(w_origin//2 - h_origin//2)
        w_2 = int(w_origin//2 + h_origin//2)
        return image[:, w_1: w_2] 
    elif w_origin < h_origin:
        h_1 = int(h_origin//2 - w_origin//2)
        h_2 = int(h_origin//2 + w_origin//2)
        return image[h_1: h_2, :]
    else:
        return image


def revert_origin_size(img, h, w):
    min_size = min(h, w)
    img = cv2.resize(img, (min_size, min_size))
    if h > min_size:
        pad_top = (h-min_size)//2
        pad_bot = h - min_size - pad_top
        pad_left = 0
        pad_right = 0
    elif w > min_size:
        pad_top = 0 
        pad_bot = 0 
        pad_left = (w-min_size)//2
        pad_right = w - min_size - pad_left
    else:
        pad_top = 0
        pad_bot = 0
        pad_left = 0
        pad_right = 0
    img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right, 
        value=0, borderType=cv2.BORDER_CONSTANT)
    
    return img


def getLargestCC(segmentation):
    try:
        labels = label(segmentation)
        count = np.bincount(labels.flat, weights=segmentation.flat)
        if len(count) < 2:
            return segmentation
        l1 = np.argsort(count)
        largest1 = labels == l1[-1]
    
        if count[-2] ==0:
            return largest1
        else:
            largest2 = labels == l1[-2]
            return largest1 + largest2
    except Exception as e:
        print(e)
