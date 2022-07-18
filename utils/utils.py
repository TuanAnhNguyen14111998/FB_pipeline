from turtle import width
import cv2
import torch
import numpy as np
import json
import urllib
import pandas as pd
from skimage.measure import label as skimage_label
from torchvision import transforms
from PIL import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def xy_infor(border):
    if len(border.shape) == 3:
        border = border[0]
    y, x = np.where(border)
    ymin, ymax, xmin, xmax = y.min(), y.max(), x.min(), x.max()
    return (ymin, ymax, xmin, xmax)


def getLargestCC(segmentation):
    labels = skimage_label(segmentation)
    count = np.bincount(labels.flat, weights=segmentation.flat)
    l1 = np.argsort(count)
    largest1 = labels == l1[-1]
    if count[-2] ==0:
        return largest1
    else:
        largest2 = labels == l1[-2]
        return largest1 + largest2


def simple_dilate(lung_segmentation, kernel = np.ones((100,100), np.uint8)):
    lung = cv2.dilate(lung_segmentation, kernel, iterations=2)
    return lung
# def lung_border(lung_raw=None):
#     lung_nested = getLargestCC(lung_raw).astype('uint8')
#     countour, _ = cv2.findContours(lung_nested,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#     h, w = lung_nested.shape
#     if len(countour) > 1:
#         countour = sorted(countour, key = lambda x: x.shape[0], reverse = True)[:2]
#         right, left = countour[0][:, 0, :], countour[1][:, 0, :]
#         left_mask = np.zeros((h, w))
#         right_mask = np.zeros((h, w))
#         cv2.fillPoly(left_mask,  pts = [left], color =(1,1,1))
#         cv2.fillPoly(right_mask, pts = [right], color =(1,1,1))
#         if np.sum(left_mask) / np.sum(right_mask) > 1.2:
#             lung_nested = (left_mask + cv2.flip(left_mask, 1)).clip(0,1).astype('uint8')
#         elif np.sum(left_mask) / np.sum(right_mask) < 1 / 1.2:
#             lung_nested = (right_mask + cv2.flip(right_mask, 1)).clip(0,1).astype('uint8')
#     elif len(countour) == 1:
#         single_points = countour[0][:, 0, :]
#         single_mask = np.zeros((h, w))
#         cv2.fillPoly(single_mask,  pts = [single_points], color =(1,1,1))
#         lung_nested = (single_mask + cv2.flip(single_mask, 1)).clip(0,1).astype('uint8')
#     else:
#         pass
    
#     return lung_nested

# def lung_border(lung_raw=None):
#     lung_nested = getLargestCC(lung_raw).astype('uint8')
#     countour, _ = cv2.findContours(lung_nested,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#     h, w = lung_nested.shape
#     if len(countour) > 1:
#         countour = sorted(countour, key = lambda x: x.shape[0], reverse = True)[:2]
#         a, b = countour[0][:, 0, :], countour[1][:, 0, :]

#         mask_raw = np.zeros((h, w))
#         if a[:, 0].min() < b[:, 0].min():
#             left = a
#             right = b
#         else:
#             left = b
#             right = a

#         left_mask = np.zeros((h, w))
#         right_mask = np.zeros((h, w))

#         cv2.fillPoly(left_mask,  pts = [left], color =(1,1,1))
#         cv2.fillPoly(right_mask, pts = [right], color =(1,1,1))

#         if np.sum(left_mask) / np.sum(right_mask) > 8 or np.sum(left_mask) / np.sum(right_mask) <  1/8:
#             mask_raw = lung_nested
#         else:
#             if np.sum(left_mask) / np.sum(right_mask) > 1.2:
#                 try:
#                     left = left_mask[:, :left[:, 0].max() + 30]
#                     combine = np.hstack((left, cv2.flip(left, 1)))
#                     h_combine, w_combine = combine.shape
#                     mask_raw[:, :w_combine] = combine
#                 except:
# #                     mask_raw = lung_nested
#                     print('CHECKK')
#             elif np.sum(left_mask) / np.sum(right_mask) < 1/1.3:
#                 try:
#                     right = right_mask[:, right[:, 0].min() - 30 :]
#                     combine = np.hstack((cv2.flip(right, 1), right))
#                     h_combine, w_combine = combine.shape
#                     mask_raw[:, w-w_combine: ] = combine
#                 except:
#                     mask_raw = lung_nested
#             else:
#                 mask_raw = lung_nested
#         return mask_raw
#     else:
#         return lung_nested


        
def lung_border(lung_raw=None):
    lung_nested = getLargestCC(lung_raw).astype('uint8')
    countour, _ = cv2.findContours(lung_nested,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    h, w = lung_nested.shape
    if len(countour) > 1:
        countour = sorted(countour, key = lambda x: x.shape[0], reverse = True)[:2]
        a, b = countour[0][:, 0, :], countour[1][:, 0, :]

        mask_raw = np.zeros((h, w))
        if a[:, 0].min() < b[:, 0].min():
            left = a
            right = b
        else:
            left = b
            right = a

        left_mask = np.zeros((h, w))
        right_mask = np.zeros((h, w))

        cv2.fillPoly(left_mask,  pts = [left], color =(1,1,1))
        cv2.fillPoly(right_mask, pts = [right], color =(1,1,1))

        if np.sum(left_mask) / np.sum(right_mask) > 8 or np.sum(left_mask) / np.sum(right_mask) <  1/8:
#             mask_raw = lung_nested
            mask_raw = (lung_nested + cv2.flip(lung_nested, 1)).clip(0, 1).astype("uint8")
        else:
            if np.sum(left_mask) / np.sum(right_mask) > 1.1:
                try:
                    left = left_mask[:, :left[:, 0].max() + 30]
                    combine = np.hstack((left, cv2.flip(left, 1)))
                    h_combine, w_combine = combine.shape
                    mask_raw[:, :min(w_combine, w)] = combine[:, :min(w_combine, w)]
                    mask_raw = (mask_raw + lung_nested).clip(0,1).astype("uint8")
                except:
                    mask_raw = (lung_nested + cv2.flip(lung_nested, 1)).clip(0, 1).astype("uint8")
            elif np.sum(left_mask) / np.sum(right_mask) < 1/1.1:
                try:
                    right = right_mask[:, right[:, 0].min() - 30 :]
                    combine = np.hstack((cv2.flip(right, 1), right))
                    h_combine, w_combine = combine.shape
                    mask_raw[:, max(0, w-w_combine): ] = combine[max(0, w-w_combine), :]
                    mask_raw = (mask_raw + lung_nested).clip(0,1).astype("uint8")
                except:
                    mask_raw = (lung_nested + cv2.flip(lung_nested, 1)).clip(0, 1).astype("uint8")
            else:
                mask_raw = (lung_nested + cv2.flip(lung_nested, 1)).clip(0, 1).astype("uint8")
        return simple_dilate(mask_raw)
    else:
        return simple_dilate((lung_nested + cv2.flip(lung_nested, 1)).clip(0, 1).astype("uint8"))

# def fill_lungborder(lung_border):
#     lung_border = lung_border.astype('uint8')
#     h, w = lung_border.shape
#     y, x = np.where(lung_border)
#     ymin, ymax, xmin, xmax = y.min(), y.max(), x.min(), x.max()
#     y_xmax = y[np.where(x == xmax)[0][0]]
#     y_xmin = y[np.where(x == xmin)[0][0]]
#     ## fill with countour
#     countour, _ = cv2.findContours(lung_border,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#     right, left = countour[0][:, 0, :], countour[1][:, 0, :]
#     y_left = left[:, 1]
#     y_right = right[:, 1]
#     x_left = left[:, 0]
#     x_right = right[:, 0]
#     y_max = max(y_left.max(), y_right.max())
#     x_yleftmin  = x_left[np.where(y_left.min() == y_left)]
#     x_yrightmin = x_right[np.where(y_right.min() == y_right)]
#     top_left = [x_yleftmin[0], y_left.min()]
#     bottom_right = [x_yrightmin[0], y_max]
#     color = (1,0,0)
#     thickness = -1
#     tmp = np.zeros((h, w))
#     tmp = cv2.rectangle(tmp, tuple(top_left), tuple(bottom_right), color, thickness)
#     lung_mask = lung_border + tmp
#     lung_mask = lung_mask.clip(0, 1).astype("uint8")
#     # fill based rectangle
#     tmp = np.zeros((h, w))
#     bottom_right = (xmax, h)
#     top_left = (xmin, min(y_xmin,y_xmax))
#     color = (1,0,0)
#     thickness = -1
#     tmp = cv2.rectangle(tmp, tuple(top_left), tuple(bottom_right), color, thickness)
#     # merge all
#     mask_total = lung_mask + tmp
#     mask_total = mask_total.clip(0, 1).astype("uint8")

#     return mask_total

def fill_lungborder(lung_border = None, other_segment = None):
    lung_border = lung_border.astype('uint8')
    clav_border = other_segment.astype('uint8')
    h, w = lung_border.shape
    y, x = np.where(lung_border)
    ymin, ymax, xmin, xmax = y.min(), y.max(), x.min(), x.max()
    y_xmax = y[np.where(x == xmax)[0][0]]
    y_xmin = y[np.where(x == xmin)[0][0]]


    clav_border = other_segment.astype('uint8')
    y_clav, x_clav = np.where(clav_border)
    xmin_clav, xmax_clav = x_clav.min(), x_clav.max()
    y_xclavmax = y_clav[np.where(x_clav == xmax_clav)[0][0]]
    y_xclavmin = y_clav[np.where(x_clav == xmin_clav)[0][0]]

    # so we have:
    pt1 = (xmin_clav, y_xclavmin)
    pt2 = (xmin, ymax)
    pt3 = ((xmin + xmax)//2, ymin)
    pt4 = (xmax_clav, y_xclavmax)
    pt5 = (xmax, ymax)

    image_0 = image_1 = np.zeros((h, w))
    cv2.circle(image_0, pt1, 2, (1,1,1), -1)
    cv2.circle(image_0, pt2, 2, (1,1,1), -1)
    cv2.circle(image_0, pt3, 2, (1,1,1), -1)
    cv2.circle(image_1, pt4, 2, (1,1,1), -1)
    cv2.circle(image_1, pt5, 2, (1,1,1), -1)
    cv2.circle(image_1, pt5, 2, (1,1,1), -1)

    triangle_left = np.array( [pt1, pt2, pt3] )
    triangle_right = np.array( [pt3, pt4, pt5] )
    cv2.drawContours(image_0, [triangle_left], 0, (1,1,1), -1)
    cv2.drawContours(image_1, [triangle_right], 0, (1,1,1), -1)

    # fill rectangle
    tmp = np.zeros((h, w))
    bottom_right = (xmax, h)
    top_left = (xmin, min(y_xmin,y_xmax))
    color = (1,0,0)
    thickness = -1
    tmp = cv2.rectangle(tmp, tuple(top_left), tuple(bottom_right), color, thickness)
    # merge all
#     mask_total = lung_mask + tmp
#     mask_total = mask_total.clip(0, 1).astype("uint8")

    return (image_0 + image_1 + lung_border + tmp).clip(0, 1).astype('uint8')


def iou(border_A, border_B):
    iou = np.sum(border_A * border_B) / np.sum(border_A)
    return iou


def mask_image(json_path, height, width):
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    label = []
    str_json = json.loads(json_path)
    for item in str_json:
        pts = []
        for poly in item['polygon']:
            pts.append([poly['x'], poly['y']])
        pts = (np.array(pts, np.int32))
        try:
            cv2.fillPoly(mask, [pts], [1, 1, 1])
        except:
            continue
    
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    label.append(mask)
    label = np.array(label)

    return label


def visualize(img_array, msk):
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
    
    return blend


def read_image_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return image


def get_other_segmentation(image=None, model=None):
    other_segmentation = None
    if model:
        other_segmentation = model.predict(image)
    
    return other_segmentation


def save_csv(dictionary_result, path_save):
    df = pd.DataFrame(dictionary_result)
    df.to_csv(path_save, index=False)

    return None


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


def preprocess_chest_cls(image, img_size, crop_size):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    trans = transforms.Compose(
        [transforms.Resize(size=img_size), transforms.CenterCrop(size=crop_size),
         transforms.ToTensor(), normalize])

    return trans(img)


def preprocess_lung_segmentation(img, img_size=(320, 320)):
    if len(img.shape) == 2:
        img = np.dstack([img,]*3)
    elif len(img.shape) == 3:
        img = img[:,:,:3]
    img = center_crop(img)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.540,0.540,0.540), std = (0.264,0.264,0.264)),
        transforms.Resize(size=img_size)
    ])
    img_tensor = transform_test(img)

    return img_tensor
