from skimage.measure import regionprops
import cv2
import numpy as np

from utils.utils import *


def inside_lung_process(mask_doctor=None, lung_segmentation=None, other_segmentation=None, thresh_rm_small = None):
    threshold = 0.02
    if len(mask_doctor.shape) == 3:
        mask_doctor = mask_doctor[0]
    
    if thresh_rm_small != None:
        (area, h, w) = thresh_rm_small
        a = np.sum(mask_doctor)
#         (ymin, ymax, xmin, xmax) = xy_infor(mask_doctor)
#         h = ymax - ymin
#         w = xmax - xmin
        if a < area:
            return False
        else:
            if iou(mask_doctor, lung_segmentation) > threshold:
                return True
            else:
                return False
    else:
        if iou(mask_doctor, lung_segmentation) > threshold:
            return True
        else:
            return False




# def cardiomegaly_process(mask_doctor=None, lung_segmentation=None, other_segmentation=None):
#     (ymin_heart, ymax_heart, xmin_heart, xmax_heart) = xy_infor(mask_doctor)
#     ymin_lung, ymax_lung, xmin_lung, xmax_lung = xy_infor(lung_segmentation)
    
#     base_distance = (xmax_heart - xmin_heart)
    
#     if xmin_heart - xmin_lung < xmin_lung  or xmax_heart > xmax_lung:
#         return False
# #     elif ymin_heart < ymin_lung + (ymax_lung - ymin_lung) / 3:
#     elif ymin_heart < ymin_lung:
#         return False
#     else:
# #         if (xmax_heart - xmin_heart) / (xmax_lung - xmin_lung) < 0.2:
# #             return False
# #         else:
#         return True



# def cardiomegaly_process(mask_doctor=None, lung_segmentation=None, other_segmentation=None):
#     (ymin_heart, ymax_heart, xmin_heart, xmax_heart) = xy_infor(mask_doctor)
#     ymin_lung, ymax_lung, xmin_lung, xmax_lung = xy_infor(lung_segmentation)
    
#     base_distance = (xmax_lung - xmin_lung)
    
#     if xmin_heart > xmin_lung and xmax_heart < xmax_lung:
#         return True
# #     elif ymin_heart < ymin_lung + (ymax_lung - ymin_lung) / 3:
#     elif xmin_heart > xmin_lung and (xmax_heart  - xmax_lung) / base_distance  < 0.6
#         return True 
    
#     else:
# #         if (xmax_heart - xmin_heart) / (xmax_lung - xmin_lung) < 0.2:
# #             return False
# #         else:
#         return True





def fracture_process(mask_doctor=None, lung_segmentation=None, other_segmentation=None, thresh_rm_small=None):
    threshold = 0.05
    lung_segmentation = fill_lungborder(lung_border = lung_segmentation, other_segment = other_segmentation)
#     thresh_rm_small = None
#     if inside_lung_process(mask_doctor, lung_segmentation, thresh_rm_small = thresh_rm_small):
#         return True
#     else:
    combine_mask = lung_segmentation + other_segmentation
    combine_mask = combine_mask.clip(0, 1).astype("uint8")

    if iou(mask_doctor, combine_mask) > threshold:
        return True
    else:
        return False


def pneumothorax_process(mask_doctor = None, lung_segmentation = None, other_segmentation=None):
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(lung_segmentation, kernel, iterations=30)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=20)

    lung_border = img_dilation.copy()
    lung_border[(img_dilation == 1) & (img_erosion != 1)] = 1
    lung_border[~((img_dilation == 1) & (img_erosion != 1))] = 0
    props = regionprops(lung_border)
    
    min_x = 1e5
    min_y = 1e5
    max_x = 0
    max_y = 0
    for prop in props:
        if prop.bbox[1] < min_x:
            min_x = prop.bbox[1]
        if prop.bbox[0] < min_y:
            min_y = prop.bbox[0]
        if prop.bbox[3] > max_x:
            max_x = prop.bbox[3]
        if prop.bbox[2] > max_y:
            max_y = prop.bbox[2]
    w_bbx = max_x - min_x
    
    lung_border[:, int(min_x + 1.5*w_bbx / 4):int(min_x + 2.5 * w_bbx / 4)] = 0

    if np.sum(lung_border * mask_doctor):
        return True
    
    return False


def pleural_effusion_process(mask_doctor=None, lung_segmentation=None, other_segmentation=None):
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(lung_segmentation, kernel, iterations=30)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=20)

    lung_border = img_dilation.copy()
    lung_border[(img_dilation == 1) & (img_erosion != 1)] = 1
    lung_border[~((img_dilation == 1) & (img_erosion != 1))] = 0
    props = regionprops(lung_border)
    
    min_x = 1e5
    min_y = 1e5
    max_x = 0
    max_y = 0
    for prop in props:
        if prop.bbox[1] < min_x:
            min_x = prop.bbox[1]
        if prop.bbox[0] < min_y:
            min_y = prop.bbox[0]
        if prop.bbox[3] > max_x:
            max_x = prop.bbox[3]
        if prop.bbox[2] > max_y:
            max_y = prop.bbox[2]
    h_bbx = max_y - min_y
    
    lung_border[int(min_y + h_bbx / 4):int(min_y + 3 * h_bbx / 4), :] = 0

    if np.sum(lung_border * mask_doctor):
        return True
    
    return False


def mediastinum_process(mask_doctor=None, lung_segmentation=None, other_segmentation=None):
    (ymin_medias, ymax_medias, xmin_medias, xmax_medias) = xy_infor(mask_doctor)
    ymin_lung, ymax_lung, xmin_lung, xmax_lung = xy_infor(lung_segmentation)
    if xmin_medias < xmin_lung or xmax_medias > xmax_lung or ymin_medias < ymin_lung:
        return False
    else:
        return True
