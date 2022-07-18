# -*- coding: utf-8 -*-
"""
Created on 3/09/2020 8:47 pm

@author: Soan Duong, UOW
"""
# Standard library imports
import os
import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Third party imports
import torch

# Local application imports
from utils.datasets import XRayClassDataset


def init_seeds(seed):
    """

    :param seed:
    :return:
    """
    # Setting seeds
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
        
def prepare_device(n_gpu_use=1):
    """
    Setup GPU device if it is available, move the model into the configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There\'s no GPU available on this machine,"
            "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available "
            "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def get_experiment_dataloaders(cfg):
    """
    Get and return the train, validation, and test dataloaders for the experiment.
    :param cfg: dict that contains the required settings for the dataloaders
                (dataset_dir and train_txtfiles)
    :return: train, validation, and test dataloaders
    """
    m_params = cfg['model_params']
    t_params = cfg['train_params']

    # Generate the train dataloader
    train_dataset = XRayClassDataset(t_params['dataset_dir'], t_params['train_txtfiles'],
                                     m_params, mode='train', n_cutoff_imgs=t_params['n_cutoff_imgs'], 
                                     labels=m_params['labels'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=t_params['train_batch_size'],
                                                   pin_memory=True, num_workers=t_params['num_workers'])

    # Generate the train dataloader
    val_dataset = XRayClassDataset(t_params['dataset_dir'], t_params['val_txtfiles'],
                                   m_params, mode='val', n_cutoff_imgs=t_params['n_cutoff_imgs'], 
                                   labels=m_params['labels'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=t_params['train_batch_size'],
                                                 pin_memory=True, num_workers=t_params['num_workers'])

    # Generate the train dataloader
    test_dataset = XRayClassDataset(t_params['dataset_dir'], t_params['test_txtfiles'],
                                    m_params, mode='test', n_cutoff_imgs=t_params['n_cutoff_imgs'], 
                                    labels=m_params['labels'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=t_params['infer_batch_size'],
                                                  pin_memory=True, num_workers=t_params['num_workers'])

    # Return the dataloaders
    return train_dataloader, val_dataloader, test_dataloader


def init_obj(module_name, module_args, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.
    `object = config.init_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    assert all([k not in module_args for k in
                kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def create_exp_dir(path, visual_folder=False):
    if not os.path.exists(path):
        os.mkdir(path)
        if visual_folder is True:
            os.mkdir(path + '/visual')  # for visual results
    else:
        print("DIR already existed.")
    print('Experiment dir : {}'.format(path))


def get_metrics(y_pr, y_gt, labels):
    """
    Compute performance metrics of y_pr and y_gt
    Args:
        y_pr: 2D array of size (batchsize, n_classes)
        y_gt: 1D array of size (batchsize,)
        labels: list of labels of the classification problem
    Returns: dictionary of metrics:
    """

    
    if len(labels) == 2:
        # Get the prob. of label-1 class
        y_pr = y_pr[:, 1]
        auc = roc_auc_score(y_true=y_gt, y_score=y_pr)

        # Get the output labels of the y_pr
        threshold = 0.5
        y_pr[y_pr >= threshold] = 1.0
        y_pr[y_pr < threshold] = 0.0
        accuracy = accuracy_score(y_true=y_gt, y_pred=y_pr)
        precision = precision_score(y_true=y_gt, y_pred=y_pr, pos_label=1, average='binary')
        recall = recall_score(y_true=y_gt, y_pred=y_pr, pos_label=1, average='binary')
        f1_score = fbeta_score(y_true=y_gt, y_pred=y_pr, beta=1, pos_label=1, average='binary')
        f2_score = fbeta_score(y_true=y_gt, y_pred=y_pr, beta=2, pos_label=1, average='binary')

    else:
        # Compute the one-hot coding of the y-gt
        try: 
            y_onehot = np.zeros(y_pr.shape)
            for k in range(len(y_gt)):
                y_onehot[k, y_gt[k]] = 1
            auc = roc_auc_score(y_true=y_onehot, y_score=y_pr)
        
        except Exception: # error when not all classes presented in y_gt
            auc = 0

        # Get the output labels of the y_pr
        y_pr = np.argmax(y_pr, axis=1)
        accuracy = accuracy_score(y_true=y_gt, y_pred=y_pr)
        precision = precision_score(y_true=y_gt, y_pred=y_pr, labels=labels, average='macro')
        recall = recall_score(y_true=y_gt, y_pred=y_pr, pos_label=1, labels=labels, average='macro')
        f1_score = fbeta_score(y_true=y_gt, y_pred=y_pr, beta=1, labels=labels, average='macro')
        f2_score = fbeta_score(y_true=y_gt, y_pred=y_pr, beta=1, labels=labels, average='macro')

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1_score': f1_score, 'f2_score': f2_score, 'auc': auc}
