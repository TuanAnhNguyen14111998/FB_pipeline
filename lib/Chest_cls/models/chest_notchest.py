# -*- coding: utf-8 -*-
"""
Created on 7/09/2020 5:52 pm

@author: Soan Duong, UOW
"""
# Standard library imports
import numpy as np
from collections import namedtuple

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local application imports
import lib.Chest_cls.models
from lib.Chest_cls.models.global_pool import GlobalPool
from lib.Chest_cls.models.attention_map import AttentionMap
from lib.Chest_cls.models.feature_extraction.densenet import densenet121, densenet161, densenet169, densenet201


# Define the chest-notchest network
class CNCNet(nn.Module):
    def __init__(self, cfg):
        """
        Initialize an instance of the chest-notchest network
        Args:
            cfg: dictionary of the network parameters
        """
        super(CNCNet, self).__init__()
        # Convert cfg from dictionary to a class object
        cfg = namedtuple('cfg', cfg.keys())(*cfg.values())
        self.cfg = cfg

        # Set the backbone for the network
        self.avai_backbones = self.get_backbones()
        if self.cfg.backbone not in self.avai_backbones.keys():
            raise KeyError(
                'Invalid backbone name. Received "{}", but expected to be one of {}'.format(
                    self.cfg.backbone, self.avai_backbones.keys()))

        self.backbone = self.avai_backbones[cfg.backbone][0](cfg)
        self.backbone_type = self.avai_backbones[self.cfg.backbone][1]

        # Set to not train backbone's parameters if necessary
        if self.cfg.bb_freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Set globalbool for this class
        self.global_pool = GlobalPool(cfg)

        # Set the number of classes, it is the number of network's output
        self.n_classes = len(cfg.labels)
        # if self.n_classes == 2:     # if binary classification
        #     self.n_classes = 1      # number of network's output is 1.
        self.n_maps = 1
        # print('No. classes: ', self.n_classes)

        self.expand = 1
        if cfg.global_pool == 'AVG_MAX':
            self.expand = 2
        elif cfg.global_pool == 'AVG_MAX_LSE':
            self.expand = 3

        # Get the number of output features of the backbone
        self.n_out_features = self.backbone.num_features

        # Set the classifier
        if cfg.conv_fc:
            fc = getattr(lib.Chest_cls.models.common, 'conv1x1')
        else:
            fc = getattr(nn, 'Linear')
        self.fc = fc(self.n_out_features * self.expand,
                     self.n_classes * self.n_maps, bias=True)

        # Initialize the classifier
        classifier = getattr(self, "fc")
        if isinstance(classifier, nn.Conv2d):
            classifier.weight.data.normal_(0, 0.01)
            classifier.bias.data.zero_()

        # Initialize the batchnorm for the output features
        self.bn = nn.BatchNorm2d(self.n_out_features * self.expand)

        # Initialize the attentionmap for the output features
        self.attention_map = AttentionMap(self.cfg, self.n_out_features)

    def forward(self, x):
        """
        Args:
            x: Tensor of size (batchsize, n_channels, H, W)

        Returns: logit of

        """
        # Get the output of the backbone
        feature_map = self.backbone(x)          # of size (batchsize, n_out_features, 7, 7)

        # Get the output of the global pool
        feat = self.global_pool(feature_map)    # of size (batchsize, n_out_features, 1, 1)

        # Get the output of batchnorm if required
        if self.cfg.fc_bn:
            bn = getattr(self, "bn")
            feat = bn(feat)                     # of size (batchsize, n_out_features, 1, 1)

        # Get the output of the dropout, of size (batchsize, n_out_features, 1, 1)
        feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)

        # Get the output of the classifier, of size (batchsize, n_classes)
        classifier = getattr(self, "fc")
        if self.cfg.conv_fc:
            if self.cfg.wildcat:
                logits = classifier(feat)
                logits = self.spatial_pooling(logits)
            else:
                logits = classifier(feat).squeeze(-1).squeeze(-1)

        else:
            logits = classifier(feat.view(feat.size(0), -1))

        return logits

    @staticmethod
    def get_backbones():
        """
        Returns: dictionary of famous networks
        """
        __factory = {'densenet121': [densenet121, 'densenet'],
                     'densenet161': [densenet161, 'densenet'],
                     'densenet169': [densenet169, 'densenet'],
                     'densenet201': [densenet201, 'densenet'], }
        return __factory


if __name__ == "__main__":
    import yaml
    # load the config file
    nn_config_path = '../configs/base.yml'
    with open(nn_config_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Generate a random input
    x = torch.rand((2, 3, 224, 224))    # got error when batchsize = 1 at batchnorm
    print(x.shape)

    # Initialize the model
    model = CNCNet(cfg=cfg['model_params'])

    # Compute the output
    y = model(x)
    print(y.shape)