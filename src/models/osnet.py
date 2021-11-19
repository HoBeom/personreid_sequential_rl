from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torchreid.utils import FeatureExtractor

class OsNet(nn.Module):
    def __init__(self, num_classes):
        super(OsNet, self).__init__()
        
        self.base = FeatureExtractor(
            'osnet_x1_0',
            model_path='../data/trained_model/osnet_x1_0_duke_market_trained.pth.tar-250',
        )
        # feature extractor
        self.feat_dim = 512

        # classifier
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))

        # collect individual frame features and global video features by avg pooling
        individual_img_features = self.base(x) # t 512
        # individual_img_features = F.avg_pool2d(spatial_out, spatial_out.shape[2:]).view(b*t, -1)

        # format into video, sequence way
        individual_img_features = individual_img_features.view(b, t, -1) # b t 512

        # prepare for video level features
        individual_features_permuted = individual_img_features.permute(0,2,1) # b 512 t
        video_features = F.avg_pool1d(individual_features_permuted, t) # b 512
        video_features = video_features.view(b, self.feat_dim) # b 512

        if not self.training:
            return video_features, individual_img_features
        
        y = self.classifier(video_features)
        return y, video_features, individual_img_features
