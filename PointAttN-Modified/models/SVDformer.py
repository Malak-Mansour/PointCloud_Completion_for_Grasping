from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import gather_operation as gather_points
import time
from models.model_utils import *
from utils.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=256):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points

class SDG(nn.Module):
    def __init__(self, channel=128,ratio=1,hidden_dim=768):
        super(SDG, self).__init__()
        self.channel = channel
        self.hidden = hidden_dim

        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = self_attention(channel*2,hidden_dim,dropout=0.0,nhead=8)
        self.cross1 = cross_attention(hidden_dim, hidden_dim, dropout=0.0,nhead=8)

        self.decoder1 = SDG_Decoder(hidden_dim,channel,ratio)

        self.decoder2 = SDG_Decoder(hidden_dim,channel,ratio)

        self.relu = nn.GELU()
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_delta = nn.Conv1d(channel, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(hidden_dim, channel*ratio, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.mlpp = MLP_CONV(in_channel=832,layer_dims=[hidden_dim])
        self.sigma_d = 0.2
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.cd_distance = chamfer_3DDist()

        self.fusionMlp = MLP_CONV(in_channel=hidden_dim*2+channel,layer_dims=[hidden_dim])

    def forward(self, local_feat, coarse,f_g,partial):
        batch_size, _, N = coarse.size()
        F = self.conv_x1(self.relu(self.conv_x(coarse)))
        f_g = self.conv_1(self.relu(self.conv_11(f_g)))
        F = torch.cat([F, f_g.repeat(1, 1, F.shape[-1])], dim=1)

        # Structure Analysis
        half_cd = self.cd_distance(coarse.transpose(1, 2).contiguous(), partial.transpose(1, 2).contiguous())[
                      0] / self.sigma_d
        embd = self.embedding(half_cd).reshape(batch_size, self.hidden, -1).permute(2, 0, 1)
        F_Q = self.sa1(F,embd)
        F_Q_ = self.decoder1(F_Q,embd)

        f_g_current = torch.max(F_Q,2)[0]

        # Similarity Alignment
        local_feat = self.mlpp(local_feat)
        F_H= self.cross1(F_Q,local_feat)
        F_H_ = self.decoder2(F_H,embd)

        # Path Selection
        score = self.fusionMlp(torch.cat([F_Q_+F_H_,f_g_current.unsqueeze(2).repeat(1,1,F_Q_.size(2)),f_g.repeat(1,1,F_Q_.size(2))],1))
        score = torch.sigmoid(score)
        F_L = score * F_Q_ + (1 - score) * F_H_


        F_L = self.conv_delta(self.conv_ps(F_L).reshape(batch_size,-1,N*self.ratio))
        O_L = self.conv_out(self.relu(self.conv_out1(F_L)))
        fine = coarse.repeat(1,1,self.ratio) + O_L

        return fine,F_L

class SDG_l(nn.Module):
    def __init__(self, channel=128,ratio=1,hidden_dim=512):
        super(SDG_l, self).__init__()
        self.channel = channel
        self.hidden = hidden_dim

        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = self_attention(channel*2,hidden_dim,dropout=0.0,nhead=8)
        self.cross1 = cross_attention(hidden_dim, hidden_dim, dropout=0.0,nhead=8)

        self.decoder1 = SDG_Decoder(hidden_dim,channel,ratio)

        self.decoder2 = SDG_Decoder(hidden_dim,channel,ratio)

        self.relu = nn.GELU()
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_delta = nn.Conv1d(channel, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(hidden_dim, channel*ratio, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.mlpp = MLP_CONV(in_channel=832,layer_dims=[hidden_dim])
        self.sigma_d = 0.2
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.cd_distance = chamfer_3DDist()

        self.fusionMlp = MLP_CONV(in_channel=hidden_dim*2+channel*2,layer_dims=[hidden_dim])

    def forward(self, local_feat, coarse,f_g,partial,F_L_Pre):
        batch_size, _, N = coarse.size()
        F = self.conv_x1(self.relu(self.conv_x(coarse)))
        f_g = self.conv_1(self.relu(self.conv_11(f_g)))
        F = torch.cat([F, f_g.repeat(1, 1, F.shape[-1])], dim=1)

        # Structure Analysis
        half_cd = self.cd_distance(coarse.transpose(1, 2).contiguous(), partial.transpose(1, 2).contiguous())[
                      0] / self.sigma_d
        embd = self.embedding(half_cd).reshape(batch_size, self.hidden, -1).permute(2, 0, 1)
        F_Q = self.sa1(F,embd)
        F_Q_ = self.decoder1(F_Q,embd)

        f_g_current = torch.max(F_Q,2)[0]

        # Similarity Alignment
        local_feat = self.mlpp(local_feat)
        F_H = self.cross1(F_Q,local_feat)
        F_H_ = self.decoder2(F_H,embd)

        # Path Selection
        score = self.fusionMlp(torch.cat([F_Q_+F_H_,F_L_Pre,f_g_current.unsqueeze(2).repeat(1,1,F_Q_.size(2)),f_g.repeat(1,1,F_Q_.size(2))],1))
        score = torch.sigmoid(score)
        F_L = score * F_Q_ + (1 - score) * F_H_

        F_L = self.conv_delta(self.conv_ps(F_L).reshape(batch_size,-1,N*self.ratio))
        O_L = self.conv_out(self.relu(self.conv_out1(F_L)))
        fine = coarse.repeat(1,1,self.ratio) + O_L

        return fine

class local_encoder(nn.Module):
    def __init__(self,cfg):
        super(local_encoder, self).__init__()
        self.gcn_1 = EdgeConv(3, 64, 16)
        self.gcn_2 = EdgeConv(64, 256, 8)
        self.gcn_3 = EdgeConv(256, 512, 4)
        self.local_number = cfg.NETWORK.local_points

    def forward(self,input):
        x1 = self.gcn_1(input)
        idx = furthest_point_sample(input.transpose(1, 2).contiguous(), self.local_number)
        x1 = gather_points(x1,idx)

        x2 = self.gcn_2(x1)

        x3 = self.gcn_3(x2)

        return torch.cat([x1,x2,x3],1)


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        self.encoder = SVFNet(cfg)
        self.localencoder = local_encoder(cfg)
        self.merge_points = cfg.NETWORK.merge_points
        self.refine1 = SDG(ratio=cfg.NETWORK.step1,hidden_dim=768,dataset=cfg.DATASET.TEST_DATASET)
        self.refine2 = SDG(ratio=cfg.NETWORK.step2,hidden_dim=512,dataset=cfg.DATASET.TEST_DATASET)

    def forward(self, partial,depth):
        partial = partial.transpose(1,2).contiguous()
        feat_g, coarse = self.encoder(partial,depth)
        local_feat = self.localencoder(partial)

        coarse_merge = torch.cat([partial,coarse],dim=2)
        coarse_merge = gather_points(coarse_merge, furthest_point_sample(coarse_merge.transpose(1, 2).contiguous(), self.merge_points))

        fine1 = self.refine1(local_feat, coarse_merge, feat_g,partial)
        fine2 = self.refine2(local_feat, fine1, feat_g,partial)

        return (coarse.transpose(1, 2).contiguous(),fine1.transpose(1, 2).contiguous(),fine2.transpose(1, 2).contiguous())