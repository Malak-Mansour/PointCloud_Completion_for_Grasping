from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np
import open3d as o3d
from utils.model_utils import *

# from utils.mm3d_pn2 import furthest_point_sample, gather_points
from utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import gather_operation as gather_points
from utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.hypercd_utils.loss_utils import get_loss1

from config_pcn import cfg

class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]


        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        src1 = src1.permute(1, 2, 0)

        return src1


class PCT_refine(nn.Module):
    def __init__(self, channel=128,ratio=1):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = cross_transformer(channel*2,512)
        self.sa2 = cross_transformer(512,512)
        self.sa3 = cross_transformer(512,channel*ratio)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio, channel*ratio, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)


    def forward(self, x, coarse,feat_g):
        batch_size, _, N = coarse.size()

        y = self.conv_x1(self.relu(self.conv_x(coarse)))  # B, C, N
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        y0 = torch.cat([y,feat_g.repeat(1,1,y.shape[-1])],dim=1)

        y1 = self.sa1(y0, y0)
        y2 = self.sa2(y1, y1)
        y3 = self.sa3(y2, y2)
        y3 = self.conv_ps(y3).reshape(batch_size,-1,N*self.ratio)

        y_up = y.repeat(1,1,self.ratio)
        y_cat = torch.cat([y3,y_up],dim=1)
        y4 = self.conv_delta(y_cat)

        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1,1,self.ratio)

        return x, y3


from torch_geometric.nn import EdgeConv as TG_EdgeConv  # EdgeConv from torch_geometric
from torch_geometric.nn import global_max_pool, knn_graph  # Global max pooling from torch_geometric

# Simple Multi-Scale Transformer implementation (custom)
class MultiScaleTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2):
        super(MultiScaleTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )

    def forward(self, x):
        # x: (B, C, N) -> transpose to (N, B, C) for MultiheadAttention
        x = x.transpose(1, 2).contiguous()  # (N, B, C)
        for attn in self.layers:
            attn_output, _ = attn(x, x, x)  # Self-attention
            x = self.norm(x + attn_output)  # Residual connection + normalization
            ffn_output = self.ffn(x)
            x = self.norm(x + ffn_output)  # Feed-forward + residual
        return x.transpose(1, 2).contiguous()  # Back to (B, C, N)


    
class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = cross_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = cross_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)
        self.sa3_1 = cross_transformer((channel)*8,channel*8)

        self.relu = nn.GELU()


        self.sa0_d = cross_transformer(channel*8,channel*8)
        self.sa1_d = cross_transformer(channel*8,channel*8)
        self.sa2_d = cross_transformer(channel*8,channel*8)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)


    def forward(self, points):
        batch_size, _, N = points.size()
        

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        # GDP
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = gather_points(x0, idx_0)
        points = gather_points(points, idx_0)
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        x1 = self.sa1_1(x1,x1).contiguous()
        # GDP
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous()
        # GDP
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)
        # SFA
        x3 = self.sa3_1(x3,x3).contiguous()
        
        # seed generator
        
        # maxpooling

        #print(x3.shape)
        
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        # print('x_g shape:', x_g.shape)
        # the shape of features is 1 / point so I can put it here
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d))
        # Calculate the size that preserves the total number of elements
        total_elements = x2_d.numel() // batch_size
        new_dim = total_elements // (self.channel * 4)
        x2_d = x2_d.reshape(batch_size, self.channel * 4, new_dim)


        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))

        return x_g, fine




from models.SVDformer import SDG, local_encoder, SDG_l

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.dataset == 'pcn':
            step1 = 4
            step2 = 8
        elif args.dataset == 'c3d':
            step1 = 1
            step2 = 4
        else:
            ValueError('dataset does not exist')

        print("point SFA init..")
        self.encoder = PCT_encoder()

        self.localencoder = local_encoder(cfg)
        # self.refine = SDG(ratio=cfg.NETWORK.step1,hidden_dim=768,dataset=cfg.DATASET.TEST_DATASET)
        # self.refine1 = SDG(ratio=cfg.NETWORK.step2,hidden_dim=512,dataset=cfg.DATASET.TEST_DATASET)
        self.refine = SDG(ratio=cfg.NETWORK.step1)
        self.refine1 = SDG_l(ratio=cfg.NETWORK.step2)

        # self.refine = PCT_refine(ratio=step1)
        # self.refine1 = PCT_refine(ratio=step2)



    
    def forward(self, x, gt=None, is_training=True):
        feat_g, coarse = self.encoder(x)
        # print('coarse shape:', coarse.shape)
        #print(feat_g.shape)
        local_feat = self.localencoder(x)
        # new_x = torch.cat([x,coarse],dim=2)
        # new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))

        new_x  = coarse
        # fine = self.refine(local_feat, new_x, feat_g,x)
        # fine1 = self.refine1(local_feat, fine, feat_g,x)

        fine, F_L_1 = self.refine(local_feat, new_x, feat_g,x)
        fine1 = self.refine1(local_feat, fine, feat_g,fine,F_L_1)



        coarse = coarse.transpose(1, 2).contiguous()
        # print(f"shape of gt {gt.shape}, shape of fine1 {fine1.shape, fine.shape}, shape of coarse {new_x.shape}")
        indices_gt = furthest_point_sample(gt, coarse.shape[1])
        gt_coarse = gather_points(gt.transpose(1, 2).contiguous(), indices_gt).transpose(1, 2).contiguous()
       

        indices = furthest_point_sample(fine.transpose(1, 2).contiguous(), gt.shape[1])
        fine = gather_points(fine, indices).transpose(1, 2).contiguous()

        indices1 = furthest_point_sample(fine1.transpose(1, 2).contiguous(), gt.shape[1])
        fine1 = gather_points(fine1, indices1).transpose(1, 2).contiguous()

        enable_hypercd = True


        if is_training:

            if enable_hypercd:
                pcds_pred = [coarse, fine, fine1, fine1]  # Using fine1 twice as P3 (finest resolution)
                # Call get_loss1 with appropriate parameters
                loss_all, losses = get_loss1(pcds_pred, x.transpose(1, 2).contiguous(), gt, sqrt=True)
                # Unpack individual losses for logging or other purposes
                cdc, cd1, cd_hyp, partial_matching = losses
                loss2 = cd_hyp
                # Add penalty loss to the total loss
                total_train_loss = loss_all 
                
                # For compatibility with the return statement
                
            else:            
                
                loss3, _ = calc_cd(fine1, gt)
                loss2, _ = calc_cd(fine, gt)
                loss1, _ = calc_cd(coarse, gt_coarse)

                total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()

            return fine, loss2, total_train_loss
            

        else:
            if gt is not None:

            # Call get_loss1 with appropriate parameters
                if enable_hypercd:

                    pcds_pred = [coarse, fine, fine1, fine1]  # Using fine1 twice as P3 (finest resolution)
                    loss_all, losses = get_loss1(pcds_pred, x.transpose(1, 2).contiguous(), gt, sqrt=True)
                    cdc, cd1, cd_hyp, partial_matching = losses
                else: # set hyper_cd to 0 if it was not enabled 
                    cd_hyp = torch.tensor(0.0).cuda()  

                cd_p, cd_t, f1 = calc_cd(fine1, gt, calc_f1=True)
                cd_p_coarse, cd_t_coarse = calc_cd(coarse, gt_coarse)
              
                

                return {'out1': coarse, 'out2': fine1, 'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse, 'cd_p': cd_p, 'cd_t': cd_t, "cd_hyp":cd_hyp, "f1": f1}
            else:
                return {'out1': coarse, 'out2': fine1}