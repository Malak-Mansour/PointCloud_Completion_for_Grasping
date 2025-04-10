# PointAttn downsampling + SVDformer for generation
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


class StrawberryPCTEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64):
        super(StrawberryPCTEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, embed_dim, 1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim * 2, 1)
        self.relu = nn.ReLU()

        # EdgeConv layers
        self.edgeconv1 = TG_EdgeConv(nn.Sequential(
            nn.Linear(2 * embed_dim * 2, embed_dim * 2),
            nn.ReLU()
        ))
        self.edgeconv2 = TG_EdgeConv(nn.Sequential(
            nn.Linear(2 * embed_dim * 4, embed_dim * 4),
            nn.ReLU()
        ))
        self.edgeconv3 = TG_EdgeConv(nn.Sequential(
            nn.Linear(2 * embed_dim * 8, embed_dim * 8),
            nn.ReLU()
        ))

        # Projection layers
        self.ps_adj = nn.Conv1d(embed_dim * 8, embed_dim * 4, 1)
        self.ps = nn.Conv1d(embed_dim * 4, embed_dim * 2, 1)
        self.ps_refuse = nn.Conv1d(embed_dim * 2, embed_dim, 1)

        # Additional transformer layers for coarse point cloud generation
        self.sa0_d = cross_transformer(embed_dim, embed_dim)
        self.sa1_d = cross_transformer(embed_dim, embed_dim)
        self.sa2_d = cross_transformer(embed_dim, embed_dim)

        # Layers for coarse point cloud
        self.conv_out1 = nn.Conv1d(embed_dim, embed_dim * 4, 1)
        self.conv_out = nn.Conv1d(embed_dim * 4, 3, 1)

    def forward(self, points):
        batch_size, _, N = points.size()

        # Initial convolutions
        x = self.relu(self.conv1(points))  # (B, D, N)
        x0 = self.conv2(x)  # (B, 2D, N)

        # FPS sampling to N//4
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = gather_points(x0, idx_0)  # (B, 2D, N//4)
        points_g0 = gather_points(points, idx_0)  # (B, 3, N//4)

        # Generate edge_index for sampled points
        points_g0_flat = points_g0.transpose(1, 2).reshape(-1, 3)  # (B*N//4, 3)
        batch_g0 = torch.arange(batch_size, device=points.device).repeat_interleave(N // 4)
        edge_index_0 = knn_graph(points_g0_flat, k=20, batch=batch_g0)
        x0_flat = x_g0.transpose(1, 2).reshape(-1, x_g0.size(1))  # (B*N//4, 2D)
        x1 = self.edgeconv1(x0_flat, edge_index_0)
        x1 = x1.view(batch_size, N // 4, -1).transpose(1, 2)  # (B, 2D, N//4)
        x1 = torch.cat([x_g0, x1], dim=1)  # (B, 4D, N//4)

        # FPS sampling to N//8
        idx_1 = furthest_point_sample(points_g0.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)  # (B, 4D, N//8)
        points_g1 = gather_points(points_g0, idx_1)  # (B, 3, N//8)

        # Generate edge_index
        points_g1_flat = points_g1.transpose(1, 2).reshape(-1, 3)  # (B*N//8, 3)
        batch_g1 = torch.arange(batch_size, device=points.device).repeat_interleave(N // 8)
        edge_index_1 = knn_graph(points_g1_flat, k=20, batch=batch_g1)
        x1_flat = x_g1.transpose(1, 2).reshape(-1, x_g1.size(1))  # (B*N//8, 4D)
        x2 = self.edgeconv2(x1_flat, edge_index_1)
        x2 = x2.view(batch_size, N // 8, -1).transpose(1, 2)  # (B, 4D, N//8)
        x2 = torch.cat([x_g1, x2], dim=1)  # (B, 8D, N//8)

        # FPS sampling to N//16
        idx_2 = furthest_point_sample(points_g1.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)  # (B, 8D, N//16)

        # Generate edge_index
        points_g2_flat = gather_points(points_g1, idx_2).transpose(1, 2).reshape(-1, 3)  # (B*N//16, 3)
        batch_g2 = torch.arange(batch_size, device=points.device).repeat_interleave(N // 16)
        edge_index_2 = knn_graph(points_g2_flat, k=20, batch=batch_g2)
        x2_flat = x_g2.transpose(1, 2).reshape(-1, x_g2.size(1))  # (B*N//16, 8D)
        x3 = self.edgeconv3(x2_flat, edge_index_2)
        x3 = x3.view(batch_size, N // 16, -1).transpose(1, 2)  # (B, 8D, N//16)
        x3 = torch.cat([x_g2, x3], dim=1)  # (B, 16D, N//16)

        # Global max pooling
        x3_flat = x3.transpose(1, 2).reshape(-1, x3.size(1))  # (B*N//16, 16D)
        x_g = global_max_pool(x3_flat, batch_g2)  # (B, 16D)
        x_g = x_g.unsqueeze(-1)  # (B, 16D, 1)

        # Projection layers
        x = self.relu(self.ps_adj(x_g))  # (B, 4D, 1)
        x = self.relu(self.ps(x))  # (B, 2D, 1)
        x = self.relu(self.ps_refuse(x))  # (B, D, 1)

        # Additional transformer layers
        x0_d = self.sa0_d(x, x)  # (B, D, 1)
        x1_d = self.sa1_d(x0_d, x0_d)  # (B, D, 1)
        x2_d = self.sa2_d(x1_d, x1_d)  # (B, D, 1)

        # Reshape for coarse point cloud generation (e.g., M = 128 points)
        M = 128  # Number of points in the coarse point cloud, adjustable
        x2_d = x2_d.expand(-1, -1, M)  # (B, D, M)

        # Generate coarse point cloud
        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))  # (B, 3, M)

        return x_g, fine
    
    
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
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 2)
        x_g0 = gather_points(x0, idx_0)
        points = gather_points(points, idx_0)
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        # SFA
        x1 = self.sa1_1(x1,x1).contiguous()
        # GDP
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous()
        # GDP
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
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




from models.SVDformer import SDG, local_encoder

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

        self.encoder = PCT_encoder()

        self.localencoder = local_encoder(cfg)
        self.refine = SDG(ratio=cfg.NETWORK.step1,hidden_dim=768,dataset=cfg.DATASET.TEST_DATASET)
        self.refine1 = SDG(ratio=cfg.NETWORK.step2,hidden_dim=512,dataset=cfg.DATASET.TEST_DATASET)

        # self.refine = PCT_refine(ratio=step1)
        # self.refine1 = PCT_refine(ratio=step2)



    
    def forward(self, x, gt=None, is_training=True):
        feat_g, coarse = self.encoder(x)
        #print(feat_g.shape)
        local_feat = self.localencoder(x)
        new_x = torch.cat([x,coarse],dim=2)
        new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))

        fine = self.refine(local_feat, new_x, feat_g,x)
        fine1 = self.refine1(local_feat, fine, feat_g,x)

        # fine, feat_fine = self.refine(None, new_x, feat_g)
        # fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)

        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()
        if is_training:
            # Organize predictions into a list [Pc, P1, P2, P3] format expected by get_loss1
            pcds_pred = [coarse, fine, fine1, fine1]  # Using fine1 twice as P3 (finest resolution)
            # Call get_loss1 with appropriate parameters
            loss_all, losses = get_loss1(pcds_pred, x.transpose(1, 2).contiguous(), gt, sqrt=True)
            
            # Unpack individual losses for logging or other purposes
            cdc, cd1, cd2, partial_matching = losses
            
            # Add penalty loss to the total loss
            #penalty_loss = self.penalty_loss(x, coarse, fine1) * 0.2
            total_train_loss = loss_all #+ penalty_loss
            
            # For compatibility with the return statement
            loss2 = cd2
            
            return fine, loss2, total_train_loss
            

        else:
            if gt is not None:

                pcds_pred = [coarse, fine, fine1, fine1]  # Using fine1 twice as P3 (finest resolution)
            
            # Call get_loss1 with appropriate parameters
                loss_all, losses = get_loss1(pcds_pred, x.transpose(1, 2).contiguous(), gt, sqrt=True)
                cd_p, cd_t = calc_cd(fine1, gt)
                
                # Unpack individual losses for logging or other purposes
                cdc, cd1, cd2, partial_matching = losses

                return {'out1': coarse, 'out2': fine1, 'cd_t_coarse': cdc, 'cd_p_coarse': cdc, 'cd_p': cd2, 'cd_t': cd_t}
            else:
                return {'out1': coarse, 'out2': fine1}


class Model_default(nn.Module):
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

        self.encoder = PCT_encoder()

        

        self.refine = PCT_refine(ratio=step1)
        self.refine1 = PCT_refine(ratio=step2)


    def penalty_loss(self, x, coarse, fine1):
        # Define half the effective ranges for x and y
        half_range_x = (0.045) / 2
        half_range_y = (0.045   ) / 2
        avg_depth = 0.038  # average depth in z
        
        # x is (B, 3, N), so transpose to (B, N, 3)
        x_src = x.transpose(1, 2)
        penalty_loss = 0.0
        B = x_src.size(0)
        for i in range(B):
            src = x_src[i]  # (N, 3)
            # Compute centroid for x and y from src
            centroid_x = (src[:, 0].min() + src[:, 0].max()) / 2
            centroid_y = (src[:, 1].min() + src[:, 1].max()) / 2
            # Minimum z from the source
            z_min = src[:, 2].min()
            
            # For coarse predictions:
            pred_coarse = coarse[i]  # (n_coarse, 3)
            
            # Calculate distance from each predicted point to the allowed thresholds:
            # For x and y: compute how far the absolute difference exceeds the half-range.
            dist_x = torch.clamp(torch.abs(pred_coarse[:, 0] - centroid_x) - half_range_x, min=0)
            dist_y = torch.clamp(torch.abs(pred_coarse[:, 1] - centroid_y) - half_range_y, min=0)
            # For z: penalize if below z_min or above z_min + avg_depth.
            dist_z_lower = torch.clamp(z_min - pred_coarse[:, 2], min=0)
            dist_z_upper = torch.clamp(pred_coarse[:, 2] - (z_min + avg_depth), min=0)
            
            # Combine the distances (you can use a sum, mean, or weighted combination)
            loss_coarse = (dist_x + dist_y + dist_z_lower + dist_z_upper).mean()
            
            # For fine predictions:
            pred_fine1 = fine1[i]    # (n_fine, 3)
            dist_x_f = torch.clamp(torch.abs(pred_fine1[:, 0] - centroid_x) - half_range_x, min=0)
            dist_y_f = torch.clamp(torch.abs(pred_fine1[:, 1] - centroid_y) - half_range_y, min=0)
            dist_z_lower_f = torch.clamp(z_min - pred_fine1[:, 2], min=0)
            dist_z_upper_f = torch.clamp(pred_fine1[:, 2] - (z_min + avg_depth), min=0)
            
            loss_fine1 = (dist_x_f + dist_y_f + dist_z_lower_f + dist_z_upper_f).mean()
            
            # Accumulate the total penalty loss.
            penalty_loss += loss_coarse + loss_fine1
        
        penalty_loss = penalty_loss / B
        return penalty_loss
    
    def forward(self, x, gt=None, is_training=True):
        feat_g, coarse = self.encoder(x)
        #print(feat_g.shape)
        new_x = torch.cat([x,coarse],dim=2)
        new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))

        fine, feat_fine = self.refine(None, new_x, feat_g)
        fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)

        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()
        # print(fine1.shape)
        if is_training:
            # Organize predictions into a list [Pc, P1, P2, P3] format expected by get_loss1
            pcds_pred = [coarse, fine, fine1, fine1]  # Using fine1 twice as P3 (finest resolution)
            # Call get_loss1 with appropriate parameters
            loss_all, losses = get_loss1(pcds_pred, x.transpose(1, 2).contiguous(), gt, sqrt=True)
            
            # Unpack individual losses for logging or other purposes
            cdc, cd1, cd2, partial_matching = losses
            
            # Add penalty loss to the total loss
            #penalty_loss = self.penalty_loss(x, coarse, fine1) * 0.2
            total_train_loss = loss_all #+ penalty_loss
            
            # For compatibility with the return statement
            loss2 = cd2
            
            return fine, loss2, total_train_loss
            

        else:
            if gt is not None:

                pcds_pred = [coarse, fine, fine1, fine1]  # Using fine1 twice as P3 (finest resolution)
            
            # Call get_loss1 with appropriate parameters
                loss_all, losses = get_loss1(pcds_pred, x.transpose(1, 2).contiguous(), gt, sqrt=True)
                cd_p, cd_t = calc_cd(fine1, gt)
                
                # Unpack individual losses for logging or other purposes
                cdc, cd1, cd2, partial_matching = losses

                return {'out1': coarse, 'out2': fine1, 'cd_t_coarse': cdc, 'cd_p_coarse': cdc, 'cd_p': cd2, 'cd_t': cd_t}
            else:
                return {'out1': coarse, 'out2': fine1}




# changes to encoder 2, 4 , 8. and hypercd