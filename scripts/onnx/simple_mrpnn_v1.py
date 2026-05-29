import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import torch.nn.functional as F
from thop import profile, clever_format



class LinearReLU(nn.Module):
    """Linear layer followed by ReLU activation"""

    def __init__(self, in_dim, out_dim, bias=True):
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return F.relu(self.linear(x))

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Implements the SE macro from CUDA code: computes attention weights for three feature streams
    """

    def __init__(self, se_dim=8, output_dim=3):
        super(SEBlock, self).__init__()
        # SE weight generation layers
        self.se_layer1 = nn.Linear(8, se_dim, bias=False)  # SEW0: 8 -> 8
        self.se_layer2 = nn.Linear(
            se_dim, output_dim, bias=False)  # SEW1: 8 -> 3

    def forward(self, x_main, x_sub, x_hg, pool_features=None):
        """
        Args:
            x_main: Main features [batch, size]
            x_sub: Sub features [batch, size]
            x_hg: HG features [batch, size]
            pool_features: Additional pooled features (g, gamma) [batch, 2] (optional)
        Returns:
            se_weights: Attention weights [batch, 3]
        """
        # AVGMAX operation: compute average and max for each feature stream
        avg_main = x_main.mean(dim=-1, keepdim=True)  # [batch, 1]
        max_main = x_main.max(dim=-1, keepdim=True)[0]  # [batch, 1]
        avg_sub = x_sub.mean(dim=-1, keepdim=True)
        max_sub = x_sub.max(dim=-1, keepdim=True)[0]
        avg_hg = x_hg.mean(dim=-1, keepdim=True)
        max_hg = x_hg.max(dim=-1, keepdim=True)[0]

        # Concatenate: [avg_main, max_main, avg_sub, max_sub, avg_hg, max_hg] -> 6 dims
        temp_pool = torch.cat(
            [avg_main, max_main, avg_sub, max_sub, avg_hg, max_hg], dim=-1)

        # Add g and gamma if provided (from pool_features)
        pool_input = torch.cat([temp_pool, pool_features[:, -2:]], dim=-1)

        # SE weight generation
        se_weight = F.relu(self.se_layer1(pool_input))  # [batch, se_dim]
        # [batch, 3] - attention weights
        se_weight = torch.sigmoid(self.se_layer2(se_weight))

        return se_weight


class SEBlockWithoutResidual(nn.Module):
    """
    SE Block with residual connection (SERES)
    Processes three feature streams with attention weighting and residual connection
    """

    def __init__(self, from_dim, to_dim, size_x, se_dim=8):
        super(SEBlockWithoutResidual, self).__init__()
        self.from_dim = from_dim
        self.to_dim = to_dim
        self.size_x = size_x

        # SE block for attention
        self.se_block = SEBlock(se_dim=se_dim, output_dim=3)

        # Three separate branches for three feature types
        self.branch_main = LinearReLU(size_x, size_x)  # WW
        self.branch_sub = LinearReLU(size_x, size_x)   # TRW
        self.branch_hg = LinearReLU(size_x, size_x)    # HGW

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2,
                pool_features=None):
        """
        Args:
            X_Val: Main features [batch, size_x] (or [batch, from_dim] if from_dim > 0)
            X_Val_Sub: Sub features [batch, size_x] (or [batch, from_dim] if from_dim > 0)
            X_Val_Hg: HG features [batch, size_x] (or [batch, from_dim] if from_dim > 0)
            pool_features: Pooled features [batch, 2] for g and gamma (optional)
        """
        X_Val_slice = X_Val[:, self.from_dim:self.from_dim+self.size_x]
        X_Val_Sub_slice = X_Val_Sub[:, self.from_dim:self.from_dim+self.size_x]
        X_Val_Hg_slice = X_Val_Hg[:, self.from_dim:self.from_dim+self.size_x]

        # SE attention
        se_weights = self.se_block(
            X_Val_slice, X_Val_Sub_slice, X_Val_Hg_slice, pool_features)

        # Apply attention weights and transform
        X_ValA_left = X_ValA[:, :self.to_dim]
        X_ValB_left = X_ValB[:, :self.to_dim]
        X_ValC_left = X_ValC[:, :self.to_dim]
        X_ValA2_left = X_ValA2[:, :self.to_dim]
        X_ValB2_left = X_ValB2[:, :self.to_dim]
        X_ValC2_left = X_ValC2[:, :self.to_dim]

        X_ValA_right = X_ValA[:, self.to_dim + self.size_x:]
        X_ValB_right = X_ValB[:, self.to_dim + self.size_x:]
        X_ValC_right = X_ValC[:, self.to_dim + self.size_x:]
        X_ValA2_right = X_ValA2[:, self.to_dim + self.size_x:]
        X_ValB2_right = X_ValB2[:, self.to_dim + self.size_x:]
        X_ValC2_right = X_ValC2[:, self.to_dim + self.size_x:]

        X_ValA_mid = X_ValA[:, self.to_dim:self.to_dim + self.size_x]
        X_ValB_mid = X_ValB[:, self.to_dim:self.to_dim + self.size_x]
        X_ValC_mid = X_ValC[:, self.to_dim:self.to_dim + self.size_x]
        X_ValA2_mid = X_ValA2[:, self.to_dim:self.to_dim + self.size_x]
        X_ValB2_mid = X_ValB2[:, self.to_dim:self.to_dim + self.size_x]
        X_ValC2_mid = X_ValC2[:, self.to_dim:self.to_dim + self.size_x]        

        X_ValA_mid = X_Val_slice * se_weights[:, 0:1]
        X_ValB_mid = X_Val_Sub_slice * se_weights[:, 1:2]
        X_ValC_mid = X_Val_Hg_slice * se_weights[:, 2:3]

        # Forward through branches
        X_ValA2_mid = self.branch_main(X_ValA_mid)
        X_ValB2_mid = self.branch_sub(X_ValB_mid)
        X_ValC2_mid = self.branch_hg(X_ValC_mid)

        X_ValA_new = torch.cat([X_ValA_left, X_ValA_mid, X_ValA_right], dim=1)
        X_ValB_new = torch.cat([X_ValB_left, X_ValB_mid, X_ValB_right], dim=1)
        X_ValC_new = torch.cat([X_ValC_left, X_ValC_mid, X_ValC_right], dim=1)

        X_ValA2_new = torch.cat([X_ValA2_left, X_ValA2_mid, X_ValA2_right], dim=1)
        X_ValB2_new = torch.cat([X_ValB2_left, X_ValB2_mid, X_ValB2_right], dim=1)
        X_ValC2_new = torch.cat([X_ValC2_left, X_ValC2_mid, X_ValC2_right], dim=1)


        return X_ValA_new, X_ValB_new, X_ValC_new, X_ValA2_new, X_ValB2_new, X_ValC2_new


class SEBlockWithResidual(nn.Module):
    """
    SE Block with residual connection (SERES)
    Processes three feature streams with attention weighting and residual connection
    """

    def __init__(self, last, from_dim, to_dim, size_x, se_dim=8):
        super(SEBlockWithResidual, self).__init__()
        self.last = last
        self.from_dim = from_dim
        self.to_dim = to_dim
        self.size_x = size_x

        # SE block for attention
        self.se_block = SEBlock(se_dim=se_dim, output_dim=3)

        # Three separate branches for three feature types
        self.branch_main = LinearReLU(size_x, size_x)  # WW
        self.branch_sub = LinearReLU(size_x, size_x)   # TRW
        self.branch_hg = LinearReLU(size_x, size_x)    # HGW

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2,
                pool_features=None):
        """
        Args:
            X_Val: Main features [batch, size_x] (or [batch, from_dim] if from_dim > 0)
            X_Val_Sub: Sub features [batch, size_x] (or [batch, from_dim] if from_dim > 0)
            X_Val_Hg: HG features [batch, size_x] (or [batch, from_dim] if from_dim > 0)
            pool_features: Pooled features [batch, 2] for g and gamma (optional)
        """
        '''
        # Extract features from specified range if needed
        if self.from_dim > 0 and x_main.shape[1] > self.size_x:
            # Extract last size_x elements
            #x_main_slice = x_main[:, -self.size_x:]
            x_main_slice = x_main[:, self.from_dim:self.from_dim+self.size_x]
            #x_sub_slice = x_sub[:, -self.size_x:]
            x_sub_slice = x_sub[:, self.from_dim:self.from_dim+self.size_x]
            #x_hg_slice = x_hg[:, -self.size_x:]
            x_hg_slice = x_hg[:, self.from_dim:self.from_dim+self.size_x]
        else:
            x_main_slice = x_main
            x_sub_slice = x_sub
            x_hg_slice = x_hg
        '''
        X_Val_slice = X_Val[:, self.from_dim:self.from_dim+self.size_x]
        X_Val_Sub_slice = X_Val_Sub[:, self.from_dim:self.from_dim+self.size_x]
        X_Val_Hg_slice = X_Val_Hg[:, self.from_dim:self.from_dim+self.size_x]

        # SE attention
        se_weights = self.se_block(
            X_Val_slice, X_Val_Sub_slice, X_Val_Hg_slice, pool_features)

        # Apply attention weights and transform


        X_ValA_left = X_ValA[:, :self.to_dim]
        X_ValB_left = X_ValB[:, :self.to_dim]
        X_ValC_left = X_ValC[:, :self.to_dim]
        X_ValA2_left = X_ValA2[:, :self.to_dim]
        X_ValB2_left = X_ValB2[:, :self.to_dim]
        X_ValC2_left = X_ValC2[:, :self.to_dim]

        X_ValA_right = X_ValA[:, self.to_dim + self.size_x:]
        X_ValB_right = X_ValB[:, self.to_dim + self.size_x:]
        X_ValC_right = X_ValC[:, self.to_dim + self.size_x:]
        X_ValA2_right = X_ValA2[:, self.to_dim + self.size_x:]
        X_ValB2_right = X_ValB2[:, self.to_dim + self.size_x:]
        X_ValC2_right = X_ValC2[:, self.to_dim + self.size_x:]

        X_ValA_mid = X_ValA[:, self.to_dim:self.to_dim + self.size_x]
        X_ValB_mid = X_ValB[:, self.to_dim:self.to_dim + self.size_x]
        X_ValC_mid = X_ValC[:, self.to_dim:self.to_dim + self.size_x]
        X_ValA2_mid = X_ValA2[:, self.to_dim:self.to_dim + self.size_x]
        X_ValB2_mid = X_ValB2[:, self.to_dim:self.to_dim + self.size_x]
        X_ValC2_mid = X_ValC2[:, self.to_dim:self.to_dim + self.size_x]    

        X_ValA_mid = X_Val_slice * se_weights[:, 0:1]
        X_ValB_mid = X_Val_Sub_slice * se_weights[:, 1:2]
        X_ValC_mid = X_Val_Hg_slice * se_weights[:, 2:3]

        X_ValA_mid_new = X_ValA_mid + X_ValA2[:, self.last:self.last+self.size_x]  
        X_ValB_mid_new = X_ValB_mid + X_ValB2[:, self.last:self.last+self.size_x]
        X_ValC_mid_new = X_ValC_mid + X_ValC2[:, self.last:self.last+self.size_x]

        # Forward through branches
        X_ValA2_mid = self.branch_main(X_ValA_mid_new)
        X_ValB2_mid = self.branch_sub(X_ValB_mid_new)
        X_ValC2_mid = self.branch_hg(X_ValC_mid_new)

        X_ValA_new = torch.cat([X_ValA_left, X_ValA_mid_new, X_ValA_right], dim=1)
        X_ValB_new = torch.cat([X_ValB_left, X_ValB_mid_new, X_ValB_right], dim=1)
        X_ValC_new = torch.cat([X_ValC_left, X_ValC_mid_new, X_ValC_right], dim=1)

        X_ValA2_new = torch.cat([X_ValA2_left, X_ValA2_mid, X_ValA2_right], dim=1)
        X_ValB2_new = torch.cat([X_ValB2_left, X_ValB2_mid, X_ValB2_right], dim=1)
        X_ValC2_new = torch.cat([X_ValC2_left, X_ValC2_mid, X_ValC2_right], dim=1)

        return X_ValA_new, X_ValB_new, X_ValC_new, X_ValA2_new, X_ValB2_new, X_ValC2_new


def ComputeAvgPoolSlice(X_Val, X_Val_Sub, X_Val_Hg, from_dim, sizex, PID, AvgPool, POOL=36):
    X_Val_slice = X_Val[:, from_dim:from_dim+sizex]
    X_Val_Sub_slice = X_Val_Sub[:, from_dim:from_dim+sizex]
    X_Val_Hg_slice = X_Val_Hg[:, from_dim:from_dim+sizex]
    #AvgPool[:, PID] = X_Val_slice.mean(dim=-1, keepdim=True)  # [batch, 1]
    AvgPool[:, PID] = X_Val_slice.mean(dim=-1)  # [batch,]
    # [batch, 1]
    #AvgPool[:, PID+POOL] = X_Val_slice.max(dim=-1, keepdim=True)[0]
    AvgPool[:, PID+POOL] = X_Val_slice.max(dim=-1)[0]
    #AvgPool[:, PID+1] = X_Val_Sub_slice.mean(dim=-1, keepdim=True)
    AvgPool[:, PID+1] = X_Val_Sub_slice.mean(dim=-1)
    #AvgPool[:, PID+POOL+1] = X_Val_Sub_slice.max(dim=-1, keepdim=True)[0]
    AvgPool[:, PID+POOL+1] = X_Val_Sub_slice.max(dim=-1)[0]
    #AvgPool[:, PID+2] = X_Val_Hg_slice.mean(dim=-1, keepdim=True)
    AvgPool[:, PID+2] = X_Val_Hg_slice.mean(dim=-1)
    #AvgPool[:, PID+POOL+2] = X_Val_Hg_slice.max(dim=-1, keepdim=True)[0]
    AvgPool[:, PID+POOL+2] = X_Val_Hg_slice.max(dim=-1)[0]
    return AvgPool


def ComputeAvgPool(X_Val, X_Val_Sub, X_Val_Hg, AvgPool):
    AvgPool = ComputeAvgPoolSlice(X_Val, X_Val_Sub, X_Val_Hg, 0, 8, 0, AvgPool)
    AvgPool = ComputeAvgPoolSlice(X_Val, X_Val_Sub, X_Val_Hg, 8, 8, 3, AvgPool)
    AvgPool = ComputeAvgPoolSlice(
        X_Val, X_Val_Sub, X_Val_Hg, 16, 16, 6, AvgPool)
    AvgPool = ComputeAvgPoolSlice(
        X_Val, X_Val_Sub, X_Val_Hg, 32, 16, 9, AvgPool)
    AvgPool = ComputeAvgPoolSlice(
        X_Val, X_Val_Sub, X_Val_Hg, 48, 16, 12, AvgPool)
    AvgPool = ComputeAvgPoolSlice(
        X_Val, X_Val_Sub, X_Val_Hg, 64, 32, 15, AvgPool)
    AvgPool = ComputeAvgPoolSlice(
        X_Val, X_Val_Sub, X_Val_Hg, 96, 32, 18, AvgPool)
    AvgPool = ComputeAvgPoolSlice(
        X_Val, X_Val_Sub, X_Val_Hg, 128, 32, 21, AvgPool)

    AvgPool = ComputeAvgPoolSlice(
        X_Val, X_Val_Sub, X_Val_Hg, 160, 8, 24, AvgPool)
    AvgPool = ComputeAvgPoolSlice(
        X_Val, X_Val_Sub, X_Val_Hg, 168, 8, 27, AvgPool)
    AvgPool = ComputeAvgPoolSlice(
        X_Val, X_Val_Sub, X_Val_Hg, 176, 8, 30, AvgPool)
    AvgPool = ComputeAvgPoolSlice(
        X_Val, X_Val_Sub, X_Val_Hg, 184, 8, 33, AvgPool)

    return AvgPool


class MainBranch(nn.Module):
    """
    Main feature extraction branch
    Processes 192-dim input through 8 SE/SE-Res blocks
    Output: 96-dim (32*3)
    """

    def __init__(self):
        super(MainBranch, self).__init__()

        # SERES blocks with progressive dimension expansion
        self.seres0 = SEBlockWithoutResidual(0, 0, 8, se_dim=8)      # 0 -> 0
        self.seres1 = SEBlockWithResidual(0, 8, 8, 8, se_dim=8)      # 8 -> 8
        self.seres2 = SEBlockWithResidual(0, 16, 16, 16, se_dim=8)   # 16 -> 16
        self.seres3 = SEBlockWithResidual(
            16, 32, 32, 16, se_dim=8)   # 32 -> 32
        self.seres4 = SEBlockWithResidual(
            32, 48, 48, 16, se_dim=8)   # 48 -> 48
        self.seres5 = SEBlockWithResidual(
            32, 64, 64, 32, se_dim=8)   # 64 -> 64
        self.seres6 = SEBlockWithResidual(
            64, 96, 96, 32, se_dim=8)   # 96 -> 96
        self.seres7 = SEBlockWithResidual(
            96, 128, 128, 32, se_dim=8)  # 128 -> 128

        # Final projection layers: 32 -> 32 for each branch
        self.final_main = LinearReLU(32, 32)
        self.final_sub = LinearReLU(32, 32)
        self.final_hg = LinearReLU(32, 32)

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg, X_ValA_init, X_ValB_init, X_ValC_init, X_ValA2_init, X_ValB2_init, X_ValC2_init, g=None, gamma=None):
        """
        Args:
            X_Val: Main features [batch, 192]
            X_Val_Sub: Sub features [batch, 192]
            X_Val_Hg: HG features [batch, 192]
            g: Scattering parameter [batch] or scalar (optional)
            gamma: Angle parameter [batch] or scalar (optional)
        """
        #import pdb;pdb.set_trace()
        batch_size = X_Val.shape[0]
        device = X_Val.device

        # Prepare pool features (g and gamma)
        if g is not None and gamma is not None:
            if isinstance(g, (int, float)):
                g = torch.tensor(g, device=device,
                                 dtype=X_Val.dtype).expand(batch_size)
            if isinstance(gamma, (int, float)):
                gamma = torch.tensor(gamma, device=device,
                                     dtype=X_Val.dtype).expand(batch_size)
            pool_features = torch.stack([g, gamma], dim=1)  # [batch, 2]
        else:
            pool_features = None

        # SE init: 192 -> 8
        # x_a, x_b, x_c = self.se_init(x_main, x_sub, x_hg, pool_features=pool_features)

        # SERES blocks with residual connections
        # SE0: 8 -> 8 (with ADD3)
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres0(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA_init, X_ValB_init, X_ValC_init, X_ValA2_init, X_ValB2_init, X_ValC2_init, pool_features=pool_features)

        X_ValA2_new = X_ValA2[:, 0:8] + X_Val[:, 0:8]
        X_ValA2 = torch.cat([X_ValA2_new, X_ValA2[:, 8:]], dim=1)
        X_ValB2_new = X_ValB2[:, 0:8] + X_Val_Sub[:, 0:8]
        X_ValB2 = torch.cat([X_ValB2_new, X_ValB2[:, 8:]], dim=1)
        X_ValC2_new = X_ValC2[:, 0:8] + X_Val_Hg[:, 0:8]
        X_ValC2 = torch.cat([X_ValC2_new, X_ValC2[:, 8:]], dim=1)

        # SERES blocks with residual connections
        # SERES1: 8 -> 8 (with ADD3)
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres1(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        
        X_ValA2 = torch.cat([X_ValA2[:, 8:8+8], X_ValA[:, 8:8+8], X_ValA2[:, 8+8:]], dim=1)
        X_ValB2 = torch.cat([X_ValB2[:, 8:8+8], X_ValB[:, 8:8+8], X_ValB2[:, 8+8:]], dim=1)
        X_ValC2 = torch.cat([X_ValC2[:, 8:8+8], X_ValC[:, 8:8+8], X_ValC2[:, 8+8:]], dim=1)
        # Expand dimensions progressively
        # SERES2: expand to 16
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres2(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)

        X_ValA2_new = X_ValA2[:, 16:16+16] + X_Val[:, 16:16+16]
        X_ValA2 = torch.cat([X_ValA2[:, :16], X_ValA2_new, X_ValA2[:, 16+16:]], dim=1)
        X_ValB2_new = X_ValB2[:, 16:16+16] + X_Val_Sub[:, 16:16+16]
        X_ValB2 = torch.cat([X_ValB2[:, :16], X_ValB2_new, X_ValB2[:, 16+16:]], dim=1)
        X_ValC2_new = X_ValC2[:, 16:16+16] + X_Val_Hg[:, 16:16+16]
        X_ValC2 = torch.cat([X_ValC2[:, :16], X_ValC2_new, X_ValC2[:, 16+16:]], dim=1)

        # SERES3: 16 -> 32
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres3(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        
        X_ValA2_new = X_ValA2[:, 32:32+16] + X_Val[:, 32:32+16]
        X_ValA2 = torch.cat([X_ValA2[:, :32], X_ValA2_new, X_ValA2[:, 32+16:]], dim=1)
        X_ValB2_new = X_ValB2[:, 32:32+16] + X_Val_Sub[:, 32:32+16]
        X_ValB2 = torch.cat([X_ValB2[:, :32], X_ValB2_new, X_ValB2[:, 32+16:]], dim=1)
        X_ValC2_new = X_ValC2[:, 32:32+16] + X_Val_Hg[:, 32:32+16]
        X_ValC2 = torch.cat([X_ValC2[:, :32], X_ValC2_new, X_ValC2[:, 32+16:]], dim=1)

        # SERES4: 32 -> 48

        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres4(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        
        X_ValA2 = torch.cat([X_ValA2[:, :32], X_ValA2[:, 48:48+16], X_ValA[:, 48:48+16], X_ValA2[:, 48+16:]], dim=1)
        X_ValB2 = torch.cat([X_ValB2[:, :32], X_ValB2[:, 48:48+16], X_ValB[:, 48:48+16], X_ValB2[:, 48+16:]], dim=1)
        X_ValC2 = torch.cat([X_ValC2[:, :32], X_ValC2[:, 48:48+16], X_ValC[:, 48:48+16], X_ValC2[:, 48+16:]], dim=1)

        # SERES5: 48 -> 64

        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres5(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        
        X_ValA2_new = X_ValA2[:, 64:64+32] + X_Val[:, 64:64+32]
        X_ValA2 = torch.cat([X_ValA2[:, :64], X_ValA2_new, X_ValA2[:, 64+32:]], dim=1)
        X_ValB2_new = X_ValB2[:, 64:64+32] + X_Val_Sub[:, 64:64+32]
        X_ValB2 = torch.cat([X_ValB2[:, :64], X_ValB2_new, X_ValB2[:, 64+32:]], dim=1)
        X_ValC2_new = X_ValC2[:, 64:64+32] + X_Val_Hg[:, 64:64+32]
        X_ValC2 = torch.cat([X_ValC2[:, :64], X_ValC2_new, X_ValC2[:, 64+32:]], dim=1)

        # SERES6: 64 -> 96

        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres6(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        
        X_ValA2_new = X_ValA2[:, 96:96+32] + X_Val[:, 96:96+32]
        X_ValA2 = torch.cat([X_ValA2[:, :96], X_ValA2_new, X_ValA2[:, 96+32:]], dim=1)
        X_ValB2_new = X_ValB2[:, 96:96+32] + X_Val_Sub[:, 96:96+32]
        X_ValB2 = torch.cat([X_ValB2[:, :96], X_ValB2_new, X_ValB2[:, 96+32:]], dim=1)
        X_ValC2_new = X_ValC2[:, 96:96+32] + X_Val_Hg[:, 96:96+32]
        X_ValC2 = torch.cat([X_ValC2[:, :96], X_ValC2_new, X_ValC2[:, 96+32:]], dim=1)

        # SERES7: 96 -> 128

        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres7(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        
        X_ValA2_new = X_ValA2[:, 128:128+32] + X_Val[:, 128:128+32]
        X_ValA2 = torch.cat([X_ValA2[:, :128], X_ValA2_new, X_ValA2[:, 128+32:]], dim=1)
        X_ValB2_new = X_ValB2[:, 128:128+32] + X_Val_Sub[:, 128:128+32]
        X_ValB2 = torch.cat([X_ValB2[:, :128], X_ValB2_new, X_ValB2[:, 128+32:]], dim=1)
        X_ValC2_new = X_ValC2[:, 128:128+32] + X_Val_Hg[:, 128:128+32]
        X_ValC2 = torch.cat([X_ValC2[:, :128], X_ValC2_new, X_ValC2[:, 128+32:]], dim=1)

        # Final projections: 128 -> 32 for each branch
        comb_main = self.final_main(X_ValA2[:, 128:])  # [batch, 32]
        comb_sub = self.final_sub(X_ValB2[:, 128:])    # [batch, 32]
        comb_hg = self.final_hg(X_ValC2[:, 128:])      # [batch, 32]

        # Concatenate: 32 + 32 + 32 = 96
        comb = torch.cat([comb_main, comb_sub, comb_hg], dim=1)  # [batch, 96]

        return comb, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2


class DirectionBranch(nn.Module):
    """
    Direction Information branch
    Processes 160-dim input through 4 SE/SE-Res blocks
    Output: 24-dim (8*3)
    """

    def __init__(self):
        super(DirectionBranch, self).__init__()

        # Initial SE: 160 -> 8
        self.seres0 = SEBlockWithoutResidual(160, 8, 8, se_dim=8)

        # SERES blocks
        self.seres1 = SEBlockWithResidual(0, 168, 8, 8, se_dim=8)    # 8 -> 8
        self.seres2 = SEBlockWithResidual(
            8, 176, 16, 8, se_dim=8)    # 16 -> 16
        self.seres3 = SEBlockWithResidual(
            16, 184, 24, 8, se_dim=8)    # 24 -> 24

        # Final projections: 24 -> 8 for each branch
        self.final_main = LinearReLU(8, 8)
        self.final_sub = LinearReLU(8, 8)
        self.final_hg = LinearReLU(8, 8)

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, g=None, gamma=None):
        """
        Args:
            X_Val: DI main features [batch, 160]
            X_Val_Sub: DI sub features [batch, 160]
            X_Val_Hg: DI HG features [batch, 160]
            g: Scattering parameter [batch] or scalar (optional)
            gamma: Angle parameter [batch] or scalar (optional)
        """
        batch_size = X_Val.shape[0]
        device = X_Val.device

        # Prepare pool features
        if g is not None and gamma is not None:
            if isinstance(g, (int, float)):
                g = torch.tensor(g, device=device,
                                 dtype=X_Val.dtype).expand(batch_size)
            if isinstance(gamma, (int, float)):
                gamma = torch.tensor(gamma, device=device,
                                     dtype=X_Val.dtype).expand(batch_size)
            pool_features = torch.stack([g, gamma], dim=1)
        else:
            pool_features = None

        # SE init: 160 -> 8
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres0(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)

        X_ValA2_new = X_ValA2[:, 0:8] + X_Val[:, 160:160+8]
        X_ValA2 = torch.cat([X_ValA2_new, X_ValA2[:, 8:]], dim=1)
        X_ValB2_new = X_ValB2[:, 0:8] + X_Val_Sub[:, 160:160+8]
        X_ValB2 = torch.cat([X_ValB2_new, X_ValB2[:, 8:]], dim=1)
        X_ValC2_new = X_ValC2[:, 0:8] + X_Val_Hg[:, 160:160+8]
        X_ValC2 = torch.cat([X_ValC2_new, X_ValC2[:, 8:]], dim=1)

        # SERES blocks

        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres1(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)

        X_ValA2_new = X_ValA2[:, 8:8+8] + X_Val[:, 168:168+8]
        X_ValA2 = torch.cat([X_ValA2[:, :8], X_ValA2_new, X_ValA2[:, 8+8:]], dim=1)
        X_ValB2_new = X_ValB2[:, 8:8+8] + X_Val_Sub[:, 168:168+8]
        X_ValB2 = torch.cat([X_ValB2[:, :8], X_ValB2_new, X_ValB2[:, 8+8:]], dim=1)
        X_ValC2_new = X_ValC2[:, 8:8+8] + X_Val_Hg[:, 168:168+8]
        X_ValC2 = torch.cat([X_ValC2[:, :8], X_ValC2_new, X_ValC2[:, 8+8:]], dim=1)


        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres2(X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2,
                                                                        pool_features=pool_features)

        X_ValA2_new = X_ValA2[:, 16:16+8] + X_Val[:, 176:176+8]
        X_ValA2 = torch.cat([X_ValA2[:, :16], X_ValA2_new, X_ValA2[:, 16+8:]], dim=1)
        X_ValB2_new = X_ValB2[:, 16:16+8] + X_Val_Sub[:, 176:176+8]
        X_ValB2 = torch.cat([X_ValB2[:, :16], X_ValB2_new, X_ValB2[:, 16+8:]], dim=1)
        X_ValC2_new = X_ValC2[:, 16:16+8] + X_Val_Hg[:, 176:176+8]
        X_ValC2 = torch.cat([X_ValC2[:, :16], X_ValC2_new, X_ValC2[:, 16+8:]], dim=1)

        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres3(X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2,
                                                                        pool_features=pool_features)

        X_ValA2_new = X_ValA2[:, 24:24+8] + X_Val[:, 184:184+8]
        X_ValA2 = torch.cat([X_ValA2[:, :24], X_ValA2_new, X_ValA2[:, 24+8:]], dim=1)
        X_ValB2_new = X_ValB2[:, 24:24+8] + X_Val_Sub[:, 184:184+8]
        X_ValB2 = torch.cat([X_ValB2[:, :24], X_ValB2_new, X_ValB2[:, 24+8:]], dim=1)
        X_ValC2_new = X_ValC2[:, 24:24+8] + X_Val_Hg[:, 184:184+8]
        X_ValC2 = torch.cat([X_ValC2[:, :24], X_ValC2_new, X_ValC2[:, 24+8:]], dim=1)

        # Final projections: 24 -> 8
        comb_main = self.final_main(X_ValA2[:, 24:32])  # [batch, 8]
        comb_sub = self.final_sub(X_ValB2[:, 24:32])    # [batch, 8]
        comb_hg = self.final_hg(X_ValC2[:, 24:32])      # [batch, 8]

        # Concatenate: 8 + 8 + 8 = 24
        comb = torch.cat([comb_main, comb_sub, comb_hg], dim=1)  # [batch, 24]

        

        return comb, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2


class FinalFusion(nn.Module):
    """
    Final fusion and output layers
    Combines main and direction branches, applies final SE and outputs radiance
    """

    def __init__(self):
        super(FinalFusion, self).__init__()

        # SE final layers for attention weighting
        self.gg_layer = LinearReLU(3, 8)  # LGGSW: processes g, gamma, sc_base
        self.se_final1 = nn.Linear(75, 16, bias=False)  # LSEFin1W: 75 -> 16
        self.se_final2 = nn.Linear(16, 6, bias=False)   # LSEFin2W: 16 -> 6

        # Fusion layers: 128 -> 64 -> 32 -> 16
        '''
        self.fusion_layers = nn.ModuleList([
            LinearReLU(128, 128),  # LC0: 128 -> 128
            LinearReLU(128, 64),   # LC1: 128 -> 64
            LinearReLU(64, 32),    # LC2: 64 -> 32
            LinearReLU(32, 16),    # LX: 32 -> 16
            LinearReLU(16, 16),    # LX: 16 -> 16
            LinearReLU(16, 16),    # LX: 16 -> 16
        ])
        '''
        self.fusion_layers_1 = LinearReLU(128, 64)
        self.fusion_layers_2 = LinearReLU(64, 32)
        self.fusion_layers_3 = LinearReLU(32, 16)
        self.fusion_layers_4 = LinearReLU(16, 16)
        self.fusion_layers_5 = LinearReLU(16, 16)
        self.fusion_layers_6 = LinearReLU(16, 16)
        self.fusion_layers_7 = LinearReLU(16, 16)


        # Final output layer
        self.output_layer = nn.Linear(16, 1)  # LX12: 16 -> 1

    def forward(self, comb_features, di_features, X_ValA, X_ValB, X_ValC, avg_pool, sc_base, g, gamma):
        """
        Args:
            comb_features: Combined features from main branch [batch, 96]
            di_features: Features from direction branch [batch, 24]
            sc_base: Scatter rate base [batch, 1]
            g: Scattering parameter [batch]
            gamma: Angle parameter [batch]
        """

        batch_size = comb_features.shape[0]
        device = comb_features.device

        # Concatenate features: 96 + 24 = 120
        comb = torch.cat([comb_features, di_features], dim=1)  # [batch, 120]

        # Prepare global features for SE
        # In CUDA, AvgPool contains avg/max for all layers (75 dims total)
        # For simplicity, we use g, gamma, sc_base and pad to 75
        g_expanded = g.unsqueeze(-1) if g.dim() == 1 else g
        gamma_expanded = gamma.unsqueeze(-1) if gamma.dim() == 1 else gamma
        sc_base_expanded = sc_base.unsqueeze(
            -1) if sc_base.dim() == 1 else sc_base

        # global_feat = torch.cat([g_expanded, gamma_expanded, sc_base_expanded], dim=1)  # [batch, 3]

        X_ValA_new = comb[:, :120]
        X_ValA = torch.cat([X_ValA_new, X_ValA[:, 120:]], dim=1)

        # Create avg_pool (75 dims) - in practice this should come from pooling across all layers
        # For now, we pad the global features
        avg_pool[:, 72+0] = g_expanded.squeeze()
        avg_pool[:, 72+1] = gamma_expanded.squeeze()
        avg_pool[:, 72+2] = sc_base_expanded.squeeze()

        # SE final weights
        X_ValA_temp = F.relu(self.gg_layer(avg_pool[:, 72:]))  # [batch, 3]
        X_ValA = torch.cat([X_ValA[:,:120], X_ValA_temp, X_ValA[:,128:]], dim=1)
        
        se_weight = F.relu(self.se_final1(avg_pool))  # [batch, 16]
        se_weight = torch.sigmoid(self.se_final2(se_weight))  # [batch, 6]

        # Apply weights to different parts of comb
        # Weights applied to: [0:32], [32:64], [64:96], [96:104], [104:112], [112:120]
        X_ValA_new = torch.cat([
            X_ValA[:, 0:32] * se_weight[:, 0:1],
            X_ValA[:, 32:64] * se_weight[:, 1:2],
            X_ValA[:, 64:96] * se_weight[:, 2:3],
            X_ValA[:, 96:104] * se_weight[:, 3:4],
            X_ValA[:, 104:112] * se_weight[:, 4:5],
            X_ValA[:, 112:120] * se_weight[:, 5:6],
            X_ValA[:, 120:]
        ], dim=1)
        X_ValA = X_ValA_new

        # Fusion layers (Plan D: 128->64->32->16 then residual blocks)
        X_ValB_new = self.fusion_layers_1(X_ValA[:, :128])
        X_ValB = torch.cat([X_ValB_new, X_ValB[:, 64:]], dim=1)
        X_ValA_new = self.fusion_layers_2(X_ValB[:, :64])
        X_ValA = torch.cat([X_ValA_new, X_ValA[:, 32:]], dim=1)
        X_ValB_new = self.fusion_layers_3(X_ValA[:, :32])
        X_ValB = torch.cat([X_ValB_new, X_ValB[:, 16:]], dim=1)
        X_ValA_new = self.fusion_layers_4(X_ValB[:, :16])
        X_ValA = torch.cat([X_ValA_new, X_ValA[:, 16:]], dim=1)
        X_ValB_new = self.fusion_layers_5(X_ValA[:, :16])
        X_ValB = torch.cat([X_ValB_new, X_ValB[:, 16:]], dim=1)
        X_ValC_new = self.fusion_layers_6(X_ValB[:, :16])
        X_ValC = torch.cat([X_ValC_new, X_ValC[:, 16:]], dim=1)

        X_ValA_new = X_ValA[:, :16] + X_ValC[:, :16]
        X_ValA = torch.cat([X_ValA_new, X_ValA[:, 16:]], dim=1)
        X_ValB_new = self.fusion_layers_6(X_ValA[:, :16])
        X_ValB = torch.cat([X_ValB_new, X_ValB[:, 16:]], dim=1)
        X_ValC_new = self.fusion_layers_7(X_ValB[:, :16])
        X_ValC = torch.cat([X_ValC_new, X_ValC[:, 16:]], dim=1)

        X_ValA_new = X_ValA[:, :16] + X_ValC[:, :16]
        X_ValA = torch.cat([X_ValA_new, X_ValA[:, 16:]], dim=1)
        # Output
        output = self.output_layer(X_ValA[:, :16])

        return output


class MRPNN(nn.Module):
    """
    Complete MRPNN Network
    Multi-feature fusion radiance prediction network for volumetric scattering
    """

    def __init__(self):
        super(MRPNN, self).__init__()
        self.main_branch = MainBranch()
        self.direction_branch = DirectionBranch()
        self.final_fusion = FinalFusion()

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg,
                scatterrate=None, g=0.0, gamma=None):
        """
        Args:
            X_Val: Main features [batch, 192]
            X_Val_Sub: Sub features [batch, 192]
            X_Val_Hg: HG features [batch, 192]
            scatterrate: Scatter rate [batch, 3] or [3]
            g: Scattering parameter (float or tensor)
            gamma: Angle parameter (float or tensor, computed from inputs if not provided)

        Returns:
            output: Predicted radiance [batch, 3]
        """

        batch_size = X_Val.shape[0]
        device = X_Val.device
        X_ValA = torch.zeros([batch_size, 160], device=device, dtype=X_Val.dtype)
        X_ValB = torch.zeros([batch_size, 160], device=device, dtype=X_Val.dtype)
        X_ValC = torch.zeros([batch_size, 160], device=device, dtype=X_Val.dtype)

        X_ValA2 = torch.zeros([batch_size, 160], device=device, dtype=X_Val.dtype)
        X_ValB2 = torch.zeros([batch_size, 160], device=device, dtype=X_Val.dtype)
        X_ValC2 = torch.zeros([batch_size, 160], device=device, dtype=X_Val.dtype)

        avg_pool = torch.zeros([batch_size, 75], device=device, dtype=X_Val.dtype)

        avg_pool = ComputeAvgPool(X_Val, X_Val_Sub, X_Val_Hg, avg_pool)



        # Convert scalar inputs to tensors
        if isinstance(g, (int, float)):
            g = torch.tensor(g, device=device,
                             dtype=X_Val.dtype).expand(batch_size)
        elif g.dim() == 0:
            g = g.expand(batch_size)

        if gamma is None:
            # gamma should be computed from XMain and LXMain in practice
            gamma = torch.zeros(batch_size, device=device, dtype=X_Val.dtype)
        elif isinstance(gamma, (int, float)):
            gamma = torch.tensor(gamma, device=device,
                                 dtype=X_Val.dtype).expand(batch_size)
        elif gamma.dim() == 0:
            gamma = gamma.expand(batch_size)

        if scatterrate is None:
            scatterrate = torch.ones(3, device=device, dtype=X_Val.dtype)
        if scatterrate.dim() == 1:
            scatterrate = scatterrate.unsqueeze(0).expand(batch_size, -1)

        # Compute scatter rate base
        srp = scatterrate ** 4.0  # [batch, 3]
        sc_base = srp
        # Main branch
        comb_main, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.main_branch(X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC,
                                     X_ValA2, X_ValB2, X_ValC2, g=g, gamma=gamma)
        
        #print(f"comb_main: {comb_main[0][-16:]}")

        # Direction branch

        comb_di, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.direction_branch(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, g=g, gamma=gamma)

        # Final fusion and output
        output_0 = self.final_fusion(
            comb_main, comb_di, X_ValA, X_ValB, X_ValC, avg_pool, sc_base[:, 0], g, gamma)
        # if sc_base[:, 0] == sc_base[:, 1]:
        #     output_1 = output_0
        # else:
        #     output_1 = self.final_fusion(
        #         comb_main, comb_di, X_ValA, X_ValB, X_ValC, avg_pool, sc_base[:, 1], g, gamma)

        # if sc_base[:, 0] == sc_base[:, 2]:
        #     output_2 = output_0
        # else:
        #     output_2 = self.final_fusion(
        #         comb_main, comb_di, X_ValA, X_ValB, X_ValC, avg_pool, sc_base[:, 2], g, gamma)

        # Apply exponential and scatter rate scaling (as in CUDA)

        # output = torch.clamp(
        #     torch.exp(torch.cat([output_0, output_1, output_2], dim=1)) - 1.0, min=0.0)
        # output = output * srp  # Scale by scatter rate base

        return output_0


def create_mrpnn_model() -> MRPNN:
    """
    创建MRPNN模型
    
    Args:

        
    Returns:
        MRPNN模型实例
    """
    model = MRPNN()
    return model

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = create_mrpnn_model()
    
    # 打印模型结构
    #print("MRPNN模型结构:")
    #print(model)
    
    # 测试前向传播
    batch_size = 1
    X_Val = torch.randn(batch_size, 192)
    X_Val_Sub = torch.randn(batch_size, 192)
    X_Val_Hg = torch.randn(batch_size, 192)
    scatterrate = torch.randn(batch_size, 3)
    g = torch.randn(batch_size)
    gamma = torch.randn(batch_size)
    
    output = model(X_Val, X_Val_Sub, X_Val_Hg, scatterrate, g, gamma)

    flops, params = profile(model, inputs=(X_Val, X_Val_Sub, X_Val_Hg, scatterrate, g, gamma))
    flops, params = clever_format([flops, params], "%.3f")
    print('FLOPs:', flops)
    print('Parameters:', params)

    print(f"输入形状: X_Val{X_Val.shape}, X_Val_Sub{X_Val_Sub.shape}, X_Val_Hg{X_Val_Hg.shape}")
    print(f"输出形状: {output.shape}")
    print("模型测试完成!")
