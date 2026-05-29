import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import torch.nn.functional as F
from thop import profile, clever_format


def replace_slice(tensor, start, size, new_val):
    """Replace a slice of tensor at [start:start+size] with new_val, return the new tensor."""
    return torch.cat([tensor[:, :start], new_val, tensor[:, start + size:]], dim=1)


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
    Computes attention weights for three feature streams
    """

    def __init__(self, se_dim=8, output_dim=3):
        super(SEBlock, self).__init__()
        self.se_layer1 = nn.Linear(8, se_dim, bias=False)
        self.se_layer2 = nn.Linear(se_dim, output_dim, bias=False)

    def forward(self, x_main, x_sub, x_hg, pool_features=None):
        # AVGMAX: compute average and max for each feature stream
        temp_pool = torch.cat([
            x_main.mean(dim=-1, keepdim=True), x_main.max(dim=-1, keepdim=True)[0],
            x_sub.mean(dim=-1, keepdim=True), x_sub.max(dim=-1, keepdim=True)[0],
            x_hg.mean(dim=-1, keepdim=True), x_hg.max(dim=-1, keepdim=True)[0],
        ], dim=-1)

        pool_input = torch.cat([temp_pool, pool_features[:, -2:]], dim=-1)
        se_weight = F.relu(self.se_layer1(pool_input))
        se_weight = torch.sigmoid(self.se_layer2(se_weight))
        return se_weight


class SEBlockWithoutResidual(nn.Module):
    """SE Block without residual connection"""

    def __init__(self, from_dim, to_dim, size_x, se_dim=8):
        super(SEBlockWithoutResidual, self).__init__()
        self.from_dim = from_dim
        self.to_dim = to_dim
        self.size_x = size_x

        self.se_block = SEBlock(se_dim=se_dim, output_dim=3)
        self.branch_main = LinearReLU(size_x, size_x)
        self.branch_sub = LinearReLU(size_x, size_x)
        self.branch_hg = LinearReLU(size_x, size_x)

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2,
                pool_features=None):
        fd, td, sx = self.from_dim, self.to_dim, self.size_x

        X_Val_slice = X_Val[:, fd:fd + sx]
        X_Val_Sub_slice = X_Val_Sub[:, fd:fd + sx]
        X_Val_Hg_slice = X_Val_Hg[:, fd:fd + sx]

        se_weights = self.se_block(X_Val_slice, X_Val_Sub_slice, X_Val_Hg_slice, pool_features)

        # Apply attention weights
        X_ValA_mid = X_Val_slice * se_weights[:, 0:1]
        X_ValB_mid = X_Val_Sub_slice * se_weights[:, 1:2]
        X_ValC_mid = X_Val_Hg_slice * se_weights[:, 2:3]

        # Forward through branches
        X_ValA2_mid = self.branch_main(X_ValA_mid)
        X_ValB2_mid = self.branch_sub(X_ValB_mid)
        X_ValC2_mid = self.branch_hg(X_ValC_mid)

        X_ValA = replace_slice(X_ValA, td, sx, X_ValA_mid)
        X_ValB = replace_slice(X_ValB, td, sx, X_ValB_mid)
        X_ValC = replace_slice(X_ValC, td, sx, X_ValC_mid)
        X_ValA2 = replace_slice(X_ValA2, td, sx, X_ValA2_mid)
        X_ValB2 = replace_slice(X_ValB2, td, sx, X_ValB2_mid)
        X_ValC2 = replace_slice(X_ValC2, td, sx, X_ValC2_mid)

        return X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2


class SEBlockWithResidual(nn.Module):
    """SE Block with residual connection"""

    def __init__(self, last, from_dim, to_dim, size_x, se_dim=8):
        super(SEBlockWithResidual, self).__init__()
        self.last = last
        self.from_dim = from_dim
        self.to_dim = to_dim
        self.size_x = size_x

        self.se_block = SEBlock(se_dim=se_dim, output_dim=3)
        self.branch_main = LinearReLU(size_x, size_x)
        self.branch_sub = LinearReLU(size_x, size_x)
        self.branch_hg = LinearReLU(size_x, size_x)

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2,
                pool_features=None):
        fd, td, sx, last = self.from_dim, self.to_dim, self.size_x, self.last

        X_Val_slice = X_Val[:, fd:fd + sx]
        X_Val_Sub_slice = X_Val_Sub[:, fd:fd + sx]
        X_Val_Hg_slice = X_Val_Hg[:, fd:fd + sx]

        se_weights = self.se_block(X_Val_slice, X_Val_Sub_slice, X_Val_Hg_slice, pool_features)

        # Apply attention weights
        X_ValA_mid = X_Val_slice * se_weights[:, 0:1]
        X_ValB_mid = X_Val_Sub_slice * se_weights[:, 1:2]
        X_ValC_mid = X_Val_Hg_slice * se_weights[:, 2:3]

        # Add residual from previous layer
        X_ValA_mid_new = X_ValA_mid + X_ValA2[:, last:last + sx]
        X_ValB_mid_new = X_ValB_mid + X_ValB2[:, last:last + sx]
        X_ValC_mid_new = X_ValC_mid + X_ValC2[:, last:last + sx]

        # Forward through branches
        X_ValA2_mid = self.branch_main(X_ValA_mid_new)
        X_ValB2_mid = self.branch_sub(X_ValB_mid_new)
        X_ValC2_mid = self.branch_hg(X_ValC_mid_new)

        X_ValA = replace_slice(X_ValA, td, sx, X_ValA_mid_new)
        X_ValB = replace_slice(X_ValB, td, sx, X_ValB_mid_new)
        X_ValC = replace_slice(X_ValC, td, sx, X_ValC_mid_new)
        X_ValA2 = replace_slice(X_ValA2, td, sx, X_ValA2_mid)
        X_ValB2 = replace_slice(X_ValB2, td, sx, X_ValB2_mid)
        X_ValC2 = replace_slice(X_ValC2, td, sx, X_ValC2_mid)

        return X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2


def ComputeAvgPool(X_Val, X_Val_Sub, X_Val_Hg, AvgPool):
    """Compute average and max pooling for all feature slices."""
    # (from_dim, size_x, pid) configuration table
    POOL = 36
    slices = [
        (0, 8, 0), (8, 8, 3), (16, 16, 6), (32, 16, 9),
        (48, 16, 12), (64, 32, 15), (96, 32, 18), (128, 32, 21),
        (160, 8, 24), (168, 8, 27), (176, 8, 30), (184, 8, 33),
    ]
    for from_dim, size_x, pid in slices:
        for i, feat in enumerate([X_Val, X_Val_Sub, X_Val_Hg]):
            s = feat[:, from_dim:from_dim + size_x]
            AvgPool[:, pid + i] = s.mean(dim=-1)
            AvgPool[:, pid + POOL + i] = s.max(dim=-1)[0]
    return AvgPool


class MainBranch(nn.Module):
    """
    Main feature extraction branch
    Processes 192-dim input through 8 SE/SE-Res blocks
    Output: 96-dim (32*3)
    """

    def __init__(self):
        super(MainBranch, self).__init__()
        self.seres0 = SEBlockWithoutResidual(0, 0, 8, se_dim=8)
        self.seres1 = SEBlockWithResidual(0, 8, 8, 8, se_dim=8)
        self.seres2 = SEBlockWithResidual(0, 16, 16, 16, se_dim=8)
        self.seres3 = SEBlockWithResidual(16, 32, 32, 16, se_dim=8)
        self.seres4 = SEBlockWithResidual(32, 48, 48, 16, se_dim=8)
        self.seres5 = SEBlockWithResidual(32, 64, 64, 32, se_dim=8)
        self.seres6 = SEBlockWithResidual(64, 96, 96, 32, se_dim=8)
        self.seres7 = SEBlockWithResidual(96, 128, 128, 32, se_dim=8)

        self.final_main = LinearReLU(32, 32)
        self.final_sub = LinearReLU(32, 32)
        self.final_hg = LinearReLU(32, 32)

    def _add_input_residual(self, X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, offset, size):
        """Add input features as residual to X_Val{A,B,C}2 at given offset."""
        X_ValA2 = replace_slice(X_ValA2, offset, size, X_ValA2[:, offset:offset + size] + X_Val[:, offset:offset + size])
        X_ValB2 = replace_slice(X_ValB2, offset, size, X_ValB2[:, offset:offset + size] + X_Val_Sub[:, offset:offset + size])
        X_ValC2 = replace_slice(X_ValC2, offset, size, X_ValC2[:, offset:offset + size] + X_Val_Hg[:, offset:offset + size])
        return X_ValA2, X_ValB2, X_ValC2

    def _copy_A_to_2(self, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, offset, size):
        """Copy X_Val{A,B,C} slice into X_Val{A,B,C}2 at given offset (for seres1/seres4 pattern)."""
        X_ValA2 = replace_slice(X_ValA2, offset, size, X_ValA[:, offset:offset + size])
        X_ValB2 = replace_slice(X_ValB2, offset, size, X_ValB[:, offset:offset + size])
        X_ValC2 = replace_slice(X_ValC2, offset, size, X_ValC[:, offset:offset + size])
        return X_ValA2, X_ValB2, X_ValC2

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, g=None, gamma=None):
        batch_size = X_Val.shape[0]
        device = X_Val.device

        pool_features = self._prepare_pool_features(X_Val, g, gamma, batch_size, device)

        # SERES0: SE without residual, then add input residual at [0:8]
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres0(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        X_ValA2, X_ValB2, X_ValC2 = self._add_input_residual(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, 0, 8)

        # SERES1: then copy X_ValA[8:16] -> X_ValA2[8:16] (special pattern)
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres1(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        # Original: X_ValA2 = torch.cat([X_ValA2[:, 8:8+8], X_ValA[:, 8:8+8], X_ValA2[:, 8+8:]], dim=1)
        # This drops X_ValA2[:, :8] and replaces it with X_ValA2[:, 8:16], then inserts X_ValA[:, 8:16]
        X_ValA2 = torch.cat([X_ValA2[:, 8:16], X_ValA[:, 8:16], X_ValA2[:, 16:]], dim=1)
        X_ValB2 = torch.cat([X_ValB2[:, 8:16], X_ValB[:, 8:16], X_ValB2[:, 16:]], dim=1)
        X_ValC2 = torch.cat([X_ValC2[:, 8:16], X_ValC[:, 8:16], X_ValC2[:, 16:]], dim=1)

        # SERES2: then add input residual at [16:32]
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres2(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        X_ValA2, X_ValB2, X_ValC2 = self._add_input_residual(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, 16, 16)

        # SERES3: then add input residual at [32:48]
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres3(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        X_ValA2, X_ValB2, X_ValC2 = self._add_input_residual(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, 32, 16)

        # SERES4: then copy X_ValA[48:64] -> X_ValA2 (special pattern like seres1)
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres4(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        # Original: X_ValA2 = torch.cat([X_ValA2[:, :32], X_ValA2[:, 48:64], X_ValA[:, 48:64], X_ValA2[:, 64:]], dim=1)
        X_ValA2 = torch.cat([X_ValA2[:, :32], X_ValA2[:, 48:64], X_ValA[:, 48:64], X_ValA2[:, 64:]], dim=1)
        X_ValB2 = torch.cat([X_ValB2[:, :32], X_ValB2[:, 48:64], X_ValB[:, 48:64], X_ValB2[:, 64:]], dim=1)
        X_ValC2 = torch.cat([X_ValC2[:, :32], X_ValC2[:, 48:64], X_ValC[:, 48:64], X_ValC2[:, 64:]], dim=1)

        # SERES5: then add input residual at [64:96]
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres5(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        X_ValA2, X_ValB2, X_ValC2 = self._add_input_residual(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, 64, 32)

        # SERES6: then add input residual at [96:128]
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres6(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        X_ValA2, X_ValB2, X_ValC2 = self._add_input_residual(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, 96, 32)

        # SERES7: then add input residual at [128:160]
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres7(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        X_ValA2, X_ValB2, X_ValC2 = self._add_input_residual(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, 128, 32)

        # Final projections: last 32 dims -> 32 for each branch
        comb_main = self.final_main(X_ValA2[:, 128:])
        comb_sub = self.final_sub(X_ValB2[:, 128:])
        comb_hg = self.final_hg(X_ValC2[:, 128:])

        comb = torch.cat([comb_main, comb_sub, comb_hg], dim=1)  # [batch, 96]
        return comb, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2

    @staticmethod
    def _prepare_pool_features(X_Val, g, gamma, batch_size, device):
        if g is not None and gamma is not None:
            if isinstance(g, (int, float)):
                g = torch.tensor(g, device=device, dtype=X_Val.dtype).expand(batch_size)
            if isinstance(gamma, (int, float)):
                gamma = torch.tensor(gamma, device=device, dtype=X_Val.dtype).expand(batch_size)
            return torch.stack([g, gamma], dim=1)
        return None


class DirectionBranch(nn.Module):
    """
    Direction Information branch
    Processes 160-dim input through 4 SE/SE-Res blocks
    Output: 24-dim (8*3)
    """

    def __init__(self):
        super(DirectionBranch, self).__init__()
        self.seres0 = SEBlockWithoutResidual(160, 8, 8, se_dim=8)
        self.seres1 = SEBlockWithResidual(0, 168, 8, 8, se_dim=8)
        self.seres2 = SEBlockWithResidual(8, 176, 16, 8, se_dim=8)
        self.seres3 = SEBlockWithResidual(16, 184, 24, 8, se_dim=8)

        self.final_main = LinearReLU(8, 8)
        self.final_sub = LinearReLU(8, 8)
        self.final_hg = LinearReLU(8, 8)

    def _add_input_residual(self, X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, val_offset, a2_offset, size):
        """Add input features as residual: X_ValA2[a2_offset:+size] += X_Val[val_offset:+size]"""
        X_ValA2 = replace_slice(X_ValA2, a2_offset, size, X_ValA2[:, a2_offset:a2_offset + size] + X_Val[:, val_offset:val_offset + size])
        X_ValB2 = replace_slice(X_ValB2, a2_offset, size, X_ValB2[:, a2_offset:a2_offset + size] + X_Val_Sub[:, val_offset:val_offset + size])
        X_ValC2 = replace_slice(X_ValC2, a2_offset, size, X_ValC2[:, a2_offset:a2_offset + size] + X_Val_Hg[:, val_offset:val_offset + size])
        return X_ValA2, X_ValB2, X_ValC2

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, g=None, gamma=None):
        batch_size = X_Val.shape[0]
        device = X_Val.device

        pool_features = MainBranch._prepare_pool_features(X_Val, g, gamma, batch_size, device)

        # SERES0: then add input residual X_Val[160:168] -> X_ValA2[0:8]
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres0(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        X_ValA2, X_ValB2, X_ValC2 = self._add_input_residual(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, 160, 0, 8)

        # SERES1: then add input residual X_Val[168:176] -> X_ValA2[8:16]
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres1(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        X_ValA2, X_ValB2, X_ValC2 = self._add_input_residual(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, 168, 8, 8)

        # SERES2: then add input residual X_Val[176:184] -> X_ValA2[16:24]
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres2(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        X_ValA2, X_ValB2, X_ValC2 = self._add_input_residual(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, 176, 16, 8)

        # SERES3: then add input residual X_Val[184:192] -> X_ValA2[24:32]
        X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.seres3(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2, pool_features=pool_features)
        X_ValA2, X_ValB2, X_ValC2 = self._add_input_residual(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA2, X_ValB2, X_ValC2, 184, 24, 8)

        # Final projections
        comb_main = self.final_main(X_ValA2[:, 24:32])
        comb_sub = self.final_sub(X_ValB2[:, 24:32])
        comb_hg = self.final_hg(X_ValC2[:, 24:32])

        comb = torch.cat([comb_main, comb_sub, comb_hg], dim=1)  # [batch, 24]
        return comb, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2


class FinalFusion(nn.Module):
    """
    Final fusion and output layers
    Combines main and direction branches, applies final SE and outputs radiance
    """

    def __init__(self):
        super(FinalFusion, self).__init__()

        self.gg_layer = LinearReLU(3, 8)
        self.se_final1 = nn.Linear(75, 16, bias=False)
        self.se_final2 = nn.Linear(16, 6, bias=False)

        self.fusion_layers_1 = LinearReLU(128, 64)
        self.fusion_layers_2 = LinearReLU(64, 32)
        self.fusion_layers_3 = LinearReLU(32, 16)
        self.fusion_layers_4 = LinearReLU(16, 16)
        self.fusion_layers_5 = LinearReLU(16, 16)
        self.fusion_layers_6 = LinearReLU(16, 16)
        self.fusion_layers_7 = LinearReLU(16, 16)

        self.output_layer = nn.Linear(16, 1)

    def forward(self, comb_features, di_features, X_ValA, X_ValB, X_ValC, avg_pool, sc_base, g, gamma):
        batch_size = comb_features.shape[0]

        # Concatenate features: 96 + 24 = 120
        comb = torch.cat([comb_features, di_features], dim=1)

        g_expanded = g.unsqueeze(-1) if g.dim() == 1 else g
        gamma_expanded = gamma.unsqueeze(-1) if gamma.dim() == 1 else gamma
        sc_base_expanded = sc_base.unsqueeze(-1) if sc_base.dim() == 1 else sc_base

        X_ValA = replace_slice(X_ValA, 0, 120, comb[:, :120])

        # Fill avg_pool with g, gamma, sc_base
        avg_pool[:, 72] = g_expanded.squeeze()
        avg_pool[:, 73] = gamma_expanded.squeeze()
        avg_pool[:, 74] = sc_base_expanded.squeeze()

        # GG layer
        X_ValA_temp = F.relu(self.gg_layer(avg_pool[:, 72:]))
        X_ValA = torch.cat([X_ValA[:, :120], X_ValA_temp, X_ValA[:, 128:]], dim=1)

        # SE final weights
        se_weight = F.relu(self.se_final1(avg_pool))
        se_weight = torch.sigmoid(self.se_final2(se_weight))

        # Apply SE weights to different feature groups
        X_ValA = torch.cat([
            X_ValA[:, 0:32] * se_weight[:, 0:1],
            X_ValA[:, 32:64] * se_weight[:, 1:2],
            X_ValA[:, 64:96] * se_weight[:, 2:3],
            X_ValA[:, 96:104] * se_weight[:, 3:4],
            X_ValA[:, 104:112] * se_weight[:, 4:5],
            X_ValA[:, 112:120] * se_weight[:, 5:6],
            X_ValA[:, 120:]
        ], dim=1)

        # Fusion layers (Plan D: 128->64->32->16 then residual blocks)
        X_ValB_new = self.fusion_layers_1(X_ValA[:, :128])
        X_ValB = replace_slice(X_ValB, 0, 64, X_ValB_new)
        X_ValA_new = self.fusion_layers_2(X_ValB[:, :64])
        X_ValA = replace_slice(X_ValA, 0, 32, X_ValA_new)
        X_ValB_new = self.fusion_layers_3(X_ValA[:, :32])
        X_ValB = replace_slice(X_ValB, 0, 16, X_ValB_new)
        X_ValA_new = self.fusion_layers_4(X_ValB[:, :16])
        X_ValA = replace_slice(X_ValA, 0, 16, X_ValA_new)
        X_ValB_new = self.fusion_layers_5(X_ValA[:, :16])
        X_ValB = replace_slice(X_ValB, 0, 16, X_ValB_new)
        X_ValC_new = self.fusion_layers_6(X_ValB[:, :16])
        X_ValC = replace_slice(X_ValC, 0, 16, X_ValC_new)

        # Residual block 1
        X_ValA = replace_slice(X_ValA, 0, 16, X_ValA[:, :16] + X_ValC[:, :16])
        X_ValB = replace_slice(X_ValB, 0, 16, self.fusion_layers_6(X_ValA[:, :16]))
        X_ValC = replace_slice(X_ValC, 0, 16, self.fusion_layers_7(X_ValB[:, :16]))

        # Residual block 2
        X_ValA = replace_slice(X_ValA, 0, 16, X_ValA[:, :16] + X_ValC[:, :16])

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
        batch_size = X_Val.shape[0]
        device = X_Val.device

        X_ValA = torch.zeros([batch_size, 160], device=device)
        X_ValB = torch.zeros([batch_size, 160], device=device)
        X_ValC = torch.zeros([batch_size, 160], device=device)
        X_ValA2 = torch.zeros([batch_size, 160], device=device)
        X_ValB2 = torch.zeros([batch_size, 160], device=device)
        X_ValC2 = torch.zeros([batch_size, 160], device=device)
        avg_pool = torch.zeros([batch_size, 75], device=device)

        avg_pool = ComputeAvgPool(X_Val, X_Val_Sub, X_Val_Hg, avg_pool)

        # Convert scalar inputs to tensors
        if isinstance(g, (int, float)):
            g = torch.tensor(g, device=device, dtype=X_Val.dtype).expand(batch_size)
        elif g.dim() == 0:
            g = g.expand(batch_size)

        if gamma is None:
            gamma = torch.zeros(batch_size, device=device, dtype=X_Val.dtype)
        elif isinstance(gamma, (int, float)):
            gamma = torch.tensor(gamma, device=device, dtype=X_Val.dtype).expand(batch_size)
        elif gamma.dim() == 0:
            gamma = gamma.expand(batch_size)

        if scatterrate is None:
            scatterrate = torch.ones(3, device=device, dtype=X_Val.dtype)
        if scatterrate.dim() == 1:
            scatterrate = scatterrate.unsqueeze(0).expand(batch_size, -1)

        srp = scatterrate ** 4.0
        sc_base = srp

        # Main branch
        comb_main, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.main_branch(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC,
            X_ValA2, X_ValB2, X_ValC2, g=g, gamma=gamma)

        # Direction branch
        comb_di, X_ValA, X_ValB, X_ValC, X_ValA2, X_ValB2, X_ValC2 = self.direction_branch(
            X_Val, X_Val_Sub, X_Val_Hg, X_ValA, X_ValB, X_ValC,
            X_ValA2, X_ValB2, X_ValC2, g=g, gamma=gamma)

        # Final fusion and output
        output_0 = self.final_fusion(
            comb_main, comb_di, X_ValA, X_ValB, X_ValC, avg_pool, sc_base[:, 0], g, gamma)

        return output_0


def create_mrpnn_model() -> MRPNN:
    """Create MRPNN model instance."""
    return MRPNN()


if __name__ == "__main__":
    model = create_mrpnn_model()

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
