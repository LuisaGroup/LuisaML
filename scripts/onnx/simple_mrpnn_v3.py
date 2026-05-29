import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Tuple
import torch.nn.functional as F
from thop import profile, clever_format


# =============================================================================
# Optimization Direction 2: Batched Linear for 3 branches
# Merge branch_main/branch_sub/branch_hg into a single batched matmul
# =============================================================================

class BatchedLinearReLU(nn.Module):
    """
    Batched Linear + ReLU for 3 branches.
    Instead of 3 separate Linear layers, use a single batched matmul.
    This reduces 3 kernel launches to 1.
    """

    def __init__(self, in_dim, out_dim, num_branches=3):
        super(BatchedLinearReLU, self).__init__()
        # weight: [num_branches, out_dim, in_dim]
        self.weight = nn.Parameter(torch.randn(num_branches, out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(num_branches, out_dim))
        self.num_branches = num_branches
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Initialize weights properly
        nn.init.kaiming_uniform_(self.weight.view(-1, in_dim))
        self.weight.data = self.weight.data.view(num_branches, out_dim, in_dim)

    def forward(self, x_stacked):
        """
        Args:
            x_stacked: [batch, num_branches, in_dim]
        Returns:
            out: [batch, num_branches, out_dim]
        """
        # x_stacked: [B, 3, in_dim] -> [B, 3, out_dim]
        # weight: [3, out_dim, in_dim] -> transpose to [3, in_dim, out_dim]
        # Use bmm: need [B*1, 3, in_dim] x [3, in_dim, out_dim]
        # Actually use einsum for clarity and efficiency
        out = torch.einsum('bni,noi->bno', x_stacked, self.weight) + self.bias.unsqueeze(0)
        return F.relu(out)


class LinearReLU(nn.Module):
    """Linear layer followed by ReLU activation"""

    def __init__(self, in_dim, out_dim, bias=True):
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return F.relu(self.linear(x))


class BatchedLinearReLUFinal(nn.Module):
    """
    Batched Linear + ReLU for final projection of 3 branches.
    Same as BatchedLinearReLU but used for final layers.
    """

    def __init__(self, in_dim, out_dim, num_branches=3):
        super(BatchedLinearReLUFinal, self).__init__()
        self.weight = nn.Parameter(torch.randn(num_branches, out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(num_branches, out_dim))
        self.num_branches = num_branches
        self.in_dim = in_dim
        self.out_dim = out_dim
        nn.init.kaiming_uniform_(self.weight.view(-1, in_dim))
        self.weight.data = self.weight.data.view(num_branches, out_dim, in_dim)

    def forward(self, x_stacked):
        """
        Args:
            x_stacked: [batch, num_branches, in_dim]
        Returns:
            out: [batch, num_branches * out_dim] (flattened)
        """
        out = torch.einsum('bni,noi->bno', x_stacked, self.weight) + self.bias.unsqueeze(0)
        out = F.relu(out)
        # Flatten: [B, 3, out_dim] -> [B, 3*out_dim]
        return out.reshape(out.shape[0], -1)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Computes attention weights for three feature streams.
    Optimized: accepts stacked input [batch, 3, size] to reduce cat operations.
    """

    def __init__(self, se_dim=8, output_dim=3):
        super(SEBlock, self).__init__()
        self.se_layer1 = nn.Linear(8, se_dim, bias=False)
        self.se_layer2 = nn.Linear(se_dim, output_dim, bias=False)

    def forward(self, x_stacked, pool_features):
        """
        Args:
            x_stacked: [batch, 3, size] - stacked main/sub/hg features
            pool_features: [batch, 2] - g and gamma
        Returns:
            se_weights: [batch, 3, 1] - attention weights (with keepdim for broadcasting)
        """
        # Compute avg and max for each branch: [batch, 3, 1] each
        avg_vals = x_stacked.mean(dim=-1)  # [batch, 3]
        max_vals = x_stacked.max(dim=-1)[0]  # [batch, 3]

        # Interleave: [avg0, max0, avg1, max1, avg2, max2] -> [batch, 6]
        # Stack and reshape to get interleaved order
        interleaved = torch.stack([avg_vals, max_vals], dim=-1)  # [batch, 3, 2]
        interleaved = interleaved.reshape(interleaved.shape[0], -1)  # [batch, 6]

        # Concat with pool_features: [batch, 8]
        pool_input = torch.cat([interleaved, pool_features], dim=-1)

        # SE weight generation
        se_weight = F.relu(self.se_layer1(pool_input))
        se_weight = torch.sigmoid(self.se_layer2(se_weight))  # [batch, 3]

        return se_weight.unsqueeze(-1)  # [batch, 3, 1]


# =============================================================================
# Optimization Direction 1: Eliminate slice+cat by using independent tensors
# Instead of maintaining X_ValA[batch, 160] and doing slice updates,
# each SERES block outputs independent small tensors.
# =============================================================================

class SEBlockWithoutResidual(nn.Module):
    """
    SE Block without residual connection.
    Optimized: operates on stacked [batch, 3, size] tensors, no slice/cat.
    """

    def __init__(self, size_x, se_dim=8):
        super(SEBlockWithoutResidual, self).__init__()
        self.size_x = size_x
        self.se_block = SEBlock(se_dim=se_dim, output_dim=3)
        # Direction 2: batched linear for 3 branches
        self.branches = BatchedLinearReLU(size_x, size_x, num_branches=3)

    def forward(self, x_input_stacked, pool_features):
        """
        Args:
            x_input_stacked: [batch, 3, size_x] - sliced from X_Val/Sub/Hg, stacked
            pool_features: [batch, 2]
        Returns:
            abc_mid: [batch, 3, size_x] - attention-weighted features (for X_ValA/B/C)
            abc2_mid: [batch, 3, size_x] - branch output (for X_ValA2/B2/C2)
        """
        # SE attention
        se_weights = self.se_block(x_input_stacked, pool_features)  # [batch, 3, 1]

        # Apply attention weights: [batch, 3, size_x]
        abc_mid = x_input_stacked * se_weights

        # Forward through batched branches: [batch, 3, size_x]
        abc2_mid = self.branches(abc_mid)

        return abc_mid, abc2_mid


class SEBlockWithResidual(nn.Module):
    """
    SE Block with residual connection.
    Optimized: operates on stacked [batch, 3, size] tensors, no slice/cat.
    """

    def __init__(self, size_x, se_dim=8):
        super(SEBlockWithResidual, self).__init__()
        self.size_x = size_x
        self.se_block = SEBlock(se_dim=se_dim, output_dim=3)
        # Direction 2: batched linear for 3 branches
        self.branches = BatchedLinearReLU(size_x, size_x, num_branches=3)

    def forward(self, x_input_stacked, prev_abc2_stacked, pool_features):
        """
        Args:
            x_input_stacked: [batch, 3, size_x] - sliced from X_Val/Sub/Hg, stacked
            prev_abc2_stacked: [batch, 3, size_x] - previous block's abc2 output (residual)
            pool_features: [batch, 2]
        Returns:
            abc_mid: [batch, 3, size_x] - attention-weighted + residual (for X_ValA/B/C)
            abc2_mid: [batch, 3, size_x] - branch output (for X_ValA2/B2/C2)
        """
        # SE attention
        se_weights = self.se_block(x_input_stacked, pool_features)  # [batch, 3, 1]

        # Apply attention weights
        abc_weighted = x_input_stacked * se_weights  # [batch, 3, size_x]

        # Add residual from previous block
        abc_mid = abc_weighted + prev_abc2_stacked  # [batch, 3, size_x]

        # Forward through batched branches
        abc2_mid = self.branches(abc_mid)  # [batch, 3, size_x]

        return abc_mid, abc2_mid


def ComputeAvgPool(X_Val, X_Val_Sub, X_Val_Hg):
    """
    Compute average and max pooling for all feature slices.
    Optimized: build the result via a single cat instead of index assignment.
    """
    POOL = 36
    # (from_dim, size_x) configuration table
    slices = [
        (0, 8), (8, 8), (16, 16), (32, 16),
        (48, 16), (64, 32), (96, 32), (128, 32),
        (160, 8), (168, 8), (176, 8), (184, 8),
    ]

    avg_parts = []
    max_parts = []

    for from_dim, size_x in slices:
        for feat in [X_Val, X_Val_Sub, X_Val_Hg]:
            s = feat[:, from_dim:from_dim + size_x]
            avg_parts.append(s.mean(dim=-1, keepdim=True))  # [batch, 1]
            max_parts.append(s.max(dim=-1, keepdim=True)[0])  # [batch, 1]

    # avg_parts: 12*3=36 items, max_parts: 36 items
    # AvgPool layout: [avg(36), max(36), g, gamma, sc_base]
    # g, gamma, sc_base will be filled later, so we just cat avg+max here
    avg_pool_72 = torch.cat(avg_parts + max_parts, dim=-1)  # [batch, 72]

    return avg_pool_72


class MainBranch(nn.Module):
    """
    Main feature extraction branch.
    Processes 192-dim input through 8 SE/SE-Res blocks.
    Output: 96-dim (32*3).

    Optimized:
    - Direction 1: Each SERES block outputs independent small tensors instead of
      updating slices of a large tensor. No more slice+cat inside blocks.
    - Direction 2: Three branches merged into batched matmul.
    """

    def __init__(self):
        super(MainBranch, self).__init__()

        # SERES blocks - note: no longer need from_dim/to_dim params
        self.seres0 = SEBlockWithoutResidual(8, se_dim=8)
        self.seres1 = SEBlockWithResidual(8, se_dim=8)
        self.seres2 = SEBlockWithResidual(16, se_dim=8)
        self.seres3 = SEBlockWithResidual(16, se_dim=8)
        self.seres4 = SEBlockWithResidual(16, se_dim=8)
        self.seres5 = SEBlockWithResidual(32, se_dim=8)
        self.seres6 = SEBlockWithResidual(32, se_dim=8)
        self.seres7 = SEBlockWithResidual(32, se_dim=8)

        # Final projection: batched 32->32 for 3 branches
        self.final_proj = BatchedLinearReLUFinal(32, 32, num_branches=3)

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg, pool_features):
        """
        Args:
            X_Val: [batch, 192]
            X_Val_Sub: [batch, 192]
            X_Val_Hg: [batch, 192]
            pool_features: [batch, 2]
        Returns:
            comb: [batch, 96] - concatenated features from 3 branches
            abc_list: list of [batch, 3, size_x] tensors for each block's abc output
            abc2_list: list of [batch, 3, size_x] tensors for each block's abc2 output
        """
        # Helper to stack slices from 3 inputs
        def stack_slice(f, t):
            return torch.stack([X_Val[:, f:t], X_Val_Sub[:, f:t], X_Val_Hg[:, f:t]], dim=1)

        # Helper to stack slices for input residual addition
        def input_residual_stacked(f, t):
            return torch.stack([X_Val[:, f:t], X_Val_Sub[:, f:t], X_Val_Hg[:, f:t]], dim=1)

        # ---- SERES0: input[0:8], no residual ----
        inp0 = stack_slice(0, 8)  # [batch, 3, 8]
        abc0, abc2_0 = self.seres0(inp0, pool_features)
        # Add input residual to abc2
        abc2_0 = abc2_0 + inp0

        # ---- SERES1: input[8:16], residual from seres0's abc2 ----
        inp1 = stack_slice(8, 16)  # [batch, 3, 8]
        abc1, abc2_1 = self.seres1(inp1, abc2_0, pool_features)
        # Special pattern: abc2_1 = abc1 (copy abc to abc2)
        abc2_1 = abc1.clone()

        # ---- SERES2: input[16:32], residual from seres1's abc2 ----
        # Note: seres1 output is 8-dim, seres2 input is 16-dim
        # The residual connection uses the last block's abc2 at matching offset
        # In v1: last=0 means residual from abc2[:, 0:16]
        # Since we use independent tensors, we need to cat seres0+seres1 abc2 for the 16-dim residual
        inp2 = stack_slice(16, 32)  # [batch, 3, 16]
        prev_abc2_for_2 = torch.cat([abc2_0, abc2_1], dim=-1)  # [batch, 3, 16]
        abc2, abc2_2 = self.seres2(inp2, prev_abc2_for_2, pool_features)
        # Add input residual
        abc2_2 = abc2_2 + inp2

        # ---- SERES3: input[32:48], residual from seres2's abc2 ----
        # last=16 means residual from abc2[:, 16:32] which is abc2_2
        inp3 = stack_slice(32, 48)  # [batch, 3, 16]
        abc3, abc2_3 = self.seres3(inp3, abc2_2, pool_features)
        # Add input residual
        abc2_3 = abc2_3 + inp3

        # ---- SERES4: input[48:64], residual from seres3's abc2 ----
        # last=32 means residual from abc2[:, 32:48] which is abc2_3 (but wait, in v1 last=32)
        # In v1: seres4 has last=32, from_dim=48, to_dim=48, size_x=16
        # residual = X_ValA2[:, 32:48] which corresponds to abc2_2 (offset 16-32 in the growing tensor)
        # Actually let me re-trace: the growing tensor at this point has:
        #   [0:8]=abc2_0, [8:16]=abc2_1, [16:32]=abc2_2, [32:48]=abc2_3
        # seres4 last=32 -> residual from [32:48] = abc2_3
        inp4 = stack_slice(48, 64)  # [batch, 3, 16]
        abc4, abc2_4 = self.seres4(inp4, abc2_3, pool_features)
        # Special pattern: abc2_4 = abc4 (copy abc to abc2, like seres1)
        abc2_4 = abc4.clone()

        # ---- SERES5: input[64:96], residual from previous ----
        # last=32 means residual from [32:64] = cat(abc2_3, abc2_4) = 32-dim
        inp5 = stack_slice(64, 96)  # [batch, 3, 32]
        prev_abc2_for_5 = torch.cat([abc2_3, abc2_4], dim=-1)  # [batch, 3, 32]
        abc5, abc2_5 = self.seres5(inp5, prev_abc2_for_5, pool_features)
        # Add input residual
        abc2_5 = abc2_5 + inp5

        # ---- SERES6: input[96:128], residual from seres5's abc2 ----
        # last=64 means residual from [64:96] = abc2_5
        inp6 = stack_slice(96, 128)  # [batch, 3, 32]
        abc6, abc2_6 = self.seres6(inp6, abc2_5, pool_features)
        # Add input residual
        abc2_6 = abc2_6 + inp6

        # ---- SERES7: input[128:160], residual from seres6's abc2 ----
        # last=96 means residual from [96:128] = abc2_6
        inp7 = stack_slice(128, 160)  # [batch, 3, 32]
        abc7, abc2_7 = self.seres7(inp7, abc2_6, pool_features)
        # Add input residual
        abc2_7 = abc2_7 + inp7

        # Final projection: use the last block's abc2 (32-dim)
        # abc2_7: [batch, 3, 32]
        comb = self.final_proj(abc2_7)  # [batch, 96]

        # Collect all abc and abc2 for later use in FinalFusion
        # abc_list stores the abc (attention-weighted) outputs
        abc_list = [abc0, abc1, abc2, abc3, abc4, abc5, abc6, abc7]
        abc2_list = [abc2_0, abc2_1, abc2_2, abc2_3, abc2_4, abc2_5, abc2_6, abc2_7]

        return comb, abc_list, abc2_list


class DirectionBranch(nn.Module):
    """
    Direction Information branch.
    Processes direction features through 4 SE/SE-Res blocks.
    Output: 24-dim (8*3).

    Optimized with same Direction 1 & 2 techniques.
    """

    def __init__(self):
        super(DirectionBranch, self).__init__()

        self.seres0 = SEBlockWithoutResidual(8, se_dim=8)
        self.seres1 = SEBlockWithResidual(8, se_dim=8)
        self.seres2 = SEBlockWithResidual(8, se_dim=8)
        self.seres3 = SEBlockWithResidual(8, se_dim=8)

        # Final projection: batched 8->8 for 3 branches
        self.final_proj = BatchedLinearReLUFinal(8, 8, num_branches=3)

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg, pool_features):
        """
        Args:
            X_Val: [batch, 192]
            X_Val_Sub: [batch, 192]
            X_Val_Hg: [batch, 192]
            pool_features: [batch, 2]
        Returns:
            comb: [batch, 24]
            abc_list: list of [batch, 3, 8] tensors
            abc2_list: list of [batch, 3, 8] tensors
        """
        def stack_slice(f, t):
            return torch.stack([X_Val[:, f:t], X_Val_Sub[:, f:t], X_Val_Hg[:, f:t]], dim=1)

        # ---- SERES0: input[160:168] ----
        inp0 = stack_slice(160, 168)  # [batch, 3, 8]
        abc0, abc2_0 = self.seres0(inp0, pool_features)
        abc2_0 = abc2_0 + inp0

        # ---- SERES1: input[168:176], residual from seres0 ----
        inp1 = stack_slice(168, 176)  # [batch, 3, 8]
        abc1, abc2_1 = self.seres1(inp1, abc2_0, pool_features)
        abc2_1 = abc2_1 + inp1

        # ---- SERES2: input[176:184], residual from seres1 ----
        inp2 = stack_slice(176, 184)  # [batch, 3, 8]
        abc2_val, abc2_2 = self.seres2(inp2, abc2_1, pool_features)
        abc2_2 = abc2_2 + inp2

        # ---- SERES3: input[184:192], residual from seres2 ----
        inp3 = stack_slice(184, 192)  # [batch, 3, 8]
        abc3, abc2_3 = self.seres3(inp3, abc2_2, pool_features)
        abc2_3 = abc2_3 + inp3

        # Final projection: use the last block's abc2 (8-dim)
        comb = self.final_proj(abc2_3)  # [batch, 24]

        abc_list = [abc0, abc1, abc2_val, abc3]
        abc2_list = [abc2_0, abc2_1, abc2_2, abc2_3]

        return comb, abc_list, abc2_list


class FinalFusion(nn.Module):
    """
    Final fusion and output layers.
    Combines main and direction branches, applies final SE and outputs radiance.

    Optimized:
    - Direction 1: Eliminated most cat/slice operations in fusion layers.
      Instead of updating slices of X_ValA/B/C, we pass tensors directly.
    """

    def __init__(self):
        super(FinalFusion, self).__init__()

        self.gg_layer = LinearReLU(3, 8)
        self.se_final1 = nn.Linear(75, 16, bias=False)
        self.se_final2 = nn.Linear(16, 6, bias=False)

        # Fusion layers: 128 -> 64 -> 32 -> 16 -> residual blocks
        self.fusion_layers_1 = LinearReLU(128, 64)
        self.fusion_layers_2 = LinearReLU(64, 32)
        self.fusion_layers_3 = LinearReLU(32, 16)
        self.fusion_layers_4 = LinearReLU(16, 16)
        self.fusion_layers_5 = LinearReLU(16, 16)
        self.fusion_layers_6 = LinearReLU(16, 16)
        self.fusion_layers_7 = LinearReLU(16, 16)

        self.output_layer = nn.Linear(16, 1)

    def forward(self, comb_features, di_features, avg_pool_72, sc_base, g, gamma):
        """
        Args:
            comb_features: [batch, 96] from main branch
            di_features: [batch, 24] from direction branch
            avg_pool_72: [batch, 72] pre-computed avg/max pool
            sc_base: [batch] scatter rate base
            g: [batch]
            gamma: [batch]
        """
        # Build the 120-dim combined feature
        comb_120 = torch.cat([comb_features, di_features], dim=1)  # [batch, 120]

        # Build avg_pool[75] = [avg_pool_72, g, gamma, sc_base]
        g_1d = g.unsqueeze(-1) if g.dim() == 1 else g
        gamma_1d = gamma.unsqueeze(-1) if gamma.dim() == 1 else gamma
        sc_1d = sc_base.unsqueeze(-1) if sc_base.dim() == 1 else sc_base
        avg_pool = torch.cat([avg_pool_72, g_1d, gamma_1d, sc_1d], dim=-1)  # [batch, 75]

        # GG layer: [g, gamma, sc_base] -> 8-dim
        gg_feat = self.gg_layer(avg_pool[:, 72:])  # [batch, 8]

        # Build X_ValA[128] = [comb_120, gg_feat]
        x_128 = torch.cat([comb_120, gg_feat], dim=1)  # [batch, 128]

        # SE final weights
        se_weight = F.relu(self.se_final1(avg_pool))  # [batch, 16]
        se_weight = torch.sigmoid(self.se_final2(se_weight))  # [batch, 6]

        # Apply SE weights to different feature groups
        # Groups: [0:32]*w0, [32:64]*w1, [64:96]*w2, [96:104]*w3, [104:112]*w4, [112:120]*w5, [120:128] unchanged
        x_128 = torch.cat([
            x_128[:, 0:32] * se_weight[:, 0:1],
            x_128[:, 32:64] * se_weight[:, 1:2],
            x_128[:, 64:96] * se_weight[:, 2:3],
            x_128[:, 96:104] * se_weight[:, 3:4],
            x_128[:, 104:112] * se_weight[:, 4:5],
            x_128[:, 112:120] * se_weight[:, 5:6],
            x_128[:, 120:]
        ], dim=1)

        # Fusion layers - Direction 1: no more slice+cat, just pass tensors directly
        h = self.fusion_layers_1(x_128)       # [batch, 64]
        h = self.fusion_layers_2(h)           # [batch, 32]
        h = self.fusion_layers_3(h)           # [batch, 16]
        h = self.fusion_layers_4(h)           # [batch, 16]
        h_a = h                               # save for residual
        h = self.fusion_layers_5(h)           # [batch, 16]
        h = self.fusion_layers_6(h)           # [batch, 16]

        # Residual block 1
        h_a = h_a + h                         # [batch, 16]
        h = self.fusion_layers_6(h_a)         # [batch, 16]
        h = self.fusion_layers_7(h)           # [batch, 16]

        # Residual block 2
        h_a = h_a + h                         # [batch, 16]

        # Output
        output = self.output_layer(h_a)       # [batch, 1]

        return output


class MRPNN(nn.Module):
    """
    Complete MRPNN Network - Optimized v3
    Optimizations:
    1. Eliminated most torch.cat/slice by using independent small tensors
    2. Merged 3-branch LinearReLU into batched matmul
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
            X_Val: [batch, 192]
            X_Val_Sub: [batch, 192]
            X_Val_Hg: [batch, 192]
            scatterrate: [batch, 3] or [3]
            g: scattering parameter
            gamma: angle parameter
        Returns:
            output: [batch, 1]
        """
        batch_size = X_Val.shape[0]
        device = X_Val.device
        
        target_dtype = next(self.parameters()).dtype
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
        elif scatterrate.dim() == 2 and scatterrate.shape[0] == 1:
            scatterrate = scatterrate.expand(batch_size, -1)

        # Compute scatter rate base
        srp = scatterrate ** 4.0
        sc_base = srp

        # Pool features for SE blocks
        pool_features = torch.stack([g, gamma], dim=1)  # [batch, 2]

        # Compute avg pool (Direction 1: single cat instead of index assignment)
        avg_pool_72 = ComputeAvgPool(X_Val, X_Val_Sub, X_Val_Hg)  # [batch, 72]

        # Main branch
        comb_main, main_abc_list, main_abc2_list = self.main_branch(
            X_Val, X_Val_Sub, X_Val_Hg, pool_features)

        # Direction branch
        comb_di, di_abc_list, di_abc2_list = self.direction_branch(
            X_Val, X_Val_Sub, X_Val_Hg, pool_features)

        # Final fusion for each scatter rate channel
        output_0 = self.final_fusion(
            comb_main, comb_di, avg_pool_72, sc_base[:, 0], g, gamma)
        output_1 = self.final_fusion(
            comb_main, comb_di, avg_pool_72, sc_base[:, 1], g, gamma)
        output_2 = self.final_fusion(
            comb_main, comb_di, avg_pool_72, sc_base[:, 2], g, gamma)
        # TODO 单色可以换单通道输出，提升性能
        # Apply exponential and scatter rate scaling (as in CUDA)
        output = torch.clamp(
            torch.exp(torch.cat([output_0, output_1, output_2], dim=1)) - 1.0, min=0.0)
        output = output * srp

        return output


def create_mrpnn_model() -> MRPNN:
    """Create MRPNN model instance."""
    model = MRPNN()
    return model


if __name__ == "__main__":
    model = create_mrpnn_model()

    batch_size = 4
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

    print(f"Input shape: X_Val{X_Val.shape}, X_Val_Sub{X_Val_Sub.shape}, X_Val_Hg{X_Val_Hg.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
