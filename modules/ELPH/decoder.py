import torch
from torch.nn import Linear
import torch.nn.functional as F

class LinkPredictor(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z, edge_index, elph_feat):
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        h = self.lin_src(torch.cat([z_src, elph_feat], dim=-1)) + self.lin_dst(torch.cat([z_dst, elph_feat], dim=-1))
        h = h.relu()
        return self.lin_final(h) #.sigmoid()
    
class LinkPredictor_h(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z, edge_index, elph_feat):
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        h = torch.mul(z_src, z_dst).reshape(-1, self.in_channels)
        return self.lin_final(torch.cat([h, elph_feat], dim=-1)) #.sigmoid()