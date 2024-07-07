import torch
from torch.nn import ModuleList, Linear
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool
from torch_geometric.utils import degree


# PNA model
def pre_deg(pre_data):
    # Compute in-degree histogram over training data.
    len_deg_safe = 500000  # take a sufficiently large value
    deg_safe = torch.zeros(len_deg_safe, dtype=torch.long)

    for data in pre_data:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg_safe += torch.bincount(d, minlength=deg_safe.numel())
    for i in range(len_deg_safe):
        if deg_safe[i:].sum() == 0:
            deg = deg_safe[:i]
            break
    return deg

class myPNA(torch.nn.Module):
    def __init__(self, data_list, in_channels, hidden_channels, num_layers):
        super(myPNA, self).__init__()

        # initialize parameters for PNA
        deg=pre_deg(pre_data=data_list)
        print(deg)
        aggregators = ['sum','mean','std','max','min']
        scalers = ['identity', 'amplification', 'attenuation']

        # initialize module list
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        # first layer
        conv = PNAConv(in_channels=in_channels, out_channels=hidden_channels,
                       aggregators=aggregators, scalers=scalers, deg=deg,
                       towers=3, pre_layers=1, post_layers=1,
                       divide_input=False)  # channel divided by towel
        self.convs.append(conv)
        self.batch_norms.append(BatchNorm(hidden_channels))

        # hidden layers
        for _ in range(num_layers-1):
            conv = PNAConv(in_channels=hidden_channels, out_channels=hidden_channels,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           towers=3, pre_layers=1, post_layers=1,
                           divide_input=False)  # channel divided by towel
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
        
        # last linear layer
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x




