import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import add_self_loops, degree

def normalize_adj(edge_index, num_nodes):
    ei, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = ei
    deg = degree(col, num_nodes=num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    return ei, deg_inv_sqrt[row] * deg_inv_sqrt[col]

class LinearProjector(nn.Module):
    """通用适配器：解决维度失配，对齐异构特征"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.PReLU()
        
    def forward(self, x):
        return self.act(self.proj(x))

class SimplifiedGCN(nn.Module):
    """去除层间非线性激活的共享编码器，防止高频异常信号过平滑"""
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        ei, ew = normalize_adj(edge_index, num_nodes)
        
        h = x
        for _ in range(self.num_layers):
            row, col = ei
            out = torch.zeros_like(h)
            out.index_add_(0, row, h[col] * ew.unsqueeze(-1))
            h = out
            
        return self.lin(h)

class GraphSAGEEncoder(nn.Module):
    """GraphSAGE 编码器，增强局部邻域模式建模能力。"""
    def __init__(self, in_dim, out_dim, num_layers=2, aggr='mean', dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        dims = [in_dim] + [out_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr=aggr))

    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

class GATEncoder(nn.Module):
    """GAT 编码器，显式建模邻域注意力权重。"""
    def __init__(self, in_dim, out_dim, num_layers=2, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers <= 1:
            self.convs.append(GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout))
        else:
            hidden = max(1, out_dim // heads)
            self.convs.append(GATConv(in_dim, hidden, heads=heads, concat=True, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden * heads, hidden, heads=heads, concat=True, dropout=dropout))
            self.convs.append(GATConv(hidden * heads, out_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

def build_relation_encoder(cfg, in_dim, out_dim):
    enc_type = str(cfg.encoder_type).lower()
    if enc_type == 'graphsage':
        return GraphSAGEEncoder(
            in_dim,
            out_dim,
            num_layers=cfg.gcn_layers,
            aggr=cfg.sage_aggr,
            dropout=cfg.enc_dropout,
        )
    if enc_type == 'gat':
        return GATEncoder(
            in_dim,
            out_dim,
            num_layers=cfg.gcn_layers,
            heads=cfg.gat_heads,
            dropout=cfg.gat_dropout,
        )
    return SimplifiedGCN(in_dim, out_dim, cfg.gcn_layers)

class Decoder(nn.Module):
    def __init__(self, hid_dim, proj_dim):
        super().__init__()
        # 恢复到投影维度 (或原始维度，取决于实现，这里恢复到投影维度更稳定)
        self.attr_dec = nn.Linear(hid_dim, proj_dim)
        
    def forward(self, z):
        x_hat = self.attr_dec(z)
        # 结构重构采用内积并经 Sigmoid
        adj_hat = torch.sigmoid(torch.matmul(z, z.T))
        return x_hat, adj_hat
