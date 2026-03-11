import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import LinearProjector, build_relation_encoder, Decoder

class HCGAD(nn.Module):
    """Hybrid Contextual & DCOR Graph Anomaly Detection"""
    def __init__(self, cfg):
        super().__init__()
        self.projector = LinearProjector(cfg.in_dim, cfg.proj_dim)
        
        # 共享编码器 (对多重图的每个视图独立编码，但参数可以共享或各自独立，这里用独立以捕获不同拓扑)
        self.num_relations = 2 
        self.encoders = nn.ModuleList([
            build_relation_encoder(cfg, cfg.proj_dim, cfg.hid_dim)
            for _ in range(self.num_relations)
        ])
        
        # 跨视图 Attention 融合权重
        self.attn_weights = nn.Parameter(torch.ones(self.num_relations))
        
        # 共享解码器
        self.decoder = Decoder(cfg.hid_dim, cfg.proj_dim)

    def forward_encoder(self, x_proj, edge_indices):
        """执行多关系图编码并融合"""
        z_views = []
        for i, enc in enumerate(self.encoders):
            z_v = enc(x_proj, edge_indices[i])
            z_views.append(z_v)
            
        # Attention 加权融合
        w = F.softmax(self.attn_weights, dim=0)
        z_fused = sum(w[i] * z_views[i] for i in range(self.num_relations))
        return z_fused

    def forward(self, x_view_A, edge_indices_A, x_view_B, edge_indices_B):
        # 1. 通用适配器对齐 (共用 Projector)
        h_A = self.projector(x_view_A)
        h_B = self.projector(x_view_B)
        
        # 2. 编码
        z_A = self.forward_encoder(h_A, edge_indices_A)
        z_B = self.forward_encoder(h_B, edge_indices_B)
        
        # 3. 解码重构
        x_hat_A, adj_hat_A = self.decoder(z_A)
        x_hat_B, adj_hat_B = self.decoder(z_B)

        return (x_hat_A, adj_hat_A), (x_hat_B, adj_hat_B), (z_A, z_B)
