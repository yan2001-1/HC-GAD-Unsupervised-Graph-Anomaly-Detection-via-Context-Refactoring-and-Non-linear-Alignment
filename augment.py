import torch
from torch_geometric.utils import dropout_edge

def build_view_A_umgad(x, edge_index, attr_mask_ratio, edge_mask_ratio):
    """
    视图 A：原始掩码视图 (UMGAD) - 迫使模型填补缺失
    """
    # 1. 属性掩码 (归零)
    mask = torch.rand(x.size(0), device=x.device) < attr_mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0.0 
    
    # 2. 边掩码 (随机丢弃部分边)
    edge_index_masked, _ = dropout_edge(edge_index, p=edge_mask_ratio, force_undirected=True)
    return x_masked, edge_index_masked

def build_view_B_croc(x_full, batch_nodes, mix_ratio):
    """
    视图 B：混合上下文视图 (CRoC) - 制造结构-属性错位
    注意：x_full 是全图特征，仅针对 batch_nodes 引入外部背景噪声混合
    """
    x_mixed = x_full.clone()
    num_total = x_full.size(0)
    
    # 为采样集内的每个节点，随机抽取一个全局背景节点
    bg_idx = torch.randint(0, num_total, (len(batch_nodes),), device=x_full.device)
    
    # 特征混合 (CRoC 核心思想)
    x_mixed[batch_nodes] = mix_ratio * x_full[batch_nodes] + (1 - mix_ratio) * x_full[bg_idx]
    
    # 提取混合后对应的子图特征返回
    return x_mixed[batch_nodes]
