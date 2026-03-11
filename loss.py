import torch
import torch.nn.functional as F

def dcor_loss(x, y, max_nodes=256):
    """距离相关性：捕捉重构结果间的非线性统计依赖性"""
    if x.size(0) != y.size(0):
        raise ValueError("x and y must have same number of nodes for dcor.")

    # 控制 O(N^2 * D) 复杂度，避免在大图上内存爆炸
    if x.size(0) > max_nodes:
        idx = torch.randperm(x.size(0), device=x.device)[:max_nodes]
        x = x[idx]
        y = y[idx]

    a = torch.norm(x[:, None] - x, p=2, dim=2)
    b = torch.norm(y[:, None] - y, p=2, dim=2)

    A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
    B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

    dcov2_xy = (A * B).mean()
    dcov2_xx = (A * A).mean()
    dcov2_yy = (B * B).mean()

    dcov2_xy = torch.clamp(dcov2_xy, min=0.0)
    dcov2_xx = torch.clamp(dcov2_xx, min=1e-12)
    dcov2_yy = torch.clamp(dcov2_yy, min=1e-12)

    dcor = torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy) + 1e-8)
    return dcor

def balanced_bce(adj_hat, adj_true):
    """缓解极度稀疏图中负样本远多于正样本导致的训练偏置。"""
    bce = F.binary_cross_entropy(adj_hat, adj_true, reduction='none')
    pos = adj_true.sum()
    neg = adj_true.numel() - pos
    pos_weight = (neg / (pos + 1e-8)).detach()
    weight = torch.where(adj_true > 0.5, pos_weight, 1.0)
    return (bce * weight).mean()

def compact_structure(adj_hat, struct_dim=16):
    """随机子列压缩结构表示，用于稳定计算 DCOR。"""
    num_nodes = adj_hat.size(0)
    k = min(struct_dim, num_nodes)
    col_idx = torch.randperm(num_nodes, device=adj_hat.device)[:k]
    return adj_hat[:, col_idx]

def info_nce_loss(z1, z2, temperature=0.2):
    """双向 InfoNCE，正样本为同一节点跨视图表示。"""
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_12 + loss_21)

def compute_dense_adj(edge_index, num_nodes):
    """将 edge_index 转换为稠密邻接矩阵用于计算 BCE"""
    adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj

def compute_total_loss(
    preds_A,
    preds_B,
    target_x,
    edge_indices_orig,
    lambda_dcor,
    z_views=None,
    lambda_nce=0.0,
    nce_temp=0.2,
    dcor_max_nodes=256,
    dcor_struct_dim=16,
):
    x_hat_A, adj_hat_A = preds_A
    x_hat_B, adj_hat_B = preds_B
    num_nodes = target_x.size(0)
    
    # 真实稠密图 (取视图1为基准重构，或多视图重构)
    adj_true = compute_dense_adj(edge_indices_orig[0], num_nodes)

    # 1. 原始视图重构误差 (学基础)
    recon_A = F.mse_loss(x_hat_A, target_x) + balanced_bce(adj_hat_A, adj_true)
    
    # 2. 混合上下文视图重构误差 (学纠错)
    recon_B = F.mse_loss(x_hat_B, target_x) + balanced_bce(adj_hat_B, adj_true)
    
    # 3. DCOR 一致性约束 (照妖镜)
    # 负号表示最大化相关性
    struct_A = compact_structure(adj_hat_A, dcor_struct_dim)
    struct_B = compact_structure(adj_hat_B, dcor_struct_dim)
    loss_dcor = -dcor_loss(x_hat_A, x_hat_B, max_nodes=dcor_max_nodes)
    loss_dcor = loss_dcor - dcor_loss(struct_A, struct_B, max_nodes=dcor_max_nodes)

    loss = recon_A + recon_B + lambda_dcor * loss_dcor

    if z_views is not None and lambda_nce > 0.0:
        z_A, z_B = z_views
        loss_nce = info_nce_loss(z_A, z_B, temperature=nce_temp)
        loss = loss + lambda_nce * loss_nce

    return loss
