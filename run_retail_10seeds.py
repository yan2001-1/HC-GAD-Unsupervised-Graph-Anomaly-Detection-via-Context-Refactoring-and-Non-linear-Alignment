import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest

from config import Config
from data import load_multiplex_data, random_subgraph_sampling
from augment import build_view_A_umgad, build_view_B_croc
from model import HCGAD
from loss import compute_total_loss
from main import build_optimizer_and_scheduler, compute_context_scores, minmax_scale_numpy
from utils import evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(seed: int):
    cfg = Config()
    cfg.data_path = r"C:\Users\ASUS\Desktop\data\Retail_like_32287.mat"
    cfg.seed = int(seed)

    # Best setting found for this dataset
    cfg.score_mode = "context_std"
    cfg.use_structural_fusion = False
    cfg.enable_aux_fusion = True
    cfg.aux_score_type = "if_raw"
    cfg.aux_fusion_weight = 0.95
    cfg.max_full_infer_nodes = 10000
    cfg.infer_batch_size = 1024

    set_seed(cfg.seed)
    features, edge_indices, labels, _ = load_multiplex_data(
        cfg.data_path,
        return_raw_edge_index=True,
        feature_knn_k=cfg.feature_knn_k,
        second_view_mode=cfg.second_view_mode,
    )
    cfg.in_dim = int(features.size(1))
    if cfg.normalize_features:
        features = F.normalize(features, p=2, dim=1)
    labels_np = labels.cpu().numpy()
    features_np = features.cpu().numpy()
    features = features.to(cfg.device)
    edge_indices = [ei.to(cfg.device) for ei in edge_indices]
    num_nodes = features.size(0)

    model = HCGAD(cfg).to(cfg.device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    model.train()

    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        batch_nodes, batch_edges = random_subgraph_sampling(num_nodes, edge_indices, cfg.batch_size)
        x_sub = features[batch_nodes]
        x_view_a, edge_a_v1 = build_view_A_umgad(
            x_sub, batch_edges[0], cfg.attr_mask_ratio, cfg.edge_mask_ratio
        )
        _, edge_a_v2 = build_view_A_umgad(
            x_sub, batch_edges[1], cfg.attr_mask_ratio, cfg.edge_mask_ratio
        )
        x_view_b = build_view_B_croc(features, batch_nodes, cfg.mix_ratio)
        preds_a, preds_b, z_views = model(x_view_a, [edge_a_v1, edge_a_v2], x_view_b, batch_edges)
        target_h = model.projector(x_sub).detach()
        loss = compute_total_loss(
            preds_a,
            preds_b,
            target_h,
            batch_edges,
            cfg.lambda_dcor,
            z_views=z_views,
            lambda_nce=cfg.lambda_nce,
            nce_temp=cfg.nce_temp,
            dcor_max_nodes=cfg.dcor_max_nodes,
            dcor_struct_dim=cfg.dcor_struct_dim,
        )
        loss.backward()
        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    model.eval()
    with torch.no_grad():
        torch.manual_seed(cfg.seed + 999)
        _, context_std_scores = compute_context_scores(
            model, features, edge_indices, cfg.mix_ratio, cfg.context_trials
        )
        context_std_scores = minmax_scale_numpy(context_std_scores)

    iso = IsolationForest(
        n_estimators=int(cfg.aux_if_estimators),
        contamination="auto",
        random_state=cfg.seed,
        n_jobs=1,
    )
    iso.fit(features_np)
    if_raw_scores = minmax_scale_numpy(-iso.score_samples(features_np))

    final_scores = (1.0 - cfg.aux_fusion_weight) * context_std_scores + cfg.aux_fusion_weight * if_raw_scores
    auc, ap = evaluate(labels_np, final_scores)
    return float(auc), float(ap)


def main():
    seeds = list(range(42, 52))
    aucs = []
    aps = []

    for seed in seeds:
        auc, ap = run_one(seed)
        aucs.append(auc)
        aps.append(ap)
        print(f"seed={seed} AUC={auc:.4f} AP={ap:.4f}", flush=True)

    aucs = np.asarray(aucs, dtype=np.float64)
    aps = np.asarray(aps, dtype=np.float64)
    print(
        "AUC mean/std/min/max/range",
        f"{aucs.mean():.4f}",
        f"{aucs.std(ddof=1):.4f}",
        f"{aucs.min():.4f}",
        f"{aucs.max():.4f}",
        f"{(aucs.max() - aucs.min()):.4f}",
    )
    print(
        "AP  mean/std/min/max/range",
        f"{aps.mean():.4f}",
        f"{aps.std(ddof=1):.4f}",
        f"{aps.min():.4f}",
        f"{aps.max():.4f}",
        f"{(aps.max() - aps.min()):.4f}",
    )


if __name__ == "__main__":
    main()
