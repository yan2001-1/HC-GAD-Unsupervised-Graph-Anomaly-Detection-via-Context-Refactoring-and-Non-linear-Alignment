import copy
import math
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import subgraph

from augment import build_view_A_umgad, build_view_B_croc
from config import Config
from data import load_multiplex_data, random_subgraph_sampling
from loss import compute_dense_adj, compute_total_loss
from model import HCGAD
from utils import evaluate, evaluate_binary, get_anomaly_scores, select_threshold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def minmax_scale_tensor(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def minmax_scale_numpy(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def compute_context_scores(model, features, edge_indices, mix_ratio, trials):
    num_nodes = features.size(0)
    all_nodes = torch.arange(num_nodes, device=features.device)

    h_ref = model.projector(features)
    z_ref = model.forward_encoder(h_ref, edge_indices)

    trial_scores = []
    for _ in range(trials):
        x_mix = build_view_B_croc(features, all_nodes, mix_ratio)
        z_mix = model.forward_encoder(model.projector(x_mix), edge_indices)
        score = 1.0 - F.cosine_similarity(z_ref, z_mix, dim=1)
        trial_scores.append(score)

    stacked = torch.stack(trial_scores, dim=1)
    score_mean = minmax_scale_tensor(stacked.mean(dim=1))
    score_std = minmax_scale_tensor(stacked.std(dim=1))
    return score_mean.cpu().numpy(), score_std.cpu().numpy()


def compute_inverse_pagerank_scores(num_nodes, edge_index, alpha=0.85, max_iter=200, tol=1e-10):
    if edge_index is None:
        return None

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    data = np.ones(len(src), dtype=np.float64)
    adj = sp.csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))

    row_sum = np.asarray(adj.sum(axis=1)).reshape(-1)
    row_sum[row_sum == 0.0] = 1.0
    trans = sp.diags(1.0 / row_sum).dot(adj)

    pr = np.ones(num_nodes, dtype=np.float64) / float(num_nodes)
    for _ in range(max_iter):
        pr_new = (1.0 - alpha) / float(num_nodes) + alpha * (trans.T.dot(pr))
        if np.abs(pr_new - pr).sum() < tol:
            pr = pr_new
            break
        pr = pr_new

    inv_pr = 1.0 - pr
    inv_pr = (inv_pr - inv_pr.min()) / (inv_pr.max() - inv_pr.min() + 1e-12)
    return inv_pr


def compute_structural_scores(num_nodes, edge_index, score_type, pagerank_alpha=0.85):
    stype = str(score_type).lower()
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    if stype == "inv_pagerank":
        return compute_inverse_pagerank_scores(num_nodes, edge_index, alpha=pagerank_alpha)

    out_deg = np.bincount(src, minlength=num_nodes).astype(np.float64)
    in_deg = np.bincount(dst, minlength=num_nodes).astype(np.float64)

    if stype == "in_degree":
        s = in_deg
    elif stype == "degree_gap":
        s = np.abs(in_deg - out_deg)
    else:
        s = out_deg

    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return s


def compute_aux_scores(cfg, features_np, h_np, z_np=None):
    stype = str(cfg.aux_score_type).lower()
    if stype == "if_raw":
        iso = IsolationForest(
            n_estimators=int(cfg.aux_if_estimators),
            contamination="auto",
            random_state=cfg.seed,
            n_jobs=1,
        )
        iso.fit(features_np)
        return minmax_scale_numpy(-iso.score_samples(features_np))

    if stype == "if_h":
        iso = IsolationForest(
            n_estimators=int(cfg.aux_if_estimators),
            contamination="auto",
            random_state=cfg.seed,
            n_jobs=1,
        )
        iso.fit(h_np)
        return minmax_scale_numpy(-iso.score_samples(h_np))
    if stype == "if_z" and z_np is not None:
        iso = IsolationForest(
            n_estimators=int(cfg.aux_if_estimators),
            contamination="auto",
            random_state=cfg.seed,
            n_jobs=1,
        )
        iso.fit(z_np)
        return minmax_scale_numpy(-iso.score_samples(z_np))
    if stype == "knn_z" and z_np is not None:
        k = int(getattr(cfg, "aux_knn_k", 30))
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=1)
        nbrs.fit(z_np)
        dist, _ = nbrs.kneighbors(z_np)
        half = max(1, k // 2)
        return minmax_scale_numpy(dist[:, half:].mean(axis=1))

    if stype == "lof_raw":
        lof = LocalOutlierFactor(
            n_neighbors=int(cfg.aux_lof_k),
            contamination="auto",
            novelty=False,
            metric="cosine",
            n_jobs=1,
        )
        _ = lof.fit_predict(features_np)
        return minmax_scale_numpy(-lof.negative_outlier_factor_)

    lof = LocalOutlierFactor(
        n_neighbors=int(cfg.aux_lof_k),
        contamination="auto",
        novelty=False,
        metric="euclidean",
        n_jobs=1,
    )
    _ = lof.fit_predict(h_np)
    return minmax_scale_numpy(-lof.negative_outlier_factor_)


def compute_recon_scores_batched(model, features, edge_indices, alpha_score, batch_size=1024):
    num_nodes = int(features.size(0))
    recon_scores = np.zeros(num_nodes, dtype=np.float64)
    device = features.device
    all_nodes = torch.arange(num_nodes, device=device)

    for start in range(0, num_nodes, batch_size):
        end = min(num_nodes, start + batch_size)
        batch_nodes = all_nodes[start:end]
        batch_edges = []
        for ei in edge_indices:
            sub_ei, _ = subgraph(batch_nodes, ei, relabel_nodes=True, num_nodes=num_nodes)
            batch_edges.append(sub_ei)

        h_sub = model.projector(features[batch_nodes])
        z_sub = model.forward_encoder(h_sub, batch_edges)
        x_hat_sub, adj_hat_sub = model.decoder(z_sub)
        adj_true_sub = compute_dense_adj(batch_edges[0], batch_nodes.numel())
        recon_sub = get_anomaly_scores(h_sub, adj_true_sub, x_hat_sub, adj_hat_sub, alpha_score)
        recon_scores[start:end] = recon_sub

    return recon_scores


def build_optimizer_and_scheduler(model, cfg):
    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = None
    if cfg.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.max_lr,
            total_steps=cfg.epochs,
            pct_start=cfg.pct_start,
            div_factor=cfg.onecycle_div_factor,
            final_div_factor=cfg.final_div_factor,
        )
    elif cfg.scheduler == "cosine":
        warmup_epochs = max(1, int(cfg.pct_start * cfg.epochs))

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, cfg.epochs - warmup_epochs)
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main():
    cfg = Config()
    set_seed(cfg.seed)

    print("1. Loading Data...")
    features, edge_indices, labels, edge_index_raw = load_multiplex_data(
        cfg.data_path,
        return_raw_edge_index=True,
        feature_knn_k=cfg.feature_knn_k,
        second_view_mode=cfg.second_view_mode,
    )
    cfg.in_dim = int(features.size(1))
    if cfg.normalize_features:
        features = F.normalize(features, p=2, dim=1)

    labels_np = labels.cpu().numpy() if labels is not None else None
    features = features.to(cfg.device)
    edge_indices = [ei.to(cfg.device) for ei in edge_indices]
    num_total_nodes = features.size(0)
    print(f"   dataset={cfg.data_path} | nodes={num_total_nodes} | feat_dim={cfg.in_dim}")

    print("2. Building Model...")
    model = HCGAD(cfg).to(cfg.device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    print("3. Start Training...")
    model.train()
    best_loss = float("inf")
    best_state = None

    for epoch in range(cfg.epochs):
        optimizer.zero_grad()

        batch_nodes, batch_edges = random_subgraph_sampling(num_total_nodes, edge_indices, cfg.batch_size)
        x_sub = features[batch_nodes]

        x_view_a, edge_a_v1 = build_view_A_umgad(x_sub, batch_edges[0], cfg.attr_mask_ratio, cfg.edge_mask_ratio)
        _, edge_a_v2 = build_view_A_umgad(x_sub, batch_edges[1], cfg.attr_mask_ratio, cfg.edge_mask_ratio)
        edges_view_a = [edge_a_v1, edge_a_v2]

        x_view_b = build_view_B_croc(features, batch_nodes, cfg.mix_ratio)
        edges_view_b = batch_edges

        preds_a, preds_b, z_views = model(x_view_a, edges_view_a, x_view_b, edges_view_b)
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

        if cfg.save_best_by_loss and loss.item() < best_loss:
            best_loss = loss.item()
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | LR: {lr_now:.6f}")

    if cfg.save_best_by_loss and best_state is not None:
        model.load_state_dict(best_state)

    print("4. Inference & Unsupervised Evaluation...")
    model.eval()
    with torch.no_grad():
        h_all = model.projector(features)
        z_all = model.forward_encoder(h_all, edge_indices)

        max_full_nodes = int(getattr(cfg, "max_full_infer_nodes", 10000))
        infer_batch_size = int(getattr(cfg, "infer_batch_size", 1024))
        if num_total_nodes <= max_full_nodes:
            x_hat, adj_hat = model.decoder(z_all)
            adj_true = compute_dense_adj(edge_indices[0], num_total_nodes)
            recon_scores = get_anomaly_scores(h_all, adj_true, x_hat, adj_hat, cfg.alpha_score)
        else:
            recon_scores = compute_recon_scores_batched(
                model, features, edge_indices, cfg.alpha_score, batch_size=infer_batch_size
            )

        torch.manual_seed(cfg.seed + 999)
        context_mean_scores, context_std_scores = compute_context_scores(
            model, features, edge_indices, cfg.mix_ratio, cfg.context_trials
        )

        w_mean = float(cfg.combo_mean_weight)
        combo_scores = w_mean * context_mean_scores + (1.0 - w_mean) * context_std_scores

        if cfg.score_mode == "context_mean":
            scores = context_mean_scores
        elif cfg.score_mode == "recon_cstd":
            wr = float(getattr(cfg, "recon_cstd_weight", 0.8))
            scores = wr * recon_scores + (1.0 - wr) * context_std_scores
        elif cfg.score_mode == "context_combo":
            scores = combo_scores
        elif cfg.score_mode == "recon":
            scores = recon_scores
        else:
            scores = context_std_scores

        struct_scores = None
        if cfg.use_structural_fusion and edge_index_raw is not None:
            struct_scores = compute_structural_scores(
                num_total_nodes,
                edge_index_raw,
                cfg.structural_score_type,
                pagerank_alpha=cfg.pagerank_alpha,
            )
            scores = (1.0 - cfg.structural_fusion_weight) * scores + cfg.structural_fusion_weight * struct_scores

        aux_scores = None
        if getattr(cfg, "enable_aux_fusion", False):
            aux_scores = compute_aux_scores(
                cfg,
                features.detach().cpu().numpy(),
                h_all.detach().cpu().numpy(),
                z_all.detach().cpu().numpy(),
            )
            scores = (1.0 - cfg.aux_fusion_weight) * minmax_scale_numpy(scores) + cfg.aux_fusion_weight * aux_scores

        threshold, preds = select_threshold(scores)

        has_eval_labels = labels_np is not None and np.unique(labels_np).size > 1
        print("=" * 40)
        if has_eval_labels:
            auc, ap = evaluate(labels_np, scores)
            acc, precision, recall, f1 = evaluate_binary(labels_np, preds)
            recon_auc, recon_ap = evaluate(labels_np, recon_scores)
            cmean_auc, cmean_ap = evaluate(labels_np, context_mean_scores)
            cstd_auc, cstd_ap = evaluate(labels_np, context_std_scores)
            combo_auc, combo_ap = evaluate(labels_np, combo_scores)

            print(f"Metrics  | AUC: {auc:.4f} | AP: {ap:.4f} | mode={cfg.score_mode}")
            print(f"ReconRef | AUC: {recon_auc:.4f} | AP: {recon_ap:.4f}")
            print(f"CMeanRef | AUC: {cmean_auc:.4f} | AP: {cmean_ap:.4f}")
            print(f"CStdRef  | AUC: {cstd_auc:.4f} | AP: {cstd_ap:.4f}")
            print(f"ComboRef | AUC: {combo_auc:.4f} | AP: {combo_ap:.4f}")

            if struct_scores is not None:
                struct_auc, struct_ap = evaluate(labels_np, struct_scores)
                fused_auc, fused_ap = evaluate(labels_np, scores)
                print(f"StructRf | AUC: {struct_auc:.4f} | AP: {struct_ap:.4f}")
                print(f"FusedRef | AUC: {fused_auc:.4f} | AP: {fused_ap:.4f}")
            if aux_scores is not None:
                aux_auc, aux_ap = evaluate(labels_np, aux_scores)
                print(f"AuxRef   | AUC: {aux_auc:.4f} | AP: {aux_ap:.4f} | type={cfg.aux_score_type}")

            print(f"ClsEval  | Acc: {acc:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        else:
            print(f"Metrics  | No labels found, skipped AUC/AP. mode={cfg.score_mode}")

        print(f"Unsuperv | Threshold: {threshold:.4f} | Anomalies Found: {sum(preds)}")
        print("=" * 40)


if __name__ == "__main__":
    main()
