import torch


class Config:
    # Basic settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    epochs = 50
    lr = 5e-4
    weight_decay = 5e-4

    # Dataset
    data_path = r'data/Amazon.mat'

    # Sampling and dimensions
    batch_size = 512
    in_dim = None
    proj_dim = 256
    hid_dim = 128
    gcn_layers = 2
    feature_knn_k = 20
    second_view_mode = 'copy'  # copy | knn | union | relation | relation:net_xxx

    # Encoder
    encoder_type = 'graphsage'  # sgcn | graphsage | gat
    sage_aggr = 'mean'          # mean | max | lstm
    enc_dropout = 0.2
    gat_heads = 4
    gat_dropout = 0.2

    # View perturbation
    attr_mask_ratio = 0.3
    edge_mask_ratio = 0.3
    mix_ratio = 0.5

    # Loss
    lambda_dcor = 0.1
    alpha_score = 0.5
    lambda_nce = 0.0
    nce_temp = 0.2

    # Training and scoring
    normalize_features = True
    dcor_max_nodes = 256
    dcor_struct_dim = 16
    context_trials = 20
    score_mode = 'recon_cstd'  # context_combo | context_std | context_mean | recon | recon_cstd
    recon_cstd_weight = 0.7692308
    combo_mean_weight = 0.4
    max_full_infer_nodes = 50000
    infer_batch_size = 1024
    use_structural_fusion = False
    structural_score_type = 'degree_gap'  # inv_pagerank | out_degree | in_degree | degree_gap
    structural_fusion_weight = 1.00
    pagerank_alpha = 0.85

    # Optional unsupervised post-score fusion
    enable_aux_fusion = True
    aux_score_type = 'if_h'  # lof_h | if_h | lof_raw | if_raw | if_z | knn_z
    aux_fusion_weight = 0.35
    aux_lof_k = 35
    aux_knn_k = 30
    aux_if_estimators = 500

    # Optimization
    optimizer = 'adamw'     # adam | adamw
    scheduler = 'onecycle'  # none | onecycle | cosine
    max_lr = 1e-3
    onecycle_div_factor = 10.0
    pct_start = 0.15
    final_div_factor = 100.0
    grad_clip_norm = 1.0
    save_best_by_loss = False
