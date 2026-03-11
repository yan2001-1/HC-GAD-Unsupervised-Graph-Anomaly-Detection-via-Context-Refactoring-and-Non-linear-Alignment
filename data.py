import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch_geometric.utils import subgraph


def build_feature_knn_edge_index(features, k=20):
    """Build symmetric kNN graph from node features (cosine similarity)."""
    x = np.asarray(features, dtype=np.float32)
    n = x.shape[0]
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)

    k = int(max(1, min(k, n - 1)))
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sim = x_norm @ x_norm.T
    np.fill_diagonal(sim, -np.inf)

    nbr = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    src = np.repeat(np.arange(n), k)
    dst = nbr.reshape(-1)

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    adj_knn = sp.csr_matrix((np.ones_like(rows), (rows, cols)), shape=(n, n))
    adj_knn.setdiag(0)
    adj_knn.eliminate_zeros()
    coo = adj_knn.tocoo()
    return torch.LongTensor(np.vstack((coo.row, coo.col)))


def _to_csr_binary(mat):
    if sp.issparse(mat):
        m = mat.tocsr()
    else:
        m = sp.csr_matrix(mat)
    m = m.copy()
    if m.nnz > 0:
        m.data = np.ones_like(m.data)
    m.setdiag(0)
    m.eliminate_zeros()
    return m


def _coo_to_edge_index(m):
    coo = m.tocoo()
    return torch.LongTensor(np.vstack((coo.row, coo.col)))


def _extract_features(data, path):
    if 'Attributes' in data:
        feats = data['Attributes']
    elif 'X' in data:
        feats = data['X']
    elif 'feature' in data:
        feats = data['feature']
    elif 'features' in data:
        feats = data['features']
    else:
        raise KeyError(f"No feature matrix found in {path}. keys={list(data.keys())}")

    if sp.issparse(feats):
        feats = feats.todense()
    return torch.FloatTensor(np.asarray(feats))


def _extract_labels_or_none(data):
    labels = None
    for k in ('Label', 'label', 'labels', 'gnd', 'y'):
        if k in data:
            labels = data[k]
            break

    if labels is None:
        return None

    labels = np.asarray(labels).reshape(-1)
    labels = (labels > 0).astype(np.int64)
    return torch.from_numpy(labels).long()


def _extract_rel_mats_from_edge_object(edge_obj):
    rel_mats = []
    flat = np.asarray(edge_obj, dtype=object).reshape(-1)
    for obj in flat:
        if obj is None:
            continue
        try:
            m = _to_csr_binary(obj)
            rel_mats.append(m)
        except Exception:
            continue
    return rel_mats


def load_multiplex_data(path, return_raw_edge_index=False, feature_knn_k=20, second_view_mode='copy'):
    """Load multiplex graph data from multiple .mat layouts.

    Returns:
        features: FloatTensor [N, F]
        edge_indices: list[edge_index_view1, edge_index_view2]
        labels: LongTensor [N] or None when labels are absent
        edge_index_raw (optional): directed raw edge index used by structural priors
    """
    if not os.path.exists(path):
        alt_path = os.path.join('data', path)
        if os.path.exists(alt_path):
            path = alt_path

    data = sio.loadmat(path)
    features = _extract_features(data, path)
    labels = _extract_labels_or_none(data)

    # Alibaba-like format: edge is (1, R) object array of sparse relation matrices.
    if 'edge' in data and isinstance(data['edge'], np.ndarray) and data['edge'].dtype == object:
        rel_mats = _extract_rel_mats_from_edge_object(data['edge'])
        if len(rel_mats) == 0:
            raise KeyError(f"'edge' exists but no valid sparse relation matrices were found in {path}.")

        # Directed union as raw structural graph.
        adj_raw = rel_mats[0].copy()
        for m in rel_mats[1:]:
            adj_raw = adj_raw + m
        if adj_raw.nnz > 0:
            adj_raw.data = np.ones_like(adj_raw.data)
        adj_raw.eliminate_zeros()
        edge_index_raw = _coo_to_edge_index(adj_raw)

        rel_undirected = []
        for m in rel_mats:
            mu = m.maximum(m.T)
            mu.setdiag(0)
            mu.eliminate_zeros()
            if mu.nnz > 0:
                mu.data = np.ones_like(mu.data)
            rel_undirected.append(mu)

        edge_index_1 = _coo_to_edge_index(rel_undirected[0])
        mode = str(second_view_mode).lower()
        if mode == 'knn':
            edge_index_2 = build_feature_knn_edge_index(features.numpy(), k=feature_knn_k)
        elif mode == 'union':
            edge_index_knn = build_feature_knn_edge_index(features.numpy(), k=feature_knn_k)
            adj_union = sp.csr_matrix(
                (np.ones(edge_index_1.size(1), dtype=np.uint8), (edge_index_1[0].numpy(), edge_index_1[1].numpy())),
                shape=(features.size(0), features.size(0)),
            )
            adj_union += sp.csr_matrix(
                (np.ones(edge_index_knn.size(1), dtype=np.uint8), (edge_index_knn[0].numpy(), edge_index_knn[1].numpy())),
                shape=(features.size(0), features.size(0)),
            )
            if adj_union.nnz > 0:
                adj_union.data = np.ones_like(adj_union.data)
            adj_union.eliminate_zeros()
            edge_index_2 = _coo_to_edge_index(adj_union)
        elif len(rel_undirected) >= 2:
            edge_index_2 = _coo_to_edge_index(rel_undirected[1])
        else:
            edge_index_2 = edge_index_1.clone()

        edge_indices = [edge_index_1, edge_index_2]
        if return_raw_edge_index:
            return features, edge_indices, labels, edge_index_raw
        return features, edge_indices, labels

    # Generic single-adjacency formats.
    if 'Network' in data:
        adj = data['Network']
    elif 'homo' in data:
        adj = data['homo']
    elif 'A' in data:
        adj = data['A']
    elif 'adj' in data:
        adj = data['adj']
    elif any(str(k).startswith('net_') for k in data.keys()):
        net_keys = sorted([str(k) for k in data.keys() if str(k).startswith('net_')])
        adj = data[net_keys[0]]
    else:
        raise KeyError(f"No adjacency found in {path}. keys={list(data.keys())}")

    adj_raw = _to_csr_binary(adj)
    edge_index_raw = _coo_to_edge_index(adj_raw)

    adj_u = adj_raw.maximum(adj_raw.T)
    adj_u.setdiag(0)
    adj_u.eliminate_zeros()
    if adj_u.nnz > 0:
        adj_u.data = np.ones_like(adj_u.data)
    edge_index_1 = _coo_to_edge_index(adj_u)

    mode_full = str(second_view_mode).lower()
    mode = mode_full
    if mode == 'knn':
        edge_index_2 = build_feature_knn_edge_index(features.numpy(), k=feature_knn_k)
    elif mode == 'union':
        edge_index_knn = build_feature_knn_edge_index(features.numpy(), k=feature_knn_k)
        adj_union = sp.csr_matrix(
            (np.ones(edge_index_1.size(1), dtype=np.uint8), (edge_index_1[0].numpy(), edge_index_1[1].numpy())),
            shape=(features.size(0), features.size(0)),
        )
        adj_union += sp.csr_matrix(
            (np.ones(edge_index_knn.size(1), dtype=np.uint8), (edge_index_knn[0].numpy(), edge_index_knn[1].numpy())),
            shape=(features.size(0), features.size(0)),
        )
        if adj_union.nnz > 0:
            adj_union.data = np.ones_like(adj_union.data)
        adj_union.eliminate_zeros()
        edge_index_2 = _coo_to_edge_index(adj_union)
    elif mode.startswith('relation'):
        net_keys = sorted([str(k) for k in data.keys() if str(k).startswith('net_')])
        chosen = None
        if ':' in mode_full:
            chosen = mode_full.split(':', 1)[1]
        elif 'net_rtr' in net_keys:
            chosen = 'net_rtr'
        elif len(net_keys) > 0:
            chosen = net_keys[0]

        if chosen is not None and chosen in data:
            rel2 = _to_csr_binary(data[chosen]).maximum(_to_csr_binary(data[chosen]).T)
            rel2.setdiag(0)
            rel2.eliminate_zeros()
            if rel2.nnz > 0:
                rel2.data = np.ones_like(rel2.data)
            edge_index_2 = _coo_to_edge_index(rel2)
        else:
            edge_index_2 = edge_index_1.clone()
    else:
        edge_index_2 = edge_index_1.clone()

    edge_indices = [edge_index_1, edge_index_2]
    if return_raw_edge_index:
        return features, edge_indices, labels, edge_index_raw
    return features, edge_indices, labels


def random_subgraph_sampling(num_nodes, edge_indices, batch_size):
    """Random node sampling + induced subgraph extraction for each view."""
    device = edge_indices[0].device
    if batch_size >= num_nodes:
        batch_nodes = torch.arange(num_nodes, device=device)
        return batch_nodes, edge_indices

    batch_nodes = torch.randperm(num_nodes, device=device)[:batch_size]

    sub_edge_indices = []
    for ei in edge_indices:
        sub_ei, _ = subgraph(batch_nodes, ei, relabel_nodes=True, num_nodes=num_nodes)
        sub_edge_indices.append(sub_ei)

    return batch_nodes, sub_edge_indices
