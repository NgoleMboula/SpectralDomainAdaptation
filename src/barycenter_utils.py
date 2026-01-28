import numpy as np
import torch
import random
import ot
import matplotlib.pyplot as plt


def unif(n, device='cpu', dtype=torch.float32):
    return torch.ones(n, device=device, dtype=dtype) / n


def init_XB_(Xs, ys):
    Xbar = np.zeros(Xs.shape)
    for y in np.unique(ys):
        idx = np.where(ys==y)[0]
        y_mean = np.mean(Xs[idx], axis=0)
        Xbar[idx, :] = y_mean + np.random.randn(len(idx), Xs.shape[1])
    
    return Xbar, ys

def penalize_y(ys, yt, M, limit_max = np.inf):
    assert M.shape == (ys.shape[0], yt.shape[0]), "Expected cost matrix to have shape ({}, {})"

    _M = M.copy()
    classes = [c for c in np.unique(ys) if c != -1]
    for c in classes:
        idx_s = np.where((ys != c) & (ys != -1))[0]
        idx_t = np.where(yt == c)[0]

        for j in idx_t:
            _M[idx_s, j] = limit_max

    return _M

def barycentric_projection(Xt, coupling):
    transp_Xs = coupling / np.sum(coupling, 1)[:, None]
    transp_Xs[~np.isfinite(transp_Xs)] = 0
    transp_Xs = np.dot(transp_Xs, Xt)
    return transp_Xs

def set_seed(seed = 10):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sample_from_dataset_(raw, domain_names, sample_size=600):
    X_list, y_list, m_list = [], [], []
    domain_id = 1

    for dom in domain_names:
        Xd, yd = raw[dom]
        if isinstance(Xd, torch.Tensor):
            Xd = Xd.numpy()
        if isinstance(yd, torch.Tensor):
            yd = yd.numpy()

        N = len(yd)
        n_sample = min(sample_size, N)

        idx = np.random.choice(N, n_sample, replace=False)

        X_list.append(Xd[idx])
        y_list.append(yd[idx])
        m_list.append(np.ones(n_sample) * domain_id)

        domain_id += 1

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    m = np.concatenate(m_list)

    dataset = np.column_stack([X, y, m])
    domains = np.unique(m).astype(int)
    targets = domains

    return dataset, domains, targets