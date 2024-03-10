import random
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale, minmax_scale
from collections import Counter

random.seed(2023)


def step_subsequence(TS, sublens, step=1):
    """
    Generate subsequences of TS using the sliding window
    """
    stride = max(1, step)
    TS = np.asarray(TS)
    n_rows = ((TS.size - sublens) // stride) + 1
    n = TS.strides[0]
    return np.lib.stride_tricks.as_strided(TS, shape=(n_rows, sublens), strides=(stride * n, n))


def IK_seq_aNNE(X, t, psi):
    """
    Feature mp of Isolation Kernel in $R^d$ domain
    """
    ik_matrix = np.zeros((X.shape[0], int(t)), dtype=int)
    X_size = len(X)

    def compute_dist(_t: int):
        sample_num = psi
        sample_list = [p for p in range(X_size)]
        sample_list = random.sample(sample_list, sample_num)
        sample = X[sample_list, :]
        # compute the distance between the sample subsequence and original subsequences
        # to determine the Voronoi diagram cell centre
        shape2dist = cdist(X, sample, metric='euclidean')
        min_idx_shape2dist = np.argmin(shape2dist, axis=1) + _t * psi
        ik_matrix[:, _t] = min_idx_shape2dist
    _jobs = 5
    Parallel(n_jobs=_jobs, backend="threading")(delayed(compute_dist)(_t) for _t in range(t))

    return ik_matrix


def idks_subsets(TS, step=1, psi=128, normalized=True):
    """
    Kernel Mean Embedding of original TS in each subset
    """
    TS_size, TS_len = len(TS), len(TS[0])
    d = TS_len // 2
    X = []
    for i, ith_TS in enumerate(TS):
        subseq = step_subsequence(ith_TS, d, step)
        if normalized:
            subseq = scale(subseq, axis=1)
        X += list(subseq)
    X = np.array(X)
    # print(f'Dataset shape:{TS.shape}, 处理后:{X.shape}')
    win_num_perTS = X.shape[0] // TS_size
    _t = 50
    if psi > len(X):
        print('PSI ERROR!!!')
        return []

    # feature mean map of Isolation Distribution Kernel
    idk_fm = np.zeros((TS_size, _t * psi), dtype=float)
    fm_matrix = IK_seq_aNNE(X, t=_t, psi=psi).reshape((TS_size, -1))
    counter_list = [Counter(it) for it in fm_matrix]
    key_list = [list(cou.keys()) for cou in counter_list]
    val_list = [list(cou.values()) for cou in counter_list]
    for i in range(TS_size):
        idk_fm[i][key_list[i]] += val_list[i]
    idk_fm /= win_num_perTS
    tmp = np.linalg.norm(idk_fm, axis=1)
    idk_fm /= tmp[:, None]
    return idk_fm


def idks_sliding(TS, win_len, step=2, psi=128, normalized=True):
    TS_size, TS_len = len(TS), len(TS[0])
    d = win_len
    X = []
    for i, ith_TS in enumerate(TS):
        subseq = step_subsequence(ith_TS, d, step)
        if normalized:
            subseq = scale(subseq, axis=1)
        X += list(subseq)
    X = np.array(X)

    win_num_perTS = X.shape[0] // TS_size
    _t = 50
    if psi > len(X):
        print('PSI ERROR!!!')
        return []

    idk_fm = np.zeros((TS_size, _t * psi), dtype=float)
    fm_matrix = IK_seq_aNNE(X, t=_t, psi=psi).reshape((TS_size, -1))
    counter_list = [Counter(it) for it in fm_matrix]
    key_list = [list(cou.keys()) for cou in counter_list]
    val_list = [list(cou.values()) for cou in counter_list]
    for i in range(TS_size):
        idk_fm[i][key_list[i]] += val_list[i]
    idk_fm /= win_num_perTS
    tmp = np.linalg.norm(idk_fm, axis=1)
    idk_fm /= tmp[:, None]
    return idk_fm
