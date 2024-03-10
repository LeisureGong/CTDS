import argparse
import numpy as np
from utils import load_ucr
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics.cluster import rand_score as ri_score
from sklearn.cluster import KMeans
from idks import idks_subsets, idks_sliding


class DataWrapper(object):

    def __init__(self, dataset_name):
        self.dataset = dataset_name
        X_train, y_train, X_test, y_test = load_ucr(self.dataset)
        X_train, X_test = np.squeeze(X_train), np.squeeze(X_test)
        self.TS = np.concatenate((X_train, X_test), axis=0)
        self.TSsize = len(X_train) + len(X_test)
        self.TSlen = X_train.shape[1]
        self.classes = len(np.unique(y_train))
        self.labels = np.hstack((y_train, y_test)).reshape(-1)
        print(f'Dataset: {self.dataset}, Shape: {self.TS.shape}')

    def get_subsets(self, sLen):
        """
        start timestamp and end timestamp of each subset
        """
        tmp_lens = self.TSlen
        osp_list = []
        osp_idx = 0
        while tmp_lens > 0:
            l = osp_idx * sLen * 2
            if 0 < tmp_lens <= sLen * 2:
                r = osp_idx * sLen * 2 + tmp_lens
                osp_list.append([l, r])
                break
            elif sLen * 2 < tmp_lens <= sLen * 4:
                # two equal length segments
                left_lens = tmp_lens // 2
                left_l = l
                left_r = left_l + left_lens
                right_l = left_r
                right_r = left_l + tmp_lens
                osp_list.append([left_l, left_r])
                osp_list.append([right_l, right_r])
                break
            else:
                r = osp_idx * sLen * 2 + sLen * 2
                osp_list.append([l, r])
                osp_idx += 1
                tmp_lens -= sLen * 2
        print(f'start timestamp and end timestamp in each subset:{osp_list}')
        return osp_list

    def con_idks_subsets(self, subset_list, psi, normalized=True):
        """
        concatenate feature mean map of each subset
        """
        raw_data = np.asarray(self.TS)
        all_subseries = list()
        for i in range(len(subset_list)):
            left, right = subset_list[i][0], subset_list[i][1]
            ith_subset = raw_data[:, left:right]
            all_subseries.append(ith_subset)

        step = 1
        all_subseries_fm = list()
        for i in range(len(all_subseries)):
            ith_idk_fm = idks_subsets(all_subseries[i], step=step, psi=psi)
            all_subseries_fm.append(ith_idk_fm)
        all_subseries_fm = np.concatenate(all_subseries_fm, axis=1)
        return all_subseries_fm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='FaceFour', help='The dataset name')
    parser.add_argument('--type', type=int, default=0, help="0: dataset length > 150; 1: dataset length <= 150")
    parser.add_argument('--psi', type=int, default=128, help="sample size of isolation kernel")
    parser.add_argument('--d', type=int, default=60, help="subsequence length")
    args = parser.parse_args()
    dataset = args.dataset
    dw = DataWrapper(dataset)
    dataset_type = int(args.type)
    _psi = int(args.psi)

    # For type 0, subset length $m$ is equal to 2*sLen
    # For type 1, sLen is the sliding window length
    sLen = int(args.d)

    five_nmi_list = []
    five_ri_list = []
    for itr in range(5):
        if dataset_type == 1:
            # feature mean map of each time series in dataset
            all_fm = idks_sliding(dw.TS, win_len=sLen, step=1, psi=_psi)
        else:
            # get start and end positions list of disjoint smaller subsets
            osp_list = dw.get_subsets(sLen)
            # the concatenated feature mean map of each subset
            all_fm = dw.con_idks_subsets(osp_list, psi=_psi)
        cluster = KMeans(n_clusters=dw.classes)
        cluster.fit(all_fm)
        kmeans_labels = cluster.predict(all_fm)
        kmeans_nmi = nmi_score(dw.labels, kmeans_labels)
        kmeans_ri = ri_score(dw.labels, kmeans_labels)
        print(f'{dataset}: nmi:{kmeans_nmi:.4f} ri:{kmeans_ri:.4f}')
        five_nmi_list.append(kmeans_nmi)
        five_ri_list.append(kmeans_ri)
        del all_fm
    nmi_mean, nmi_std = np.mean(five_nmi_list), np.std(five_nmi_list)
    ri_mean, ri_std = np.mean(five_ri_list), np.std(five_ri_list)
    print(f'{dataset}:  psi:{_psi}, nmi:{nmi_mean:.4f} {nmi_std:.4f}, ri:{ri_mean:.4f} {ri_std:.4f}')
