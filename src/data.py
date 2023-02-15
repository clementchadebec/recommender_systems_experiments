import os

import numpy as np
import pandas as pd
import torch
from scipy import sparse


def load_train_data(csv_file, n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


class Dataset():
    def __init__(self, args, data_dir=None):
        self.device = args.device
        self.data = args.data
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        if data_dir:
            DATA_DIR = data_dir + str(self.data)
        else:
            DATA_DIR = '../data/{}'.format(self.data)
        if args.data in {'foursquare', 'gowalla', 'ml25m', 'ml20m', 'ml100k'}:
            unique_sid = list()
            with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'r') as f:
                for line in f:
                    unique_sid.append(line.strip())

            self.n_items = len(unique_sid)
            n_items = self.n_items
            self.train_data = load_train_data(os.path.join(DATA_DIR, 'train.csv'), n_items)
            self.N = self.train_data.shape[0]

            self.vad_data_tr, self.vad_data_te = load_tr_te_data(os.path.join(DATA_DIR, 'validation_tr.csv'),
                                                                 os.path.join(DATA_DIR, 'validation_te.csv'), n_items)
            self.N_vad = self.vad_data_tr.shape[0]
        else:
            raise ModuleNotFoundError

    def next_train_batch(self):
        """
        Training batches will reshuffle every epoch and involve dynamic
        binarization
        """
        idxlist = np.arange(self.N)
        np.random.shuffle(idxlist)
        for bnum, st_idx in enumerate(range(0, self.N, self.train_batch_size)):
            end_idx = min(st_idx + self.train_batch_size, self.N)
            X = self.train_data[idxlist[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            yield X

    def next_val_batch(self):
        """
        Validation batches will be used for ELBO estimates without importance
        sampling (could change)
        """
        idxlist_vad = np.arange(self.N_vad)
        for bnum, st_idx in enumerate(range(0, self.N_vad, self.val_batch_size)):
            end_idx = min(st_idx + self.val_batch_size, self.N_vad)
            X = self.vad_data_tr[idxlist_vad[st_idx:end_idx]]
            X_ = self.vad_data_te[idxlist_vad[st_idx:end_idx]]
            if sparse.isspmatrix(X):
                X = X.toarray()
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            yield X, X_

    def next_test_batch(self):
        """
        Test batches are same as validation but with added binarization
        """
        for test_batch in self.test_dataloader:
            yield test_batch
