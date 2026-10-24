import torch
import numpy as np
import scanpy as sc
from scipy import sparse
from torch.utils.data import Dataset, DataLoader


class MetaQDataset(Dataset):
    def __init__(self, x_list, sf_list, raw_list):
        super().__init__()
        self.x_list = x_list
        self.sf_list = sf_list
        self.raw_list = raw_list

        self.cell_num = self.x_list[0].shape[0]
        self.omics_num = len(self.x_list)

        for i in range(self.omics_num):
            self.x_list[i] = torch.from_numpy(self.x_list[i]).float()
            self.sf_list[i] = torch.from_numpy(self.sf_list[i]).float()
            self.raw_list[i] = torch.from_numpy(self.raw_list[i]).float()

    def __len__(self):
        return int(self.cell_num)

    def __getitem__(self, index):
        x_list = []
        sf_list = []
        raw_list = []
        for i in range(self.omics_num):
            x_list.append(self.x_list[i][index])
            sf_list.append(self.sf_list[i][index])
            raw_list.append(self.raw_list[i][index])
        data = {"x": x_list, "sf": sf_list, "raw": raw_list}
        return data


def preprocess(adata, data_type):
    if isinstance(adata.X, sparse.csr_matrix) or isinstance(adata.X, sparse.csc_matrix):
        adata.X = adata.X.toarray()
    raw = adata.X.copy()
    """
    if data_type == "RNA":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sf = np.array((raw.sum(axis=1) / 1e4).tolist()).reshape(-1, 1)
        sc.pp.log1p(adata)
        adata_ = adata.copy()
        if adata.shape[1] < 5000:
            sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        else:
            sc.pp.highly_variable_genes(adata)
        hvg_index = adata.var["highly_variable"].values
        raw = raw[:, hvg_index]
        adata = adata[:, hvg_index]
    """
    if data_type == "RNA":
        # 归一化到每个细胞总表达量为1e4
        sc.pp.normalize_total(adata, target_sum=1e4)
        sf = np.array((raw.sum(axis=1) / 1e4).tolist()).reshape(-1, 1)

        # log1p 转换
        sc.pp.log1p(adata)
        adata_ = adata.copy()

        # 筛选高变基因
        #if adata.shape[1] < 5000:
            #sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        #else:
            #sc.pp.highly_variable_genes(adata)

        #hvg_index = adata.var["highly_variable"].values
        #raw = raw[:, hvg_index]
        #adata = adata[:, hvg_index]


    elif data_type == "ADT":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sf = np.array((raw.sum(axis=1) / 1e4).tolist()).reshape(-1, 1)
        sc.pp.log1p(adata)
        adata_ = adata.copy()
    elif data_type == "ATAC":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sf = np.array((raw.sum(axis=1) / 1e4).tolist()).reshape(-1, 1)
        sc.pp.log1p(adata)
        adata_ = adata.copy()
        sc.pp.highly_variable_genes(adata, n_top_genes=30000)
        hvg_index = adata.var["highly_variable"].values
        raw = raw[:, hvg_index]
        adata = adata[:, hvg_index]

    sc.pp.scale(adata, max_value=10)
    x = adata.X

    return x, sf, raw, adata_

import pandas as pd
import anndata as ad

def load_data(df,metacell_num,batch_size):
    print("=======Loading and Preprocessing Data=======")



    x_list = []
    sf_list = []
    raw_list = []
    adata_list = []


    data_type = "RNA"


    
    #df = df.iloc[:, 1:]  # 去掉第一列索引
    adata = ad.AnnData(df)
    x, sf, raw, adata = preprocess(adata, data_type)

    x_list.append(x)
    sf_list.append(sf)
    raw_list.append(raw)
    adata_list.append(adata)

    print("loaded with shape", list(x.shape))

    dataset = MetaQDataset(x_list, sf_list, raw_list)
    if metacell_num > 1000 and batch_size <= 512:
        batch_size = 4096
    dataloader_train = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    dataloader_eval = DataLoader(
        dataset=dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    input_dims = [x.shape[1] for x in x_list]

    return adata_list, dataloader_train, dataloader_eval, input_dims


def compute_metacell(adata, meta_ids, type_key="celltype"):
    meta_ids = meta_ids.astype(int)
    non_empty_metacell = np.zeros(meta_ids.max() + 1).astype(bool)
    non_empty_metacell[np.unique(meta_ids)] = True

    data = adata.X
    data_meta = np.stack(
        [data[meta_ids == i].mean(axis=0) for i in range(meta_ids.max() + 1)]
    )
    data_meta = data_meta[non_empty_metacell]


    #metacell_adata = sc.AnnData(data_meta)
    metacell_adata = sc.AnnData(data_meta, var=pd.DataFrame(index=adata.var_names.copy()))

    if type_key in adata.obs_keys():
        type_int = torch.from_numpy(adata.obs[type_key].cat.codes.values).long()
        type_map = {
            i: adata.obs[type_key].cat.categories[i]
            for i in range(type_int.max() + 1)
        }
        type_one_hot = torch.zeros(type_int.shape[0], type_int.max() + 1)
        type_one_hot.scatter_(1, type_int.unsqueeze(1), 1)
        type_meta = (
            torch.stack(
                [
                    type_one_hot[meta_ids == i].mean(dim=0)
                    for i in range(meta_ids.max() + 1)
                ]
            )
            .argmax(dim=1)
            .numpy()
        )
        type_meta = np.array([type_map[i] for i in type_meta])
        type_meta = type_meta[non_empty_metacell]
        metacell_adata.obs[type_key] = type_meta

    return metacell_adata
