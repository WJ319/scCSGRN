import warnings

warnings.filterwarnings("ignore")

import os
import torch
import random
import argparse
import matplotlib
import numpy as np
import scanpy as sc
import seaborn as sns
from model import MetaQ
from matplotlib import rcParams
from alive_progress import alive_bar
from engine import train_one_epoch, warm_one_epoch, inference
from data_utils import load_data, compute_metacell

import pandas as pd





def main(df,save_path,metacell_num,train_epoch=200,batch_size=32,codebook_init="Random"):
    device = torch.device('cpu')

    adata_list, dataloader_train, dataloader_eval, input_dims = load_data(df,metacell_num,batch_size)
    omics_num = len(adata_list)

    print("Target metacell number:", metacell_num)

    net = MetaQ(
        input_dims=input_dims,
        data_types=["RNA"],
        entry_num=metacell_num,
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

    print("======= Training Start =======")

    with alive_bar(train_epoch, enrich_print=False) as bar:
        loss_rec_his = loss_vq_his = 1e7
        stable_epochs = 0
        if codebook_init == "Random":
            warm_epochs = 0
        else:
            # For Kmeans and Geometric initialization
            warm_epochs = min(50, int(train_epoch * 0.2))
        for epoch in range(train_epoch):
            bar()
            if epoch < warm_epochs:
                warm_one_epoch(
                    model=net,
                    data_types=["RNA"],
                    dataloader=dataloader_train,
                    optimizer=optimizer,
                    epoch=epoch,
                    device=device,
                )
            elif epoch == warm_epochs:
                embeds, ids, _, _, _ = inference(
                    model=net,
                    data_types=["RNA"],
                    data_loader=dataloader_eval,
                    device=device,
                )
                net.quantizer.init_codebook(embeds, method=codebook_init)
                if omics_num == 1:
                    net.copy_decoder_q()
            else:
                loss_rec, loss_vq = train_one_epoch(
                    model=net,
                    data_types=["RNA"],
                    dataloader=dataloader_train,
                    optimizer=optimizer,
                    epoch=epoch,
                    device=device,
                )
                converge = (abs(loss_vq_his - loss_vq) <= 1e-5) and (
                    abs(loss_rec_his - loss_rec) <= 1e-5
                )
                if converge:
                    stable_epochs += 1
                    if stable_epochs >= 10:
                        print("Early Stopping.")
                        break
                else:
                    stable_epochs = 0
                    loss_rec_his = loss_rec
                    loss_vq_his = loss_vq

    print("======= Training Done =======")

    if not os.path.exists("./save/"):
        os.makedirs("./save/")

    for i in range(omics_num):

        adata = adata_list[i]
        metacell_adata = compute_metacell(adata, ids)

        # === 保存为 CSV 格式 ===
        # metacell_adata.X 是 numpy 数组，行为元细胞，列为基因
        
        df = pd.DataFrame(
            metacell_adata.X,
            #index=metacell_adata.obs_names,
            columns=metacell_adata.var_names
        )
        df.to_csv(save_path, index=True)

        print( " metacell matrix saved as CSV at:", save_path)
        
if __name__ == "__main__":

    # Randomization
    random_seed=1
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)

    torch.cuda.manual_seed_all(random_seed)

    

    main()



