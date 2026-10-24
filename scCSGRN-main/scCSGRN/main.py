
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import argparse
import utils_GRN, utils_gz
from model import TransformerE
import warnings
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)
from timeit import default_timer as timer
from utils_GRN import compute_total_loss

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='scCSGRN', help='model name')
parser.add_argument('--iteration', type=int, default=5, help='the number of training and test')
parser.add_argument('--batch_sizes', type=int, default='256', help='batch_sizes')
parser.add_argument('--out_dim_GT', type=int, default='64', help='the output of GT model')
parser.add_argument('--hidden_dim_GT', type=int, default='64', help='the hidden of GT model')
parser.add_argument('--num_heads_GT', type=int, default='4', help='the heads of GT model')
parser.add_argument('--input_dim_exp', type=int, default='200', help='the input of expression Transformer model')
parser.add_argument('--d_models', type=int, default='200', help='the input of expression Transformer model')
parser.add_argument('--lambda_exp_recon', type=float, default=0.1, help='weight of VAE expression reconstruction loss')
parser.add_argument('--beta_kl', type=float, default=0.001, help='weight of VAE KL-divergence loss')
parser.add_argument('--lambda_gt_recon', type=float, default=0.1, help='weight of GT node-feature reconstruction loss')

args = parser.parse_args()
save_folder = 'scCSGRN_GRN'
processed_data = f'scCSGRN_data'
#'BT1303'：Hoc,'BT1307':Hoc(na),'scrEXT009':Hcc,'sc5rJUQ045':Hbc,'E-MTAB-14691':Mc,'Mtab-11821':Mch
#dataset_names = ['BT1303','BT1307','scrEXT009','sc5rJUQ045','E-MTAB-14691','Mtab-11821']

dataset_names = ['Mtab-11821']

GRN_types=["celltype_grnboost2_GRN"]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Rank_nums=[500]

file_nme='scCSGRN_GRN'

for Rank_num in Rank_nums:
    if Rank_num==500:
        epochs=60
    else:
        epochs=40 
    for dataset_name in dataset_names:
        network_types = []
        if dataset_name == 'BT1303':
            network_types = ['Chondrocytes','Endothelial_cells','Fibroblasts','Macrophage','Monocyte','Smooth_muscle_cells','T_cells'] 

        elif dataset_name=='BT1307':
            network_types = ['Fibroblasts','Epithelial_cells','Endothelial_cells','Monocyte','T_cells','Smooth_muscle_cells','Tissue_stem_cells']

        elif dataset_name == 'scrEXT009':
            network_types = [ 'B_cell','CMP', 'Endothelial_cells','Epithelial_cells','Fibroblasts','Monocyte','T_cells','Tissue_stem_cells']

        elif dataset_name == 'sc5rJUQ045' :
            network_types = ['Chondrocytes','Monocyte','MSC','T_cells','Tissue_stem_cells']

        elif dataset_name == 'E-MTAB-14691' :
            network_types = ['Astrocytes','Fibroblasts','Macrophages','Neurons']
            
        elif dataset_name == 'Mtab-11821' :
            network_types = ['B_cells']#'B_cells','Endothelial_cells','Monocytes','NK_cells'
                        
        else:
            print("network type error")


        print(dataset_name + ': ')
        print(network_types)

        for nt in range(len(network_types)):
            network_type = network_types[nt]
            
            
            for st in range(len(GRN_types)):

                GRN_type=GRN_types[st]

                data_path = '..\\data_preprocess\\' + processed_data + '\\' + dataset_name +'\\' + network_type + '\\'+ GRN_type + '\\Top' + str(
                    Rank_num) + '\\'

                log_dir = "log\\" + "\\" + str(Rank_num) + "\\"  + dataset_name + '\\'+ network_type + "\\"+ GRN_type + "\\"
                if (not os.path.isdir(log_dir)):
                    os.makedirs(log_dir)

                matrix_data = np.load(data_path + 'matrix.npy')
                label_data = np.load(data_path + 'label.npy')
                gene_pair_tf = np.load(data_path + 'pair_list_id_tf.npy')
                gene_pair_target = np.load(data_path + 'pair_list_id_target.npy')
                exp_file = data_path + 'Final_expression.csv'
                gene_list_file = data_path + 'Final_gene_list.csv'

                # save path
                save_path = save_folder  + '\\' + dataset_name  + '\\'+ network_type+'\\' + GRN_type  + '\\Top' + str(Rank_num) + '\\'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                network_dict = {}
                all_network_dict = {}
                AUROCs = []
                AUPRs = []
                SPEs = []
                Recalls = []
                Precisions = []
                F1s = []
                MCCs = []
                ACCs = []
                times=[]
                for ki in range(args.iteration):
                    bigan=timer()
                    print('\n'+'-----------------the ' + str(ki + 1) + 'th iteration-------'+ dataset_name +'-------------' + network_type + '-------'+ GRN_type +'------'+ f"{Rank_num}" +'--------')

                    x_train, x_t, y_train, y_t, gene_pair_tf_train, gene_pair_tf_t, gene_pair_target_train, gene_pair_target_t = train_test_split(matrix_data, label_data, gene_pair_tf, gene_pair_target, test_size=0.4, stratify=label_data)
                    x_val, x_test, y_val, y_test, gene_pair_tf_val, gene_pair_tf_test, gene_pair_target_val, gene_pair_target_test = train_test_split(x_t, y_t, gene_pair_tf_t, gene_pair_target_t, test_size=0.5, stratify=y_t)

                    # transform train set into torch_geometric.data.Data
                    train_og_pos, train_og_neg, train_og_pos_T, train_og_neg_T = utils_gz.get_pos_neg_T_2(gene_pair_tf_train,gene_pair_target_train, y_train,
                                                                                                        exp_file,
                                                                                                        gene_list_file)

                    train_g_pos = utils_GRN.transform_savebinI(train_og_pos, train_og_pos, train_og_pos_T,
                                                            katz_alpha=0.02)

                    val_og_pos, val_og_neg, val_og_pos_T, val_og_neg_T = utils_gz.get_pos_neg_T_2(
                        gene_pair_tf_val, gene_pair_target_val, y_val,
                        exp_file,
                        gene_list_file)

                    val_g_pos = utils_GRN.transform_savebinI(val_og_pos, val_og_pos, val_og_pos_T,
                                                            katz_alpha=0.02)

                    test_og_pos, test_og_neg, test_og_pos_T, test_og_neg_T = utils_gz.get_pos_neg_T_2(
                        gene_pair_tf_test, gene_pair_target_test, y_test,
                        exp_file,
                        gene_list_file)
                    test_g_pos = utils_GRN.transform_savebinI(test_og_pos, test_og_pos, test_og_pos_T,
                                                            katz_alpha=0.02)


                    X_trainloader, y_trainloader, gene_pair_tf_trainloader, gene_pair_target_trainloader = utils_GRN.numpy2loader(x_train, y_train, gene_pair_tf_train, gene_pair_target_train, args.batch_sizes)
                    X_valloader, y_valloader, gene_pair_tf_valloader, gene_pair_target_valloader = utils_GRN.numpy2loader(x_val, y_val, gene_pair_tf_val, gene_pair_target_val, args.batch_sizes)
                    X_testloader, y_testloader, gene_pair_tf_testloader, gene_pair_target_testloader = utils_GRN.numpy2loader(x_test, y_test, gene_pair_tf_test, gene_pair_target_test, args.batch_sizes)

                    X_trainList = utils_GRN.loaderToList(X_trainloader)
                    y_trainList = utils_GRN.loaderToList(y_trainloader)
                    gene_pair_tf_trainList = utils_GRN.loaderToList(gene_pair_tf_trainloader)
                    gene_pair_target_trainList = utils_GRN.loaderToList(gene_pair_target_trainloader)

                    X_valList = utils_GRN.loaderToList(X_valloader)
                    y_valList = utils_GRN.loaderToList(y_valloader)
                    gene_pair_tf_valList = utils_GRN.loaderToList(gene_pair_tf_valloader)
                    gene_pair_target_valList = utils_GRN.loaderToList(gene_pair_target_valloader)

                    X_testList = utils_GRN.loaderToList(X_testloader)
                    #print("X_testList:",X_testList)
                    y_testList = utils_GRN.loaderToList(y_testloader)
                    #print("y_testList:",y_testList)
                    gene_pair_tf_testList = utils_GRN.loaderToList(gene_pair_tf_testloader)
                    #print("gene_pair_tf_testList:",gene_pair_tf_testList)
                    gene_pair_target_testList = utils_GRN.loaderToList(gene_pair_target_testloader)
                    #print("gene_pair_target_valList:",gene_pair_target_valList)
                    in_dim_GT = train_g_pos.ndata['x'].shape[1]
                    I_dim = train_g_pos.ndata['I'].shape[1]
                    model = TransformerE.scCSGRN(in_dim_GT, args.out_dim_GT, args.hidden_dim_GT, args.num_heads_GT, I_dim, args.input_dim_exp, num_gtlayers=4, d_model=args.d_models, num_classes=2).to(device)

                    criterion = nn.CrossEntropyLoss()

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

                    for epoch in range(epochs):
                        model.train()

                        train_loss = []
                        train_cls_loss = []
                        train_exp_recon_loss = []
                        train_kl_loss = []
                        train_gt_recon_loss = []
                        train_accs = []

                        for j in range(0, len(X_trainList)):
                            data = X_trainList[j].to(device)
                            labels = y_trainList[j].to(device).long()
                            all_tf_train, all_target_train, train_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels.cpu(), gene_pair_tf_trainList[j], gene_pair_target_trainList[j], train_g_pos)




                            logits, loss_terms = model(
                                train_g_batch.to(device), data, all_tf_train, all_target_train, device,
                                return_loss_terms=True)
                            
                            loss, cls_loss, exp_recon_loss, kl_loss, gt_recon_loss = compute_total_loss(
                                logits, labels, loss_terms, data, criterion, args)
                            optimizer.zero_grad()
                            loss.backward()
                            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                            optimizer.step()

                            acc = (logits.argmax(dim=-1) == labels).float().mean()
                            train_loss.append(loss.item())
                            train_cls_loss.append(cls_loss.item())
                            train_exp_recon_loss.append(exp_recon_loss.item())
                            train_kl_loss.append(kl_loss.item())
                            train_gt_recon_loss.append(gt_recon_loss.item())
                            train_accs.append(acc)
                        train_loss = sum(train_loss) / len(train_loss)
                        train_cls_loss = sum(train_cls_loss) / len(train_cls_loss)
                        train_exp_recon_loss = sum(train_exp_recon_loss) / len(train_exp_recon_loss)
                        train_kl_loss = sum(train_kl_loss) / len(train_kl_loss)
                        train_gt_recon_loss = sum(train_gt_recon_loss) / len(train_gt_recon_loss)
                        train_acc = sum(train_accs) / len(train_accs)

                        print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, cls = {train_cls_loss:.5f}, exp_rec = {train_exp_recon_loss:.5f}, kl = {train_kl_loss:.5f}, gt_rec = {train_gt_recon_loss:.5f}, acc = {train_acc:.5f}")

                        model.eval()
                        predictions = []
                        labelss = []
                        y_val_label = []
                        y_val_predict = []
                        valid_loss = []
                        valid_accs = []
                        for k in range(0, len(X_valList)):
                            val_data = X_valList[k].to(device)
                            labels = y_valList[k].to(device).long()
                            all_tf_val, all_target_val, val_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels.cpu(), gene_pair_tf_valList[k], gene_pair_target_valList[k], val_g_pos)


                            with torch.no_grad():
                                # logits = model(val_data)
                                logits = model(val_g_batch.to(device), val_data, all_tf_val, all_target_val, device)

                            loss = criterion(logits, labels)
                            valid_loss.append(loss.item())

                            acc = (logits.argmax(dim=-1) == labels).float().mean()
                            valid_accs.append(acc)

                            predt = F.softmax(logits, dim=1)
                            labelss.extend(labels.cpu().numpy().tolist())
                            y_val_label.extend(labels.cpu().numpy())

                            temps = predt.cpu().numpy().tolist()
                            for i in temps:
                                t = i[1]
                                y_val_predict.append(t)
                            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                        valid_loss = sum(valid_loss) / len(valid_loss)
                        valid_acc = sum(valid_accs) / len(valid_accs)
                        AUC_val, AUPR_val, ACC_val, F1_val, SPE_val, MCC_val, Precision_val, Recall_val = utils_gz.metric_scores(y_val_label, y_val_predict, th=0.5)

                        print(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.4f}, auc = {AUC_val:.4f}, aupr = {AUPR_val:.4f}")


                    y_test_label = []
                    y_predict = []
                    predictions = []
                    labelss = []
                    model.eval()
                    for k in range(0, len(X_testList)):
                        test_data = X_testList[k].to(device)
                        labels = y_testList[k].to(device).long()

                        all_tf_test, all_target_test, test_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels.cpu(), gene_pair_tf_testList[k], gene_pair_target_testList[k], test_g_pos)

                        with torch.no_grad():
                            # logits = model(data)
                            logits = model(test_g_batch.to(device), test_data, all_tf_test, all_target_test,
                                        device)
                        predt = F.softmax(logits, dim=1)
                        labelss.extend(labels.cpu().numpy().tolist())
                        y_test_label.extend(labels.cpu().numpy())

                        # temps = logits.cpu().numpy().tolist()
                        temps = predt.cpu().numpy().tolist()
                        for i in temps:
                            t = i[1]
                            y_predict.append(t)
                        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())


                    AUC, AUPR, ACC, F1, SPE, MCC, Precision, Recall = utils_gz.metric_scores(y_test_label, y_predict, th=0.5)
                    end=timer()
                    bessa=end-bigan                
                    print(f"[Test]--------------------------------------------auc = {AUC:.4f}, aupr = {AUPR:.4f}")

                    AUROCs.append(AUC)
                    AUPRs.append(AUPR)
                    ACCs.append(ACC)
                    F1s.append(F1)
                    SPEs.append(SPE)
                    MCCs.append(MCC)
                    Precisions.append(Precision)
                    Recalls.append(Recall)
                    times.append(bessa)
                all_network_dict["AUROC"] = AUROCs
                all_network_dict["AUPR"] = AUPRs
                all_network_dict["Accuracy"] = ACCs
                all_network_dict["F1"] = F1s
                all_network_dict["SPE"] = SPEs
                all_network_dict["MCC"] = MCCs
                all_network_dict["Precision"] = Precisions
                all_network_dict["Recall"] = Recalls
                all_network_dict["times"] = times

                filename = open(save_path + f'{file_nme}.csv', 'w')
                for k, v in all_network_dict.items():
                    filename.write(k + ':' + str(v))
                    filename.write('\n')
                filename.close()



