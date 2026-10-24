
import numpy as np
import pandas as pd
import csv
from numpy import *
import os
from utils_data import preprocess_scCSGRN


Ranknums = [500]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_source = os.path.join(BASE_DIR, 'scCSGRN_data')

dataset_names = ['BT1303','BT1307','scrEXT009','sc5rJUQ045','E-MTAB-14691','Mtab-11821']#'BT1303','BT1307','scrEXT009','sc5rJUQ045','E-MTAB-14691','Mtab-11821'
GRN_types=["celltype_grnboost2_GRN"]
for dataset_name in dataset_names:
    network_types = []
    if dataset_name == 'BT1303':
        network_types = ['Chondrocytes','Endothelial_cells','Fibroblasts','Macrophage','Monocyte','Smooth_muscle_cells','T_cells'] #'Chondrocytes','Endothelial_cells','Fibroblasts','Macrophage','Monocyte','Smooth_muscle_cells','T_cells'
        meta_cells=[300,150,1000,60,500,300,400]
    elif dataset_name=='BT1307':
        network_types = ['Fibroblasts','Epithelial_cells','Endothelial_cells','Monocyte','T_cells','Smooth_muscle_cells','Tissue_stem_cells']
        meta_cells=[400,100,100,400,400,100,600]
    elif dataset_name == 'scrEXT009' :
        network_types = ['B_cell', 'CMP', 'Endothelial_cells','Epithelial_cells','Fibroblasts','Monocyte','T_cells','Tissue_stem_cells']#'B_cell', 'CMP', 'Endothelial_cells','Epithelial_cells','Fibroblasts','Monocyte','T_cells','Tissue_stem_cells'
        meta_cells=[200,40,100,500,100,150,400,100]
    elif dataset_name == 'scrSOL004' :
        network_types = ['B_cell', 'Endothelial_cells','Fibroblasts','Monocyte','Neurons','T_cells','Tissue_stem_cells']#'B_cell', 'Endothelial_cells','Fibroblasts','Monocyte','Neurons','T_cells','Tissue_stem_cells'
        meta_cells=[60,100,200,200,700,200,400]
    elif dataset_name == 'sc5rJUQ045' :
        network_types = ['Chondrocytes','Monocyte','MSC','T_cells','Tissue_stem_cells']
        meta_cells=[400,200,100,80,60]
    elif dataset_name == 'E-MTAB-14691' :
        network_types = ['Astrocytes','Fibroblasts','Macrophages','Neurons']
        meta_cells=[200,400,40,200]
    elif dataset_name == 'Mtab-11821' :
        network_types = ['B_cells','Endothelial_cells','Monocytes','NK_cells']
        meta_cells=[500,20,800,200]
    else:
        print("network type error")
    final_rank_nums = []
    for network_type,meta_cell in zip(network_types,meta_cells):
        if  dataset_name in ['BT1307','BT1303','scrEXT009','scrSOL004','sc5rJUQ045']:
            for GRN_type in GRN_types:
                for Rank_num in Ranknums:
                    print('\n\n****************************'+ dataset_name + '————' + network_type+ '————'+ GRN_type +'————' + str(Rank_num) +'————' f'{meta_cell}'+'****************************')
                    # raw data path
                    
                    gold_network_path = os.path.join(BASE_DIR, 'rawData', 'Networks', 'human', GRN_type, network_type + '.csv')
                    pathHumanTF = os.path.join(BASE_DIR, 'rawData', 'Networks', 'human-tfs.csv')
                    gene_expression_path = os.path.join(BASE_DIR, 'rawData', 'scRNA-Seq', dataset_name, network_type, 'ExpressionData.csv')
                    gene_order_path = os.path.join(BASE_DIR, 'rawData', 'scRNA-Seq', dataset_name, network_type, 'GeneOrdering.csv')
                    # save path
                    result_path = os.path.join(data_source, dataset_name, network_type, GRN_type, 'Top' + str(Rank_num)) + os.sep
                    if not os.path.isdir(result_path):
                        os.makedirs(result_path)

                    preprocess_scCSGRN(result_path, gold_network_path, gene_expression_path, dataset_name, gene_order_path, Rank_num, pathHumanTF,meta_cell,network_type)
        else:
            for GRN_type in GRN_types:
                for Rank_num in Ranknums:
                    print('\n\n****************************'+ dataset_name + '————' + network_type + '————'+ GRN_type  +'————' + str(Rank_num) + '————' f'{meta_cell}''****************************')
                    # raw data path          
                    gold_network_path = os.path.join(BASE_DIR, 'rawData', 'Networks', 'mouse', GRN_type, network_type + '.csv')
                    pathHumanTF = os.path.join(BASE_DIR, 'rawData', 'Networks', 'mouse-tfs.csv')
                    gene_expression_path = os.path.join(BASE_DIR, 'rawData', 'scRNA-Seq', dataset_name, network_type, 'ExpressionData.csv')
                    gene_order_path = os.path.join(BASE_DIR, 'rawData', 'scRNA-Seq', dataset_name, network_type, 'GeneOrdering.csv')
                    # save path
                    result_path = os.path.join(data_source, dataset_name, network_type, GRN_type, 'Top' + str(Rank_num)) + os.sep
                    if not os.path.isdir(result_path):
                        os.makedirs(result_path)

                    preprocess_scCSGRN(result_path, gold_network_path, gene_expression_path, dataset_name, gene_order_path, Rank_num, pathHumanTF,meta_cell,network_type)







