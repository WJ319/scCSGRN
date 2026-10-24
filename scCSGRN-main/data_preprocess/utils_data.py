import csv
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm




def get_origin_expression_data(gene_expression_path):
    # return gene expression matrix, cells list, number of genes, number of cells
    f_expression = open(gene_expression_path, encoding="utf-8")
    expression_reader = list(csv.reader(f_expression))
    cells = expression_reader[0][1:]
    num_cells = len(cells)

    expression_record = {}
    num_genes = 0
    for single_expression_reader in expression_reader[1:]:
        if single_expression_reader[0] in expression_record:
            print('Gene name ' + single_expression_reader[0] + ' repeat!')
        expression_record[single_expression_reader[0]] = list(map(float, single_expression_reader[1:]))
        num_genes += 1
    print(str(num_genes) + ' genes and ' + str(num_cells) + ' cells are included in origin expression data.' + '\n')
    return expression_record, cells, num_genes, num_cells


def get_origin_expression_data_mouse(gene_expression_path):
    # return gene expression matrix, cells list, number of genes, number of cells
    exp = pd.read_csv(gene_expression_path)
    cells = exp.columns[1:]
    gene_name = exp.iloc[:,0]
    expression_record = {}
    for i  in range(exp.shape[0]):
        row = exp.iloc[i, :].to_list()
        genename_row = row[0].upper()
        expression_row = row[1:]
        expression_record[genename_row] = expression_row

    return expression_record, cells, len(gene_name), len(cells)

def get_low_express_gene(origin_expression_record, num_cells):
    # get gene_list who were expressed in fewer than 10% of the cells
    gene_list = []
    threshold = num_cells // 10
    for gene in origin_expression_record:
        num = 0
        for expression in origin_expression_record[gene]:
            if expression != 0:
                num += 1
                if num > threshold:
                    break
        if num <= threshold:
            gene_list.append(gene)
    # up_gene_list = []
    # for gene in gene_list:
    #     up_gene_list.append(gene.upper())
    return gene_list

def get_gene_ranking(gene_order_path, low_express_gene_list, gene_num, flag):  # flag=True:write to output_path
    # 1.delete genes p-value>=0.01
    # 2.delete genes with low expression
    # 3.rank genes in descending order of variance
    # 4.return gene names list of top genes and variance_record of p-value<0.01
    f_order = open(gene_order_path)
    order_reader = list(csv.reader(f_order))
    # if flag:
    #     f_rank = open(output_path, 'w', newline='\n')
    #     f_rank_writer = csv.writer(f_rank)
    variance_record = {}
    variance_list = []
    significant_gene_list = []
    for single_order_reader in order_reader[1:]:
        # column 0:gene name
        # column 1:p value
        # column 2:variance
        #if float(single_order_reader[1]) >= 0.01:
            #break
            # continue
        if single_order_reader[0] in low_express_gene_list:
            continue
        variance = float(single_order_reader[1])
        if variance not in variance_record:  # 1 variance corresponding to 1 gene

            # print('variance', variance)
            variance_record[variance] = single_order_reader[0]
        else:
            # print(str(variance_record[variance]) + ' and ' + single_order_reader[0] + ' variance repeat!')
            # print(variance)
            variance_record[variance] = [variance_record[variance]]
            # print(variance_record[variance])
            variance_record[variance].append(single_order_reader[0])
            # print(variance_record[variance])

        variance_list.append(variance)
        tstr = single_order_reader[0]
        single_order_reader[0] = tstr.upper()
        significant_gene_list.append(single_order_reader[0])
    # print(variance_record[0.802089040840064])
    print('After delete genes with low expression, ' + str(len(variance_list)) + ' genes left.')

    variance_list.sort(reverse=True)


    gene_rank = []
    for single_variance_list in variance_list[0:gene_num]:

    #     print('single_variance_list',single_variance_list)
        if type(variance_record[single_variance_list]) is str:
            gene_rank.append(variance_record[single_variance_list])
        else:

            gene_rank.append(variance_record[single_variance_list][0])
            del variance_record[single_variance_list][0]
            if len(variance_record[single_variance_list]) == 1:
                variance_record[single_variance_list] = variance_record[single_variance_list][0]

    f_order.close()
    return gene_rank, significant_gene_list






def get_filtered_gold(gold_network_path, rank_list):
    # 1.Load origin gold file
    # 2.Delete genes not in rank_list
    # 3.return tf-targets dict and pair-score dict
    # Note: If no score in gold network, score=999
    f_gold = open(gold_network_path, encoding='UTF-8-sig')
    gold_reader = list(csv.reader(f_gold))
    num_origin_goldpair = len(gold_reader)-1

    for i in range(0, len(gold_reader) - 1):
        temp = gold_reader[i]
        s1 = str(temp[0])
        s2 = str(temp[1])

        temp[0] = s1.upper()
        temp[1] = s2.upper()

        gold_reader[i] = temp
    # print("gold_reader",gold_reader)
    # print("rank_list",rank_list)
    # print("gold_reader",gold_reader)
    # print("gold_reader[0]", gold_reader[0])
    has_score = True
    if len(gold_reader[0]) < 3:
        has_score = False

    gold_pair_record = {}
    gold_score_record = {}
    unique_gene_list = []
    #
    for single_gold_reader in gold_reader[1:]:
        # column 0: TF
        # column 1: target gene
        # column 2: regulate score


        if (single_gold_reader[0] not in rank_list) or (single_gold_reader[1] not in rank_list):
            continue
        gene_pair = [single_gold_reader[0], single_gold_reader[1]]
        str_gene_pair = single_gold_reader[0] + ',' + single_gold_reader[1]

        if single_gold_reader[0] not in unique_gene_list: unique_gene_list.append(single_gold_reader[0])
        if single_gold_reader[1] not in unique_gene_list: unique_gene_list.append(single_gold_reader[1])

        if str_gene_pair in gold_score_record:
            print('Gold pair repeat!')
        if has_score:
            # print("single_gold_reader[2]", single_gold_reader[2])
            gold_score_record[str_gene_pair] = float(single_gold_reader[2])
        else:
            gold_score_record[str_gene_pair] = 99999999
        if gene_pair[0] not in gold_pair_record:
            gold_pair_record[gene_pair[0]] = [gene_pair[1]]
        else:
            gold_pair_record[gene_pair[0]].append(gene_pair[1])
    '''
    At this time, the standard network gold_pair_record has been obtained, and the specific information is as follows:
    '''
    # print("gold_pair_record", gold_pair_record)
    # Some statistics of gold_network
    print(str(len(gold_pair_record)) + ' TFs and ' + str(
        len(gold_score_record)) + ' edges in gold_network consisted of genes in rank_list.')
    print(str(len(unique_gene_list)) + ' genes are common in rank_list and gold_network.')
    numTFs = len(gold_pair_record)
    num_gold_pair = len(gold_score_record)
    num_genes = len(unique_gene_list)

    # if flag:
    #     f_unique = open(output_path, 'w', encoding="utf-8", newline='\n')
    #     f_unique_writer = csv.writer(f_unique)
    #     out_unique = np.array(unique_gene_list).reshape(len(unique_gene_list), 1)
    #     f_unique_writer.writerows(out_unique)
    #     f_unique.close()
    return gold_pair_record, gold_score_record, unique_gene_list, numTFs,num_gold_pair,num_genes, num_origin_goldpair





def generate_filtered_gold(gold_pair_record, gold_score_record, output_path):
    # write filtered_gold to output_path
    # print("cnm")
    f_filtered = open(output_path, 'w', encoding="utf-8", newline='\n')
    f_filtered_writer = csv.writer(f_filtered)
    f_filtered_writer.writerow(['TF', 'Target', 'Score'])
    # print("cnm")
    for tf in gold_pair_record:
        once_output = []
        for target in gold_pair_record[tf]:
            # print(tf,target)
            single_output = [tf, target, gold_score_record[tf + ',' + target]]
            once_output.append(single_output)
        f_filtered_writer.writerows(once_output)
    f_filtered.close()


def generate_filtered_gold_2(gold_pair_record, output_path):
    # write filtered_gold to output_path
    f_filtered = open(output_path, 'w', encoding="utf-8", newline='\n')
    f_filtered_writer = csv.writer(f_filtered)
    f_filtered_writer.writerow(['TF', 'Target'])
    for tf in gold_pair_record:
        once_output = []
        for target in gold_pair_record[tf]:
            single_output = [tf, target]
            once_output.append(single_output)
        f_filtered_writer.writerows(once_output)
    f_filtered.close()

def save_expression(final_expression_matrix, output_path_exp, output_path_name):
    # write filtered_gold to output_path
    # print("cnm")
    gene_names = final_expression_matrix.keys()
    gene_index = list(range(len(gene_names)))
    gene_names_index = [gene_names,gene_index]
    gene_names_index = pd.DataFrame(gene_names_index).T
    gene_names_index.columns = ['gene', 'index']
    pd.DataFrame(gene_names_index).to_csv(output_path_name)

    if isinstance(final_expression_matrix, pd.DataFrame):
        final_expression_matrix.to_csv(output_path_exp)
    else:
        pd.DataFrame(final_expression_matrix).to_csv(output_path_exp)



def Trans_expression(final_expression_matrix, output_path_name):
    # write filtered_gold to output_path
    # print("cnm")
    gene_names = final_expression_matrix.keys()
    gene_index = list(range(len(gene_names)))
    gene_names_index = [gene_names,gene_index]
    gene_names_index = pd.DataFrame(gene_names_index).T
    gene_names_index.columns = ['gene', 'index']
    pd.DataFrame(gene_names_index).to_csv(output_path_name)

    if isinstance(final_expression_matrix, pd.DataFrame):
        df=final_expression_matrix
    else:
        df=pd.DataFrame(final_expression_matrix)
    return df

def get_normalized_expression_data(gene_expression_matirx):
    # expression_record, cells = get_origin_expression_data(gene_expression_path)
    gene_names = gene_expression_matirx.keys()
    
    Expression_matrix = {}
    # Expression_matrix = np.zeros((len(gene_expression_matirx), len(gene_expression_matirx[0])))
    index_row = 0
    for gene in gene_expression_matirx:
        gene_expression_matirx[gene] = np.log10(np.array(gene_expression_matirx[gene]) + 10 ** -2)
        Expression_matrix[gene] = gene_expression_matirx[gene]
        index_row += 1

    # Heat map
    # plt.figure(figsize=(15,15))
    # sns.heatmap(expression_matrix[0:100,0:100])
    # plt.show()

    return Expression_matrix

def data_transform(path_gene_list,path_TF_list, path_gene_pair_list, save_TF_list, save_gene_pair_index_list):

    gene_list = pd.read_csv(path_gene_list)
    # print(gene_list.shape)
    # gene_name = gene_list.iloc[:,1]
    # gene_index = gene_list.iloc[:,2]
    # print(gene_name)

    TF_list = pd.read_csv(path_TF_list)
    TF_name = TF_list.iloc[:,1]
    # print(TF_name)

    TF_index = []
    for i in TF_name:
        a = gene_list[gene_list.gene == i].index.tolist()
        TF_index.extend(a)
    # print(TF_index)
    # print(len(TF_index))

    TF_name_index = [TF_name.tolist(),TF_index]
    TF_name_index = pd.DataFrame(TF_name_index).T
    TF_name_index.columns = ['TF','index']
    # print(TF_name_index.shape)
    TF_name_index.to_csv(save_TF_list)

    gene_pair_name = pd.read_csv(path_gene_pair_list)
    TF_pair = gene_pair_name.iloc[:,0]
    target_pair = gene_pair_name.iloc[:,1]
    # print(TF_pair.shape)
    TF_pair_index = []
    for i in TF_pair:
        a = gene_list[gene_list.gene == i].index.tolist()
        TF_pair_index.extend(a)

    target_pair_index = []
    for i in target_pair:
        a = gene_list[gene_list.gene == i].index.tolist()
        target_pair_index.extend(a)
    TF_target_index = [TF_pair_index,target_pair_index]
    TF_target_index = pd.DataFrame(TF_target_index).T
    TF_target_index.columns = ['TF','target']
    # print(TF_target_index.shape)
    TF_target_index.to_csv(save_gene_pair_index_list)





import numpy as np
import pandas as pd


def interleave_blocks_np(a, b, k=4):

    if isinstance(a, pd.Series):
        a = a.to_numpy()
    if isinstance(b, pd.Series):
        b = b.to_numpy()

    n = len(a)
    result = []

    for i in range(0, n, k):
        result.append(a[i:i+k])
        result.append(b[i:i+k])

    return np.concatenate(result)


def preprocess_scCSGRN(result_path, gold_network_path, gene_expression_path, dataset_name, gene_order_path, Rank_num, pathHumanTF,meta_cell,network_type):
    # save dataset
    final_GRN_pos_file_name = result_path + 'Final_GRN_pos.csv' # prior GRN
    save_gene_pair_index_list = result_path + 'Final_GRN_pos_index.csv'  # prior GRN
    final_expression_file_name = result_path + 'Final_expression.csv'
    final_expression_gene_name = result_path + 'Final_gene_list.csv'
    final_TF_common_file_name = result_path + 'Final_TF_common.csv'
    save_TF_index_list = result_path + 'Final_TF_common_index.csv'

    Info_file_name = result_path + 'Info.csv'

    Info = {}

    # get the raw gene expression matrix, cell list, gene number, cell number
    if dataset_name in ['E-MTAB-14691', 'Mtab-11821']:
        origin_expression_record, cells, num_genes, num_cells = get_origin_expression_data_mouse(gene_expression_path)
    else:
        origin_expression_record, cells, num_genes, num_cells = get_origin_expression_data(gene_expression_path)

    Info["original genes"] = [num_genes]
    Info["original cells"] = [num_cells]

    # get the list of genes that are expressed in less than 10% of cells
    low_express_gene_list = get_low_express_gene(origin_expression_record, num_cells)

    print(str(len(low_express_gene_list)) + ' genes in low expression.'+ '\n')
    # Info["low expression genes"] = [len(low_express_gene_list)]

    # get the significant genes, significant genes are genes with p value less than 0.01, and expressed more than 10% cells, highly variable top Rank_num
    print('\n' + "The top 500/1000 genes were selected in order of variance..........")
    rank_list, significant_gene_list = get_gene_ranking(gene_order_path, low_express_gene_list, Rank_num, False)
    # significant genes is genes with p value less than 0.01, and expressed more than 10% cells
    Info["significant genes"] = [len(significant_gene_list)]
    #if Rank_num < len(significant_gene_list):
        # ranknum = Rank_num

    for i in range(0, len(rank_list) - 1):
        tstr = str(rank_list[i])
        tstr = tstr.upper()
        rank_list[i] = tstr

    # Based on the original ground-truth network, the ground-truth network with N=500/1000 genes was retrieved
    print('\n' + 'Find the prior network gold_pair_record, gold_score_record of top N genes, and the list of genes after intersection...............')
    gold_pair_record, gold_score_record, unique_gene_list, numTFs,num_gold_pair,num_genes,num_origin_gold_pair = get_filtered_gold(gold_network_path, rank_list)
    Info["original GRN"] = [num_origin_gold_pair]
    Info["filtered GRN"] = [num_gold_pair]
    Info["genes in filtered GRN"] = [num_genes]

    # utils.generate_filtered_gold(gold_pair_record, gold_score_record, FilteredGRN_file_name)

    # Obtain the normalized gene expression matrix according to the new gene list
    num_tf = -1
    num_label1 = 0
    num_label0 = 0
    # miss = 0
    flagFalse = [] # tf and target gene names that were not in the gene expression matrix
    all_gene_name = []
    all_TF_name_gold_pair = []
    final_gold_pair_record_pos = {}
    final_gold_pair_record_neg = {}
    pair_list_name_tf = []
    pair_list_name_target = []

    for i in gold_pair_record: # positve samples, key: TF, value: target genes regulated by this TF
        num_tf += 1
        for j in range(len(unique_gene_list)):
            #print('Generating matrix of gene pair ' + str(num_tf) + ' ' + str(j))
            tf_name = i
            target_name = unique_gene_list[j]

            flag = False
            # If the gene expression matrix contains TF and target genes, flag is set to true
            if (origin_expression_record.__contains__(tf_name) & origin_expression_record.__contains__(target_name)):
                flag = True
            else: # do not have gene expression
                flagFalse.extend([tf_name,target_name])
                # print("Do not have gene expression: " + tf_name +'--'+ target_name)
            if (flag):

                all_TF_name_gold_pair.append(tf_name)
                all_gene_name.append(tf_name)
                all_gene_name.append(target_name)
            else:
                # miss = miss + 1
                continue

    new_all_gene_name=list(set(all_gene_name))
    Info["final genes "] = [len(new_all_gene_name)]
    new_all_gene_name.sort(key=all_gene_name.index)
    # print(len(all_gene_name))
    # print(len(new_all_gene_name))

    new_all_TF_gold_pair = list(set(all_TF_name_gold_pair))
    Info["all TFs from gold pair"] = [len(new_all_TF_gold_pair)]
    # pd.DataFrame(new_all_TF_gold_pair).to_csv(final_TF_goldpair_file_name)


    human_tf = pd.read_csv(pathHumanTF)
    human_tfs = human_tf['TF'].to_list()
    all_tfs_human_tf = []
    for gene in new_all_gene_name:
        if gene in human_tfs:
            all_tfs_human_tf.append(gene)

    # pd.DataFrame(all_tfs_human_tf).to_csv(final_TF_human_file_name)
    Info["all TFs from human tf"] = [len(all_tfs_human_tf)]

    common_TFs = []
    for tf in new_all_TF_gold_pair:
        if tf not in common_TFs:
            common_TFs.append(tf)
    for tf in all_tfs_human_tf:
        if tf not in common_TFs:
            common_TFs.append(tf)
    pd.DataFrame(common_TFs).to_csv(final_TF_common_file_name)
    Info['common TFs'] = [len(common_TFs)]
    new_expression_matrix = {}
    for i in origin_expression_record:
        if i in new_all_gene_name:
            new_expression_matrix[i] = origin_expression_record[i]


    pro_meta=Trans_expression(new_expression_matrix,final_expression_gene_name)

    metaq_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MetaQ-main')
    if metaq_dir not in sys.path:
        sys.path.insert(0, metaq_dir)
    import MetaQ 
    if dataset_name=='Mtab-11821' and network_type=='Endothelial_cells':

        MetaQ.main(df=pro_meta,save_path=final_expression_file_name,metacell_num=meta_cell,batch_size=8)
    else:
        MetaQ.main(df=pro_meta,save_path=final_expression_file_name,metacell_num=meta_cell)
    df = pd.read_csv(final_expression_file_name)
    expression_record=df    
    
    
    
    label_list = []
    pair_list = []
    pair_list_name_tf = []
    pair_list_name_target = []


    total_matrix = []
    for i in gold_pair_record:
        num_tf += 1
        for j in range(len(new_all_gene_name)):
            
            tf_name = i
            target_name = new_all_gene_name[j]

            flag = False
            # if TF and target gene both have gene expression data, flag is True
            if (expression_record.__contains__(tf_name) & expression_record.__contains__(
                    target_name)) & (tf_name != target_name):
                flag = True
            else:  # BCL11AC3ORF14
                flagFalse.extend([tf_name, target_name])
                
            if (flag):
                # If TF is in gold_pair_record and the target gene is also regulated by this TF, the label is set to 1, otherwise the label is 0
                if tf_name in gold_pair_record and target_name in gold_pair_record[tf_name]:
                    label = 1
                    num_label1 += 1

                    if tf_name not in final_gold_pair_record_pos:
                        final_gold_pair_record_pos[tf_name] = [target_name]
                    else:
                        final_gold_pair_record_pos[tf_name].append(target_name)
                else:
                    label = 0
                    num_label0 += 1

                    if tf_name not in final_gold_pair_record_neg:
                        final_gold_pair_record_neg[tf_name] = [target_name]
                    else:
                        final_gold_pair_record_neg[tf_name].append(target_name)
                tf_data = expression_record[tf_name]
                target_data = expression_record[target_name]
                label_list.append(label)
                pair_list.append(tf_name + ',' + target_name)

                pair_list_name_tf.append(tf_name)
                pair_list_name_target.append(target_name)

                all_TF_name_gold_pair.append(tf_name)
                all_gene_name.append(tf_name)
                all_gene_name.append(target_name)
            else:
                # miss = miss + 1
                continue

            single_tf_list = []
            gap = 100
            for k in range(0, len(tf_data), gap):
                #feature = []
                a = tf_data[k:k + gap]
                b = target_data[k:k + gap]


                feature=interleave_blocks_np(a,b,k=4)



                if len(feature) == 2 * gap:
                    single_tf_list.append(feature)
                elif len(feature) > 0:
                        pad = np.zeros(2 * gap - len(feature))
                        single_tf_list.append(np.concatenate([feature, pad]))

            single_tf_list = np.stack(single_tf_list, axis=0)  
            total_matrix.append(single_tf_list)

    total_matrix = np.asarray(total_matrix)
    label_list = np.array(label_list)
    # pair_list = np.array(pair_list)

    np.save(result_path + 'matrix.npy', total_matrix)
    np.save(result_path + 'label.npy', label_list)
    np.save(result_path + 'gene_pair.npy', pair_list)

    # save prior GRN
    generate_filtered_gold_2(final_gold_pair_record_pos, final_GRN_pos_file_name)
    
    Info["final GRN "] = [num_label1]


    Info = pd.DataFrame(Info).T
    Info.to_csv(Info_file_name)

    # add index for TF and gene
    data_transform(final_expression_gene_name, final_TF_common_file_name, final_GRN_pos_file_name,
                    save_TF_index_list, save_gene_pair_index_list)

    
    gene_list = pd.read_csv(final_expression_gene_name)
    pair_list_id_tf = []
    for i in pair_list_name_tf:
        a = gene_list[gene_list.gene == i].index.tolist()
        pair_list_id_tf.extend(a)

    # gene_list = pd.read_csv(final_expression_gene_name)
    pair_list_id_target = []
    for i in pair_list_name_target:
        a = gene_list[gene_list.gene == i].index.tolist()
        pair_list_id_target.extend(a)

    np.save(result_path + 'pair_list_id_tf.npy', pair_list_id_tf)
    np.save(result_path + 'pair_list_id_target.npy', pair_list_id_target)


    #else:
        #print('ranknum is: ' + str(Rank_num) + ' but the number of all gene is ' + str(len(significant_gene_list)))
        # break

    ranknum = 0





