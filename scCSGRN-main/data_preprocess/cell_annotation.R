# 加载必要包 --------------------------------------------------------------
if (!require("Seurat")) install.packages("Seurat")        # 单细胞分析
if (!require("clustree")) install.packages("clustree")    # 聚类树分析
if (!require("ggplot2")) install.packages("ggplot2")      # 可视化
if (!require("cluster")) install.packages("cluster")      # 轮廓系数
if (!require("dplyr")) install.packages("dplyr")          # 数据操作
if (!require("SingleR")) install.packages("SingleR")
if (!requireNamespace("BiocManager", quietly = TRUE))

if (!require("celldex")) install.packages("celldex")  # 提供参考数据集


#if (!require("ROGUE")) {
 # install.packages("devtools")
 # devtools::install_github("PaulingLiu/ROGUE")
#}


library(Seurat)
library(clustree)
library(ggplot2)
library(cluster)
library(dplyr)
library(SingleR)
library(celldex)
#install.packages("Matrix")  # 如果未安装过该包
library(Matrix)   
# https://github.com/PaulingLiu/ROGUE.git
#library(SeuratObject)
# 参数设置 ----------------------------------------------------------------
# 预处理开关（TRUE: 执行预处理，FALSE: 跳过）
do.preprocessing <- TRUE    # 总开关
do.qc <- TRUE               # 质控过滤
do.normalize <- TRUE        # 归一化
do.variable_features <- TRUE # 高变基因筛选
do.scale <- TRUE            # 标准化
do.pca <- TRUE              # PCA降维

# 聚类参数
pca.dims <- 1:10           # PCA使用的维度（根据ElbowPlot调整）
resolutions <- seq(0.1, 1.5, by = 0.2)  # 测试的resolution范围



# 文件路径

#input_csv <- "H:\\GRN\\scCSGRN\\R\\dataset\\sc5rJUQ045\\sc5rJUQ045.csv"# 输入CSV文件路径
#output_clusters <- "H:\\GRN\\scCSGRN\\R\\dataset\\sc5rJUQ045\\sc5rJUQ045_clusters.csv" # 聚类结果保存路径
#output_dir <- "H:\\GRN\\scCSGRN\\R\\dataset\\A_jieguo\\sc5rJUQ045" # 新增图表输出目录
#output_umap <- "H:\\GRN\\scCSGRN\\R\\dataset\\A_jieguo\\sc5rJUQ045\\umap_coordinates.csv" # UMAP坐标保存路径

##################
#output_variable_features <- "F:\\liulanqi\\dataset\\hCBMC\\hCBMC_features.csv"  # 保存高变基因
#output_variance <- "F:\\liulanqi\\dataset\\hCBMC\\hCBMC_variance.csv"  # 高变基因方差保存路径
####################


# 设置数据集名称
dataset_name <- "Mtab-11821"

# 设置基础路径
base_path <- "D:\\GRN\\scCSGRN\\R\\dataset"

# 输入文件路径
input_csv <- paste0(base_path, "\\", dataset_name, "\\", dataset_name, ".csv")

# 聚类结果保存路径
output_clusters <- paste0(base_path, "\\", dataset_name, "\\", dataset_name, "_clusters.csv")

# 图表输出目录
output_dir <- paste0(base_path, "\\B_jieguo\\", dataset_name)

# UMAP坐标保存路径
output_umap <- paste0(output_dir, "\\umap_coordinates.csv")









# 步骤1：读取数据并创建Seurat对象 ------------------------------------------
# 注意：CSV文件行为细胞，列为基因，需转置为基因为行、细胞为列
data <- read.csv(input_csv, row.names = 1, check.names = FALSE)  # 第一列为细胞名
data_matrix <- Matrix(as.matrix(data), sparse = TRUE)  # 转换为稀疏矩阵
#data_matrix <- t(data) %>% as("dgCMatrix")  # 转置为稀疏矩阵
seurat_obj <- CreateSeuratObject(counts = data_matrix, project = "scRNA_project")

# 步骤2：可选预处理流程 ---------------------------------------------------
if (do.preprocessing) {
  # --- 质控过滤（示例：过滤低表达细胞）---
  if (do.qc) {
    # 示例：过滤检测到少于100个基因的细胞（根据数据调整阈值）
    seurat_obj <- subset(seurat_obj, subset = nFeature_RNA > 100)
  }
  
  # --- 归一化 ---
  if (do.normalize) {
    seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize")
    #logcounts <- GetAssayData(seurat_obj, slot = "data")
  }
  
  # --- 高变基因筛选 ---
  if (do.variable_features) {
    seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
    logcounts <- GetAssayData(seurat_obj, slot = "data")
    # 保存高变基因名单
    #variable_genes <- VariableFeatures(seurat_obj)
    #write.csv(data.frame(Gene = variable_genes), 
              #file = output_variable_features, 
              #row.names = FALSE)
    #print("高变基因保存完成")
  } else {
    # 若跳过高变基因筛选，使用所有基因
    VariableFeatures(seurat_obj) <- rownames(seurat_obj)
  }
  
  # --- 标准化 ---
  if (do.scale) {
    seurat_obj <- ScaleData(seurat_obj)
  }
  
  # --- PCA降维 ---
  if (do.pca) {
    seurat_obj <- RunPCA(seurat_obj, npcs = 50)
    ElbowPlot(seurat_obj, ndims = 50)  # 根据拐点选择pca.dims
  }
}

# 步骤3：自动优化resolution值 ---------------------------------------------
# 3.1 构建KNN图并运行多个resolution的聚类
seurat_obj <- FindNeighbors(seurat_obj, dims = pca.dims)
seurat_obj <- FindClusters(seurat_obj, resolution = resolutions, verbose = FALSE)

# 3.2 使用clustree可视化分群稳定性
clustree_plot <- clustree(seurat_obj, prefix = "RNA_snn_res.", 
                          node_colour = "pink", 
                          layout = "sugiyama") + 
  ggtitle("Clustering Tree for Resolution Optimization")
ggsave(file.path(output_dir, "clustree.png"), plot = clustree_plot, width = 10, height = 8, dpi = 600)
print(clustree_plot)

# 3.3 计算轮廓系数选择最佳resolution
pca_data <- Embeddings(seurat_obj, "pca")[, pca.dims]  # 提取PCA数据
sil_scores <- sapply(resolutions, function(res) {
  cluster_col <- paste0("RNA_snn_res.", res)
  clusters <- as.numeric(as.character(seurat_obj[[cluster_col]][, 1]))
  sil <- silhouette(clusters, dist(pca_data))
  mean(sil[, 3])
})

# 绘制轮廓系数曲线
plot(resolutions, sil_scores, type = "b", pch = 19,
     xlab = "Resolution", ylab = "Average Silhouette Score",
     main = "Optimal Resolution Selection")
optimal_res <- resolutions[which.max(sil_scores)]
cat("Optimal resolution selected:", optimal_res, "\n")
print(resolutions)
print(sil_scores)
# 检查最佳resolution对应的列是否存在
expected_column <- paste0("RNA_snn_res.", optimal_res)
if (!expected_column %in% colnames(seurat_obj@meta.data)) {
  stop("列 ", expected_column, " 不存在，请检查是否已运行 FindClusters!")
}

# 查看该列的数据类型
print(class(seurat_obj@meta.data[[expected_column]]))
# 期望输出为 "factor" 或 "character"/"numeric"

Idents(seurat_obj) <- seurat_obj@meta.data[[expected_column]]


# 验证设置是否成功
head(Idents(seurat_obj))  # 应显示因子型聚类标签


# 步骤4：应用最佳resolution并可视化 ----------------------------------------
#Idents(seurat_obj) <- seurat_obj[[paste0("RNA_snn_res.", optimal_res)]]
seurat_obj <- RunUMAP(seurat_obj, dims = pca.dims)

# 绘制UMAP图
umap_plot1 <-DimPlot(seurat_obj, reduction = "umap", label = TRUE, pt.size = 0.3) +
  #ggtitle(paste("UMAP Clustering (Resolution =)", optimal_res)) +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # 移除主要网格线
    panel.grid.minor = element_blank(),  # 移除次要网格线
    panel.border = element_rect(colour = "black", fill = NA, size = 0.5)  # 添加黑色边框
  )
print(umap_plot1)
ggsave(file.path(output_dir, "cluster.png"), plot = umap_plot1, width = 8, height = 7, dpi = 600)
# 加载参考数据集（以HumanPrimaryCellAtlas为例）



ref_data <- celldex::HumanPrimaryCellAtlasData()
#ref_data <- MouseRNAseqData()
# 提取Seurat对象的log归一化数据
#gcounts <- GetAssayData(seurat_obj, slot = "data")


# 运行SingleR（按簇注释）
singler_results <- SingleR(
  test = logcounts,
  ref = ref_data,
  labels = ref_data$label.main,
  clusters = Idents(seurat_obj)
)

# 更新Seurat对象的细胞类型标签
cluster_labels <- singler_results$labels
names(cluster_labels) <- levels(seurat_obj)
seurat_obj <- RenameIdents(seurat_obj, cluster_labels)
seurat_obj$celltype <- Idents(seurat_obj)



# 可视化
umap_plot2 <-DimPlot(seurat_obj, reduction = "umap", label = TRUE, pt.size = 0.3)+# group.by = "celltype",
  #ggtitle("SingleR Cell Type Annotation")
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # 移除主要网格线
    panel.grid.minor = element_blank(),  # 移除次要网格线
    panel.border = element_rect(colour = "black", fill = NA, size = 0.5)  # 添加黑色边框
  )
print(umap_plot2)
ggsave(file.path(output_dir, "celltypes.png"), plot = umap_plot2, width = 8, height = 7, dpi = 600)



# 步骤5：保存结果 --------------------------------------------------------
# 保存聚类标签
write.csv(data.frame(Cell = colnames(seurat_obj), 
                     Cluster = Idents(seurat_obj)),
          file = output_clusters, row.names = FALSE)

# 保存UMAP坐标 + 细胞类型信息（不含Cluster）
umap_coords <- Embeddings(seurat_obj, "umap") %>%
  as.data.frame() %>%
  mutate(Cell = rownames(.),
         CellType = seurat_obj$celltype)


# 保存为CSV
write.csv(umap_coords, file = output_umap, row.names = FALSE)
print("已保存完成（UMAP坐标 + 细胞类型）。")

