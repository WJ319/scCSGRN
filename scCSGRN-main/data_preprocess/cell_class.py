import pandas as pd
import os

def split_expression_by_cell_type(expression: pd.DataFrame, cell_types_df: pd.DataFrame, base_path: str):
    """
    将表达矩阵按照细胞类型分组，并分别保存到对应的文件夹中。

    参数:
    - expression: pd.DataFrame，基因表达矩阵，行为基因，列为细胞名
    - cell_types_df: pd.DataFrame，第一列为细胞名，第二列为细胞类型
    - base_path: str，保存文件的根目录路径
    """

    # 创建 cell -> type 的映射字典
    cell_to_type = dict(zip(cell_types_df.iloc[:, 0], cell_types_df.iloc[:, 1]))

    # 只保留有注释的细胞
    available_cells = [cell for cell in expression.columns if cell in cell_to_type]
    expression = expression[available_cells]

    # 构建类型到细胞列表的映射
    type_to_cells = {}
    for cell in available_cells:
        cell_type = cell_to_type[cell]
        type_to_cells.setdefault(cell_type, []).append(cell)

    # 创建目录并保存文件
    os.makedirs(base_path, exist_ok=True)
    for cell_type, cells in type_to_cells.items():
        sub_expression = expression[cells]
        folder_path = os.path.join(base_path, str(cell_type))
        os.makedirs(folder_path, exist_ok=True)
        filename = os.path.join(folder_path, f"{cell_type}.csv")
        sub_expression.to_csv(filename)
        print(f"保存文件: {filename}，包含 {len(cells)} 个细胞")

    print("所有细胞分类完成！")


expression = pd.read_csv(r"H:\GRN\scCSGRN\R\dataset\Mtab-11821\Mtab-11821.csv", index_col=0)
cell_types_df = pd.read_csv(r"H:\GRN\scCSGRN\R\dataset\Mtab-11821\Mtab-11821_clusters.csv", header=None)
base_path = r"H:\GRN\scCSGRN\R\dataset\Mtab-11821\Mtab-11821"
split_expression_by_cell_type(expression, cell_types_df, base_path)  
