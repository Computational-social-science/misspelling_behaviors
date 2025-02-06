import os
import pandas as pd


def process_csv_file(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 计算第二列及之后的列的总和
    df['sum'] = df.iloc[:, 2:].sum(axis=1)
    df['ture'] = df.iloc[:, 1]
    first_non_zero_sum_idx = (df['sum'] != 0).idxmax()
    first_non_zero_ture_idx = (df['ture'] != 0).idxmax()
    # 找到较大的索引
    start_idx = max(first_non_zero_sum_idx, first_non_zero_ture_idx)
    # 从较大的索引开始截取数据
    df = df.loc[start_idx:].reset_index(drop=True)
    return df


def extract_and_save_sums(folder_path, output_file):
    # 获取文件夹中所有CSV文件的文件名
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 初始化一个空的DataFrame来存储所有的sum列
    all_sums_df = pd.DataFrame()
    all_ture_df = pd.DataFrame()

    for i, file in enumerate(csv_files):
        file_path = os.path.join(folder_path, file)
        df = process_csv_file(file_path)

        # 检查sum的和是否小于ture的和
        if df['sum'].sum() < df['ture'].sum():
            # 将sum_series添加为新的一行
            all_sums_df = all_sums_df.append(df['sum'], ignore_index=True)
            all_ture_df = all_ture_df.append(df['ture'],ignore_index=True)

    # 重命名列名为时间序列的序列数
    all_sums_df.columns = range(1, all_sums_df.shape[1] + 1)
    all_ture_df.columns = range(1, all_ture_df.shape[1] + 1)

    # 将结果写入一个新的CSV文件
    all_sums_df.to_csv(output_file, index=False)
    all_ture_df.to_csv('./all_ture.csv', index=False)


# 使用该函数
folder_path = './data'  # 替换为你的CSV文件夹路径
output_file = 'all_sums.csv'  # 输出文件名
extract_and_save_sums(folder_path, output_file)