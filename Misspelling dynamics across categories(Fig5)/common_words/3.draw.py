import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os

# 读取数据
df_sums = pd.read_csv('./all_sums.csv')
df_ture = pd.read_csv('./all_ture.csv')
plt.rcParams["font.family"] = "Times New Roman"

# 初始化保存最大核密度点的列表
max_density_points_x_sums = []
max_density_points_y_sums = []
max_density_points_x_ture = []
max_density_points_y_ture = []

# 准备绘图
plt.figure(figsize=(8.5, 5))

for column in df_ture.columns:
    data = df_ture[column].dropna()  # 去除NaN值
    if len(data) > 1:  # 确保有足够的数据进行KDE估计
        kde = gaussian_kde(data)
        density = kde(data)  # 计算核密度估计值

        # 创建散点图，颜色深浅根据密度值变化
        plt.scatter([column] * len(data), data, c="#A5B6C5", s=5, alpha=0.3, edgecolors='none')

        # 找到密度最大的点，并存储
        max_density_idx = np.argmax(density)
        if data.iloc[max_density_idx] > 0:
            max_density_points_x_ture.append(column)
            max_density_points_y_ture.append(data.iloc[max_density_idx])

# 绘制 all_sums.csv 文件中的数据
folder_path = './data'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df['sum'] = df.iloc[:, 2:].sum(axis=1)
    df['ture'] = df.iloc[:, 1]
    first_non_zero_sum_idx = (df['sum'] != 0).idxmax()
    first_non_zero_ture_idx = (df['ture'] != 0).idxmax()
    start_idx = max(first_non_zero_sum_idx, first_non_zero_ture_idx)
    df = df.loc[start_idx:].reset_index(drop=True)

for column in df_sums.columns:
    data = df_sums[column].dropna()  # 去除NaN值
    if len(data) > 1:  # 确保有足够的数据进行KDE估计
        kde = gaussian_kde(data)
        density = kde(data)  # 计算核密度估计值

        # 创建散点图，颜色深浅根据密度值变化
        plt.scatter([column] * len(data), data, c="#EECCC8", s=5, alpha=0.3, edgecolors='none')

        # 找到密度最大的点，并存储
        max_density_idx = np.argmax(density)
        if data.iloc[max_density_idx] > 0:
            max_density_points_x_sums.append(column)
            max_density_points_y_sums.append(data.iloc[max_density_idx])

# 保存最大核密度点到同一个CSV文件
df_max_density = pd.DataFrame({
    'x_sums': max_density_points_x_sums,
    'y_sums': max_density_points_y_sums,
    'x_ture': max_density_points_x_ture,
    'y_ture': max_density_points_y_ture
})

df_max_density.to_csv('max_density_points.csv', index=False)

# 绘制最大密度点的连线
for i in range(len(max_density_points_x_sums) - 1):
    if int(max_density_points_x_sums[i]) != int(max_density_points_x_sums[i + 1]) - 1:
        plt.plot([max_density_points_x_sums[i], max_density_points_x_sums[i + 1]], [max_density_points_y_sums[i], max_density_points_y_sums[i + 1]], color='#C26B61', linestyle=':', linewidth=1.5, marker=' ')
    else:
        plt.plot([max_density_points_x_sums[i], max_density_points_x_sums[i + 1]], [max_density_points_y_sums[i], max_density_points_y_sums[i + 1]], color='#C26B61', linestyle='solid', linewidth=1.5, marker=' ')

for i in range(len(max_density_points_x_ture) - 1):
    if int(max_density_points_x_ture[i]) != int(max_density_points_x_ture[i + 1]) - 1:
        plt.plot([max_density_points_x_ture[i], max_density_points_x_ture[i + 1]], [max_density_points_y_ture[i], max_density_points_y_ture[i + 1]], color='#496C88', linestyle=':', linewidth=1.5, marker=' ')
    else:
        plt.plot([max_density_points_x_ture[i], max_density_points_x_ture[i + 1]], [max_density_points_y_ture[i], max_density_points_y_ture[i + 1]], color='#496C88', linestyle='solid', linewidth=1.5, marker=' ')

# 添加颜色条和其他图形设置
legend_items = [
    mlines.Line2D([], [], color='#A5B6C5', marker='o', markersize=8, linestyle='None', label='Correct spelling'),
    mlines.Line2D([], [], color='#496C88', label='Correct spelling of fit curve'),
    mlines.Line2D([], [], color='#EECCC8', marker='o', markersize=8, linestyle='None', label='Misspelling'),
    mlines.Line2D([], [], color='#C26B61', label='Misspelling fit curve')
]

ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False,fontsize=12)
plt.xlabel('Year', fontsize=18, fontweight='bold')
plt.ylabel('Frequency', fontsize=18, fontweight='bold')
plt.yscale('log')
plt.xlim(0, 519)
plt.xticks(np.arange(0, len(df_ture.columns), step=50))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=3, color='black', direction='out', width=0.6)
plt.tick_params(axis='both', labelsize=14)  # 设置刻度字体大小为12

plt.savefig('output_words_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('output_words_plot.jpg', dpi=300, bbox_inches='tight')
plt.savefig('output_words_plot.tiff', dpi=300, bbox_inches='tight')
plt.savefig('output_words_plot.svg', dpi=300, bbox_inches='tight')
plt.show()




