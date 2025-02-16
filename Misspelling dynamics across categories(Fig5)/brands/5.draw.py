import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import matplotlib.lines as mlines

# 读取数据
df_sums = pd.read_csv('./all_sums.csv')
df_ture = pd.read_csv('./all_ture.csv')
plt.rcParams["font.family"] = "Times New Roman"
max_density_points_x_sums = []
max_density_points_y_sums = []
max_density_points_x_ture = []
max_density_points_y_ture = []

# 准备绘图
plt.figure(figsize=(8.5, 5))


def create_custom_cmap():
    # 定义颜色的渐变（这里定义了从红色到黄色到绿色的渐变）
    colors = ["#C26B61","#DF9B92","#CFDAEE"]  # 自定义渐变的颜色
    cmap_name = "custom_cmap"
    return LinearSegmentedColormap.from_list(cmap_name, colors[::-1])

custom_cmap_sums = create_custom_cmap()  # 为 all_sums 数据创建自定义颜色渐变
custom_cmap_ture = create_custom_cmap()  # 为 ture 数据创建自定义颜色渐变
for column in df_ture.columns:
    data = df_ture[column].dropna()
    if len(data) > 1:
        data = np.array(data).reshape(-1, 1)
        data += 1e-9 * np.random.randn(*data.shape)  # 添加小噪声
        try:
            kde = gaussian_kde(data.T)
            density = kde(data.T)
            plt.scatter([column] * len(data), data, c="#A5B6C5", s=5, alpha=0.3,edgecolors='none')

            max_density_idx = np.argmax(density)
            if data[max_density_idx] > 0:
                max_density_points_x_ture.append(column)
                max_density_points_y_ture.append(data[max_density_idx])
        except np.linalg.LinAlgError:
            print(f"Error with KDE for column: {column}")


# 绘制 all_sums.csv 文件中的数据，作为绿色散点图
folder_path = './data'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 初始化绘图


for file in csv_files:
    file_path = os.path.join(folder_path, file)
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
    # 检查是否绘制曲线
    #if df['sum'].sum() < df['ture'].sum():
        # 绘制ture列（蓝色填充曲线）
        #plt.plot(df['ture'], color='#A5B6C5', alpha=1)
        #plt.fill_between(range(len(df['ture'])), df['ture'], color='#A5B6C5', alpha=0.6)


for column in df_sums.columns:
    data = df_sums[column].dropna()
    if len(data) > 1:
        data = np.array(data).reshape(-1, 1)
        data += 1e-9 * np.random.randn(*data.shape)
        try:
            kde = gaussian_kde(data.T)
            density = kde(data.T)
            plt.scatter([column] * len(data), data, c="#EECCC8", s=5, alpha=0.3,edgecolors='none')

            max_density_idx = np.argmax(density)
            if data[max_density_idx] > 0:
                max_density_points_x_sums.append(column)
                max_density_points_y_sums.append(data[max_density_idx])
        except np.linalg.LinAlgError:
            print(f"Error with KDE for column: {column}")


# 保存最大核密度点到同一个CSV文件
df_max_density = pd.DataFrame({
    'x_sums': pd.Series(max_density_points_x_sums),
    'y_sums': pd.Series(max_density_points_y_sums),
    'x_ture': pd.Series(max_density_points_x_ture),
    'y_ture': pd.Series(max_density_points_y_ture)
})

df_max_density.to_csv('max_density_points.csv', index=False)

# 绘制最大密度点的连线（对于 all_sums.csv 数据）
for i in range(len(max_density_points_x_sums) - 1):
    if int(max_density_points_x_sums[i]) != int(max_density_points_x_sums[i + 1]) - 1:
        plt.plot([max_density_points_x_sums[i], max_density_points_x_sums[i + 1]], [max_density_points_y_sums[i], max_density_points_y_sums[i + 1]], color='#C26B61', linestyle=':', linewidth=1.5, marker=' ')
    else:
        plt.plot([max_density_points_x_sums[i], max_density_points_x_sums[i + 1]], [max_density_points_y_sums[i], max_density_points_y_sums[i + 1]], color='#C26B61', linestyle='solid', linewidth=1.5, marker=' ')

# 绘制最大密度点的连线（对于 ture.csv 数据）
for i in range(len(max_density_points_x_ture) - 1):
    if int(max_density_points_x_ture[i]) != int(max_density_points_x_ture[i + 1]) - 1:
        plt.plot([max_density_points_x_ture[i], max_density_points_x_ture[i + 1]], [max_density_points_y_ture[i], max_density_points_y_ture[i + 1]], color='#496C88', linestyle=':', linewidth=1.5, marker=' ')
    else:
        plt.plot([max_density_points_x_ture[i], max_density_points_x_ture[i + 1]], [max_density_points_y_ture[i], max_density_points_y_ture[i + 1]], color='#496C88', linestyle='solid', linewidth=1.5, marker=' ')


legend_items = [
    mlines.Line2D([], [], color='#A5B6C5', marker='o', markersize=8, linestyle='None', label='Correct spelling'),
    mlines.Line2D([], [], color='#496C88', label='Correct spelling of fit curve'),
    mlines.Line2D([], [], color='#EECCC8', marker='o', markersize=8, linestyle='None', label='Misspelling'),
    mlines.Line2D([], [], color='#C26B61', label='Misspelling fit curve')

]

ax = plt.gca()  # 获取当前坐标轴
ax.spines['bottom'].set_linewidth(1.5)  # x 轴
ax.spines['left'].set_linewidth(1.5)    # y 轴
ax.spines['top'].set_visible(False)   # 关闭顶部边框
ax.spines['right'].set_visible(False) # 关闭右侧边框

# 添加自定义图例，放置在图的外部上方
plt.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4,frameon=False,fontsize=12)
# 设置轴标签
plt.xlabel('Year', fontsize=18, fontweight='bold')
plt.ylabel('Frequency', fontsize=18, fontweight='bold')
plt.yscale('log')
plt.xlim(0, 519)  # 将 x 轴范围设置为 0 到 100
plt.xticks(np.arange(0, len(df_ture.columns), step=50))  # 每10个点显示一个刻度
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=3, color='black', direction='out', width=0.6)
plt.tick_params(axis='both', labelsize=14)  # 设置刻度字体大小为12

plt.savefig('output_brand_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('output_brand_plot.jpg', dpi=300, bbox_inches='tight')
plt.savefig('output_brand_plot.tiff', dpi=300, bbox_inches='tight')
plt.savefig('output_brand_plot.svg', dpi=300, bbox_inches='tight')
# 显示图形
plt.show()



