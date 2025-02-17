import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.5, 25), sharex=True)

#------------------------------------
# common_words
#-----------------------------------
# 读取数据
df_sums = pd.read_csv('common_words/all_sums.csv')
df_ture = pd.read_csv('common_words/all_ture.csv')
plt.rcParams["font.family"] = "Times New Roman"

# 初始化保存最大核密度点的列表
max_density_points_x_sums = []
max_density_points_y_sums = []
max_density_points_x_ture = []
max_density_points_y_ture = []

# 准备绘图

for column in df_ture.columns:
    data = df_ture[column].dropna()  # 去除NaN值
    if len(data) > 1:  # 确保有足够的数据进行KDE估计
        kde = gaussian_kde(data)
        density = kde(data)  # 计算核密度估计值

        # 创建散点图，颜色深浅根据密度值变化
        ax3.scatter([column] * len(data), data, c="#A5B6C5", s=5, alpha=0.3, edgecolors='none')

        # 找到密度最大的点，并存储
        max_density_idx = np.argmax(density)
        if data.iloc[max_density_idx] > 0:
            max_density_points_x_ture.append(column)
            max_density_points_y_ture.append(data.iloc[max_density_idx])

# 绘制 all_sums.csv 文件中的数据
folder_path = 'common_words/data'
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
        ax3.scatter([column] * len(data), data, c="#EECCC8", s=5, alpha=0.3, edgecolors='none')

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
        ax3.plot([max_density_points_x_sums[i], max_density_points_x_sums[i + 1]], [max_density_points_y_sums[i], max_density_points_y_sums[i + 1]], color='#C26B61', linestyle=':', linewidth=1.5, marker=' ')
    else:
        ax3.plot([max_density_points_x_sums[i], max_density_points_x_sums[i + 1]], [max_density_points_y_sums[i], max_density_points_y_sums[i + 1]], color='#C26B61', linestyle='solid', linewidth=1.5, marker=' ')

for i in range(len(max_density_points_x_ture) - 1):
    if int(max_density_points_x_ture[i]) != int(max_density_points_x_ture[i + 1]) - 1:
        ax3.plot([max_density_points_x_ture[i], max_density_points_x_ture[i + 1]], [max_density_points_y_ture[i], max_density_points_y_ture[i + 1]], color='#496C88', linestyle=':', linewidth=1.5, marker=' ')
    else:
        ax3.plot([max_density_points_x_ture[i], max_density_points_x_ture[i + 1]], [max_density_points_y_ture[i], max_density_points_y_ture[i + 1]], color='#496C88', linestyle='solid', linewidth=1.5, marker=' ')

# 添加颜色条和其他图形设置
# legend_items = [
#     mlines.Line2D([], [], color='#A5B6C5', marker='o', markersize=8, linestyle='None', label='Correct spelling'),
#     mlines.Line2D([], [], color='#496C88', label='Correct spelling of fit curve'),
#     mlines.Line2D([], [], color='#EECCC8', marker='o', markersize=8, linestyle='None', label='Misspelling'),
#     mlines.Line2D([], [], color='#C26B61', label='Misspelling fit curve')
# ]

# ax3 = plt.gca()
ax3.spines['bottom'].set_linewidth(1.5)
ax3.spines['left'].set_linewidth(1.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax3.tick_params(axis='both', labelsize=12)
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontname("Times New Roman")


# ax3.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False,fontsize=12)
ax3.set_xlabel('Year', fontsize=13.5, fontweight='bold', fontname="Times New Roman")
ax3.set_ylabel('Frequency', fontsize=13.5, fontweight='bold', fontname="Times New Roman")
ax3.set_yscale('log')
ax3.set_xlim(0, 519)
ax3.set_xticks(np.arange(0, len(df_ture.columns), step=50))
ax3.minorticks_on()
ax3.tick_params(axis='x', which='minor', length=3, color='black', direction='out', width=0.6)
ax3.tick_params(axis='both', labelsize=12)  # 设置刻度字体大小为12

# plt.show()

#------------------------------------
# scientific_terms
#-----------------------------------
# 读取数据
df_sums = pd.read_csv('scientific_terms/all_sums.csv')
df_ture = pd.read_csv('scientific_terms/all_ture.csv')
plt.rcParams["font.family"] = "Times New Roman"
max_density_points_x_sums = []
max_density_points_y_sums = []
max_density_points_x_ture = []
max_density_points_y_ture = []



# 准备绘图
# plt.figure(figsize=(8.5, 5))


def create_custom_cmap():
    # 定义颜色的渐变（这里定义了从红色到黄色到绿色的渐变）
    colors = ["#C26B61","#DF9B92","#CFDAEE"]  # 自定义渐变的颜色
    cmap_name = "custom_cmap"
    return LinearSegmentedColormap.from_list(cmap_name, colors[::-1])

custom_cmap_sums = create_custom_cmap()  # 为 all_sums 数据创建自定义颜色渐变
custom_cmap_ture = create_custom_cmap()  # 为 ture 数据创建自定义颜色渐变
for column in df_ture.columns:
    data = df_ture[column].dropna()  # 去除NaN值
    if len(data) > 1:  # 确保有足够的数据进行KDE估计
        kde = gaussian_kde(data)
        density = kde(data)  # 计算核密度估计值

        # 创建散点图，颜色深浅根据密度值变化
        #plt.scatter([column] * len(data), data, c=density, cmap='Blues', s=1, alpha=0.5)
        ax1.scatter([column] * len(data), data, c="#A5B6C5", s=5, alpha=0.3,edgecolors='none')

        # 找到密度最大的点，并存储
        max_density_idx = np.argmax(density)
        if data.iloc[max_density_idx] > 0:
            max_density_points_x_ture.append(column)
            max_density_points_y_ture.append(data.iloc[max_density_idx])


# 绘制 all_sums.csv 文件中的数据，作为绿色散点图
folder_path = 'scientific_terms/data'
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
    data = df_sums[column].dropna()  # 去除NaN值
    if len(data) > 1:  # 确保有足够的数据进行KDE估计
        kde = gaussian_kde(data)
        density = kde(data)  # 计算核密度估计值

        # 创建散点图，颜色深浅根据密度值变化
        #plt.scatter([column] * len(data), data, c=density, cmap=custom_cmap_ture, s=5, alpha=1)
        ax1.scatter([column] * len(data), data,  c="#EECCC8", s=5, alpha=0.3,edgecolors='none')

        # 找到密度最大的点，并存储
        max_density_idx = np.argmax(density)
        if data.iloc[max_density_idx] > 0:
            max_density_points_x_sums.append(column)
            max_density_points_y_sums.append(data.iloc[max_density_idx])


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
        ax1.plot([max_density_points_x_sums[i], max_density_points_x_sums[i + 1]], [max_density_points_y_sums[i], max_density_points_y_sums[i + 1]], color='#C26B61', linestyle=':', linewidth=1.5, marker=' ')
    else:
        ax1.plot([max_density_points_x_sums[i], max_density_points_x_sums[i + 1]], [max_density_points_y_sums[i], max_density_points_y_sums[i + 1]], color='#C26B61', linestyle='solid', linewidth=1.5, marker=' ')

# 绘制最大密度点的连线（对于 ture.csv 数据）
for i in range(len(max_density_points_x_ture) - 1):
    if int(max_density_points_x_ture[i]) != int(max_density_points_x_ture[i + 1]) - 1:
        ax1.plot([max_density_points_x_ture[i], max_density_points_x_ture[i + 1]], [max_density_points_y_ture[i], max_density_points_y_ture[i + 1]], color='#496C88', linestyle=':', linewidth=1.5, marker=' ')
    else:
        ax1.plot([max_density_points_x_ture[i], max_density_points_x_ture[i + 1]], [max_density_points_y_ture[i], max_density_points_y_ture[i + 1]], color='#496C88', linestyle='solid', linewidth=1.5, marker=' ')

# 添加颜色条
# cbar = plt.colorbar()
# cbar.set_label('Density')

legend_items = [
    mlines.Line2D([], [], color='#A5B6C5', marker='o', markersize=8, linestyle='None', label='Correct spelling'),
    mlines.Line2D([], [], color='#496C88', label='Correct spelling of fit curve'),
    mlines.Line2D([], [], color='#EECCC8', marker='o', markersize=8, linestyle='None', label='Misspelling'),
    mlines.Line2D([], [], color='#C26B61', label='Misspelling fit curve')

]

# ax = plt.gca()  # 获取当前坐标轴
ax1.spines['bottom'].set_linewidth(1.5)  # x 轴
ax1.spines['left'].set_linewidth(1.5)    # y 轴
ax1.spines['top'].set_visible(False)   # 关闭顶部边框
ax1.spines['right'].set_visible(False) # 关闭右侧边框

ax1.tick_params(axis='both', labelsize=12)
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontname("Times New Roman")


# 添加自定义图例，放置在图的外部上方
ax1.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4,frameon=False,fontsize=11,handletextpad=0.5,columnspacing=0.8) # 调整列之间的间距)
# 设置轴标签
# ax1.set_xlabel('Year', fontsize=18, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=13.5, fontweight='bold', fontname="Times New Roman")
ax1.set_yscale('log')
ax1.set_xlim(0, 519)  # 将 x 轴范围设置为 0 到 100
ax1.set_xticks(np.arange(0, len(df_ture.columns), step=50))  # 每10个点显示一个刻度
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', length=3, color='black', direction='out', width=0.6)
ax1.tick_params(axis='both', labelsize=12)  # 设置刻度字体大小为12


# plt.savefig('output_technic_plot.png', dpi=300, bbox_inches='tight')
# plt.savefig('output_technic_plot.jpg', dpi=300, bbox_inches='tight')
# plt.savefig('output_technic_plot.tiff', dpi=300, bbox_inches='tight')
# plt.savefig('output_technic_plot.svg', dpi=300, bbox_inches='tight')
# 显示图形
# plt.show()

#---------------------------------------
# brands
#---------------------------------------
# 读取数据
df_sums = pd.read_csv('brands/all_sums.csv')
df_ture = pd.read_csv('brands/all_ture.csv')
plt.rcParams["font.family"] = "Times New Roman"
max_density_points_x_sums = []
max_density_points_y_sums = []
max_density_points_x_ture = []
max_density_points_y_ture = []

# 准备绘图
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
            ax2.scatter([column] * len(data), data, c="#A5B6C5", s=5, alpha=0.3,edgecolors='none')

            max_density_idx = np.argmax(density)
            if data[max_density_idx] > 0:
                max_density_points_x_ture.append(column)
                max_density_points_y_ture.append(data[max_density_idx])
        except np.linalg.LinAlgError:
            print(f"Error with KDE for column: {column}")


# 绘制 all_sums.csv 文件中的数据，作为绿色散点图
folder_path = 'brands/data'
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
            ax2.scatter([column] * len(data), data, c="#EECCC8", s=5, alpha=0.3,edgecolors='none')

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
        ax2.plot([max_density_points_x_sums[i], max_density_points_x_sums[i + 1]], [max_density_points_y_sums[i], max_density_points_y_sums[i + 1]], color='#C26B61', linestyle=':', linewidth=1.5, marker=' ')
    else:
        ax2.plot([max_density_points_x_sums[i], max_density_points_x_sums[i + 1]], [max_density_points_y_sums[i], max_density_points_y_sums[i + 1]], color='#C26B61', linestyle='solid', linewidth=1.5, marker=' ')

# 绘制最大密度点的连线（对于 ture.csv 数据）
for i in range(len(max_density_points_x_ture) - 1):
    if int(max_density_points_x_ture[i]) != int(max_density_points_x_ture[i + 1]) - 1:
        ax2.plot([max_density_points_x_ture[i], max_density_points_x_ture[i + 1]], [max_density_points_y_ture[i], max_density_points_y_ture[i + 1]], color='#496C88', linestyle=':', linewidth=1.5, marker=' ')
    else:
        ax2.plot([max_density_points_x_ture[i], max_density_points_x_ture[i + 1]], [max_density_points_y_ture[i], max_density_points_y_ture[i + 1]], color='#496C88', linestyle='solid', linewidth=1.5, marker=' ')

# 添加颜色条
# cbar = plt.colorbar()
# cbar.set_label('Density')

legend_items = [
    mlines.Line2D([], [], color='#A5B6C5', marker='o', markersize=8, linestyle='None', label='Correct spelling'),
    mlines.Line2D([], [], color='#496C88', label='Correct spelling of fit curve'),
    mlines.Line2D([], [], color='#EECCC8', marker='o', markersize=8, linestyle='None', label='Misspelling'),
    mlines.Line2D([], [], color='#C26B61', label='Misspelling fit curve')

]


ax2.spines['bottom'].set_linewidth(1.5)  # x 轴
ax2.spines['left'].set_linewidth(1.5)    # y 轴
ax2.spines['top'].set_visible(False)   # 关闭顶部边框
ax2.spines['right'].set_visible(False) # 关闭右侧边框


ax2.tick_params(axis='both', labelsize=12)
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontname("Times New Roman")

ax2.set_yscale('log')
ax2.set_ylabel('Frequency', fontsize=13.5, fontweight='bold', fontname="Times New Roman")
ax2.set_xlim(0, 519)  # 将 x 轴范围设置为 0 到 100
ax2.set_xticks(np.arange(0, len(df_ture.columns), step=50))  # 每10个点显示一个刻度
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', length=3, color='black', direction='out', width=0.6)
ax2.tick_params(axis='both', labelsize=12)  # 设置刻度字体大小为12

# 在每个子图的左上角添加标签
ax1.text(-0.08, 1.15, 'A', transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right', fontname="Times New Roman")
ax2.text(-0.08, 1.15, 'B', transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='right', fontname="Times New Roman")
ax3.text(-0.08, 1.15, 'C', transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top', ha='right', fontname="Times New Roman")

plt.savefig('combined_plots.svg', dpi=300, bbox_inches='tight')

plt.show()