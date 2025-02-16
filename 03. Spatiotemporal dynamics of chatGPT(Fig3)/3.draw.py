import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20  # 可根据需要设置字体大小

df = pd.read_csv('combined_data.csv')
df['Time'] = pd.to_datetime(df['Time'])
start_date = df['Time'].min()
df['Time_days'] = (df['Time'] - start_date).dt.days
end_date = df['Time'].max()

# 避免零值对数问题
df['Frequency'] = df['Frequency'].replace(0, 1e-06)
df['Log_Frequency'] = np.log(df['Frequency'])

# 提取需要的数据列
x = df['Time_days'].values
y = df['Log_Frequency'].values
z = df['AvgVal'].values
categories = df['Node'].values  # 假设 Node 是类别列

# 创建每隔两个月的日期列表，确保结束日期是数据的最大日期
all_dates = pd.date_range(start='2022-12-16', end=end_date, freq='3MS')  # 根据数据的最大日期调整
manual_ticks = [(date - start_date).days for date in all_dates]  # 转换为 Time_days 刻度
manual_labels = [date.strftime('%b %Y') for date in all_dates]  # 格式化日期标签

# 过滤 z 值在 [0, 1] 范围的数据
valid_mask = (z >= 0) & (z <= 1)
x = x[valid_mask]
y = y[valid_mask]
z = z[valid_mask]
categories = categories[valid_mask]

# 对相同 (x, y) 点的 z 进行均值处理
df_filtered = pd.DataFrame({'x': x, 'y': y, 'z': z, 'category': categories})
df_grouped = df_filtered.groupby(['x', 'y'], as_index=False).agg({'z': 'mean', 'category': 'first'})

step = 1
filtered_mask = (df_grouped['x'] % step == 0)
x_filtered = df_grouped['x'][filtered_mask].values
y_filtered = df_grouped['y'][filtered_mask].values
z_filtered = df_grouped['z'][filtered_mask].values
categories_filtered = df_grouped['category'][filtered_mask].values

# 创建网格，利用刚才筛选点插值
x_grid = np.linspace(np.min(x_filtered), np.max(x_filtered), 40)
y_grid = np.linspace(np.min(y_filtered), np.max(y_filtered), 70)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# 插值计算 Z_grid
Z_grid = griddata((x_filtered, y_filtered), z_filtered, (X_grid, Y_grid), method='linear')
Z_grid = np.clip(Z_grid, 0, 1)  # 限制 Z_grid 值在 [0, 1] 范围

# 展开网格 (X_grid, Y_grid)为一维数据，便于之后计算
tree = cKDTree(np.vstack([X_grid.flatten(), Y_grid.flatten()]).T)

# 查找每个实际散点 (x, y) 在网格 (X_grid, Y_grid) 上的最近点
dist, indices = tree.query(np.vstack([x, y]).T)

# 筛选出实际点(x,y)与网格点(X_grid, Y_grid)距离小的点
valid_indices = dist > 100
z[valid_indices] = -1
# 使用这些索引从 Z_grid 中提取对应的曲面 Z 值
Z_surface = Z_grid.flatten()[indices]

# 计算与曲面之间的距离
distance = np.abs(z - Z_surface)

#筛选出实际点(x,y)的z值与曲面网格点(X_grid, Y_grid)的z距离小的点
distance_threshold = 100 # 你可以根据需求调整这个值

# 提取符合条件的散点
valid_points_mask = (abs(distance) <= distance_threshold) & (y > -13.8155105579642)  # 添加筛选条件

# 提取符合条件的散点
x_filtered_valid = x[valid_points_mask]
y_filtered_valid = y[valid_points_mask]
z_filtered_valid = z[valid_points_mask]
categories_filtered_valid = categories[valid_points_mask]
print(y)

# 创建自定义颜色映射
colors = ['#986400', '#C38501', '#FFBC35', '#F9D58D', '#FEE69B',
          '#F9FDB2', '#E5F1A9', '#A5C09C', '#81A479', '#357828']
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# 获取唯一类别
unique_categories = np.unique(categories_filtered_valid)
category_color_map = {cat: color for cat, color in zip(unique_categories,
                                                      ['#2C497F', '#E1404C', '#99939F', '#D39A71', '#7BABD9','#377728', '#7F5095'])}


fig = plt.figure(figsize=(9, 9))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')  # [0, 0, 1, 1] 代表图形的布局位置和大小

Z_grid = gaussian_filter(Z_grid, sigma=0.1)
# 绘制插值曲面
surface = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cmap, edgecolor='grey', alpha=0.6, linewidth=0.3)



# 绘制符合条件的散点
for category in unique_categories:
    mask = categories_filtered_valid == category
    color = category_color_map.get(category, '#CCCCCC')  # 默认颜色为灰色
    ax.scatter(x_filtered_valid[mask], y_filtered_valid[mask], z_filtered_valid[mask], label=f'Category {category}',
               edgecolor='none', color=color, s=8, alpha=1)
# 设置 z 轴范围为 [0, 1]
ax.set_zlim(0, 1)


# 设置标签和标题
ax.set_xlabel('Date', fontweight='bold', labelpad=11)
ax.set_ylabel('Frequency (log)', fontweight='bold', labelpad=10)
# 绘制 z 轴标签
ax.text2D(0.004, 0.57, 'Average causal strength', color='black', fontweight='bold',fontsize=22, ha='center', va='center', rotation=92, transform=ax.transAxes)


# # 添加图例
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#377728', label='ChatGPT', markersize=14),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#99939F', label='Chat GPT', markersize=14),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#D39A71', label='Chat-GPT', markersize=14),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#7F5095', label='ChatGTP', markersize=14),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#7BABD9', label='ChatGBT', markersize=14),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#E1404C', label='ChadGPT', markersize=14),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#2C497F', label='Chad GPT', markersize=14)]

ax.legend(frameon=False, handles=legend_elements, loc='upper center', prop={'size': 16.5, 'family': 'Times New Roman'},
          ncol=7, handletextpad=0.01, handleheight=1, columnspacing=0.5, bbox_to_anchor=(0.53, 1.0))

# 调整 x 轴刻度标签的位置，向左偏移
ax.set_xticks(manual_ticks)
ax.set_xticklabels(manual_labels, rotation=0, fontsize=16)

# plt.ion()  # 开启交互模式
# 设置视角（如果需要）
ax.view_init(elev=30, azim=-120)
# 添加 colorbar
colorbar = plt.colorbar(surface, ax=ax, shrink=0.70, aspect=22, pad=0.01)
colorbar.set_label('Average causal strength', fontsize=22, fontweight='bold',labelpad= 10)
# 显示图表
plt.savefig('PCMCI_3D.svg', dpi=300, bbox_inches='tight')
plt.savefig('PCMCI_3D.tif', dpi=300, bbox_inches='tight')
plt.savefig('PCMCI_3D.jpg', dpi=300, bbox_inches='tight')
plt.show()
