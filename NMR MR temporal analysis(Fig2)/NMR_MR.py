import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
from scipy.stats import gaussian_kde
from datetime import datetime


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'Times New Roman'
font = {'family': 'serif',
        'weight': "medium"
        }

'''
1、tipping_point为每个单词的起始临界点
2、Discipline_Nomenclature为研究的单词
3、z_offset_orange为c图为了让橙色的点没有遮挡在z轴上增加的偏离量
4、z_offset_red为c图为了让红色的点没有遮挡在z轴上增加的偏离量
5、lag是b图竖的文字标注离虚线的距离，由于每个案例的起点不一样，x轴长度不一样，所以标注的距离也要单独设置
'''

tipping_point = 6
Discipline_Nomenclature = "ChatGPT"
z_offset_orange = 0.003
z_offset_red = 0.01
lag = 20

# Load the data
file_path = './'+Discipline_Nomenclature+'.xlsx'  # Replace with the path to your data file
data = pd.read_excel(file_path, sheet_name="Sheet2")

# Calculate d(t) and a(t)
data['a(t)'] = data.iloc[tipping_point:, 2:].sum(axis=1)  # Sum of all columns except the first
data['d(t)'] = data.iloc[tipping_point:, 1]


# print(data)
# Calculate MR and NMR
data['MR'] = data['a(t)'] / (data['a(t)'] + data['d(t)'])
data['NMR'] = data['d(t)'] / data['a(t)'].replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero


# 计算核密度
# 在计算核密度之前，确保没有inf或NaN值
data['NMR'] = data['NMR'].replace([np.inf, -np.inf], np.nan)  # 将inf替换为NaN
data['MR'] = data['MR'].replace([np.inf, -np.inf], np.nan)  # 将inf替换为NaN

# 删除包含NaN的行
data.dropna(subset=['NMR', 'MR'], inplace=True)
values = np.vstack([data['NMR'], data['MR']])
kernel = gaussian_kde(values)

# Calculate the density for each point
data['Density'] = kernel(values)

# Find the point with the highest density
max_density_index = data['Density'].idxmax()

# Plotting MR vs NMR
fig = plt.figure(figsize=(22, 6))

ax1 = fig.add_subplot(131)

num_data_points = len(data)
color_map = np.linspace(0, 1, num_data_points)
scatter = ax1.scatter(data['NMR'], data['MR'], c=color_map, cmap='viridis', alpha=0.8,marker='x')
ax1.set_xlabel('NMR (Nomenclature-to-Misspelling Ratio)', size=15)
ax1.set_ylabel('MR (Misspelling Ratio)', size=15)
xlabel = ax1.get_xaxis().get_label()
xlabel.set_weight('bold')
ylabel = ax1.get_yaxis().get_label()
ylabel.set_weight('bold')

# Creating colorbar tick positions and labels
colorbar_ticks = np.linspace(0, 1, 6)
# print(colorbar_ticks)
data_ = pd.read_excel(file_path, sheet_name="Sheet1")
print(np.linspace(0, len(data.iloc[:,0]), 6, dtype=int))
colorbar_labels = [data_.iloc[i,0].strftime("%Y-%m-%d") for i in np.linspace(0, len(data.iloc[:,0]), 6, dtype=int)]

# 为colorbar创建一个新的Axes实例，位置紧邻主图
cbar_ax = fig.add_axes([0.34, 0.1, 0.01, 0.8])  # 参数为[left, bottom, width, height]
# Adjusting colorbar
cbar = fig.colorbar(scatter, label='Date', cax=cbar_ax)
cbar.set_ticks(colorbar_ticks)
cbar.set_ticklabels(colorbar_labels)
# 设置字体属性
font_properties = fm.FontProperties(size=8)
# 应用字体属性到colorbar的刻度标签
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font_properties)

ax2 = fig.add_subplot(132)

ax2.scatter(data.index, data['Density'], color='#D9D9D9',s=15)

polynomial_degree = 20  # 或者您选择的其他度数
coefficients = np.polyfit(data.index, data['Density'], polynomial_degree)
polynomial = np.poly1d(coefficients)
x_fit = np.linspace(data.index.min(), data.index.max(), len(data.index))
y_fit = polynomial(x_fit)
ax2.plot(x_fit, y_fit, color='black')  # 拟合曲线
derivative_coefficients = np.polyder(polynomial.coefficients)
derivative_polynomial = np.poly1d(derivative_coefficients)
# 评估导数函数
x_vals = np.linspace(data.index.min(), data.index.max(), 1000)
y_derivative = derivative_polynomial(x_vals)
# 找到零点的近似位置
zero_crossings = np.where(np.diff(np.sign(y_derivative)))[0]
first_peak = None
for crossing in zero_crossings:
    if y_derivative[crossing] > 0 and y_derivative[crossing + 1] < 0:
        first_peak = x_vals[crossing]
        break
print(round(first_peak)+1)
print(data.loc[data['time'] == round(first_peak)+1].index[0])
max_density_index = data.loc[data['time'] == round(first_peak)+1].index[0]


# 核密度最高的点
ymin2,ymax2 = ax2.get_ylim()
# ymax2 = 80
print("核密度最高点")
print(ymin2,ymax2)
ax2.vlines(max_density_index, ymin2, ymax2, 'gray', '--',linewidth=1.5, alpha=0.7,zorder=0) # 垂直
ax2.vlines(0, ymin2, ymax2, 'gray', '--',linewidth=1.5, alpha=0.7,zorder=0) # 垂直
ax2.set_ylim(ymin2,ymax2)

# print(max_density_index)
# print(data)
# print(data.loc[data.index == max_density_index].iloc[0, 0])
print("Time:",data.loc[data.index == max_density_index].iloc[0, 0])
print("MR:",round(data.loc[data.index == max_density_index,'MR'].iloc[0],5))
print("NMR:",round(data.loc[data.index == max_density_index,'NMR'].iloc[0],5))
print("Kernel Density:",round(data.loc[data.index == max_density_index,'Density'].iloc[0],5))
a_t_min = data.loc[data.index == tipping_point,'a(t)'].iloc[0]
d_t_min = data.loc[data.index == tipping_point,'d(t)'].iloc[0]
print("a(t)_min:",a_t_min)
print("d(t)_min:",d_t_min)
a_t_max = data.loc[data.index == max_density_index,'a(t)'].iloc[0]
d_t_max = data.loc[data.index == max_density_index,'d(t)'].iloc[0]
print("a(t)_max:",a_t_max)
print("d(t)_max:",d_t_max)

ax2.scatter(tipping_point, data.loc[data.index == tipping_point]['Density'], color='orange', label='Tipping point')
ax2.scatter(max_density_index, data.loc[data.index == max_density_index]['Density'], color='red', label='Initial steady state')

ax2.set_xlabel('Date', size=15)
ax2.set_ylabel('Kernel Density', size=15)
xlabel = ax2.get_xaxis().get_label()
xlabel.set_weight('bold')
ylabel = ax2.get_yaxis().get_label()
ylabel.set_weight('bold')

data_time = pd.read_excel(file_path, sheet_name="Sheet1")
datetime_string=str(data_time.loc[data_time.index == max_density_index].iloc[0, 0])
datetime_string2=str(data_time.loc[data_time.index == 0].iloc[0, 0])

ax2.legend(
    # frameon=False, #去掉图例边框
    fontsize=12,
    loc = 'upper right',
    #labelspacing=0.3,
    labelspacing=0.3,
    handletextpad=0.01,
    #bbox_to_anchor=(1.01, 0.999, 0.0005, 0.01),
    handlelength=1.25,  # 调整这里的值以设置符号的长度
    #handleheight=0.2,
    markerscale=1.18,
)


try:
    # 将字符串转换为 datetime 对象
    datetime_obj = datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")
    # 将 datetime 对象格式化为仅包含日期的字符串
    formatted_date = datetime_obj.strftime("%Y-%m-%d")
    ax2.text(max_density_index+lag-tipping_point, ymax2, "Initial steady state = "+str(formatted_date)+"    ", rotation=90, va="top")
    # 将字符串转换为 datetime 对象
    datetime_obj2 = datetime.strptime(datetime_string2, "%Y-%m-%d %H:%M:%S")
    # 将 datetime 对象格式化为仅包含日期的字符串
    formatted_date2 = datetime_obj2.strftime("%Y-%m-%d")
    ax2.text(0 - lag-8, ymax2, "Tipping point = " + str(formatted_date2)+"    ", rotation=90, va="top")
except:
    ax2.text(max_density_index + lag-tipping_point, ymax2, "Initial steady state = " + datetime_string+"    ", rotation=90, va="top")
    ax2.text(0 - lag-8, ymax2, "Tipping point = " + str(tipping_point)+"    ", rotation=90, va="top")
ax2.set_xlim(0 - (lag+20),data.iloc[-1,0]+120)

xticks = np.linspace(0, len(data.iloc[:,0]), 4, dtype=int)
xticks_labels = [data_.iloc[i,0].strftime("%Y-%m-%d") for i in xticks]
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticks_labels, fontsize=10,zorder=0) #y轴字体设置

# plt.title('Kernel Density for Each Data Point with Highest Density Point Marked')
# plt.legend()
plt.grid(False)
# rect2 = patches.Rectangle((max_density_index, ymin), len(data)-max_density_index, ymax-ymin
#                           , linewidth=0,edgecolor='none',facecolor='black',alpha=0.2,zorder=0)
# ax2.add_patch(rect2)


# a图阴影
ymin1, ymax1 = ax1.get_ylim()
ax1.set_ylim(ymin1,ymax1)

ax3 = fig.add_subplot(133,projection='3d')

# 指定的e和r值
specific_values = [(a_t_min, d_t_min),
                   (a_t_max, d_t_max)]

# 计算e和r的范围
a_values_all = [val[0] for val in specific_values]
d_values_all = [val[1] for val in specific_values]
a_min, a_max = min(a_values_all), max(a_values_all)
d_min, d_max = min(d_values_all), max(d_values_all)

# 扩展范围以提供一些边缘空间
a_range = a_max - a_min
d_range = d_max - d_min
a_values = np.linspace(a_min - 0.001 * a_range, a_max + 0.1 * a_range, 400)
d_values = np.linspace(d_min - 0.1 * d_range, d_max + 0.001 * d_range, 400)

# 初始化一个二维数组来存储函数值
f_values = np.zeros((len(a_values), len(d_values)))

# 计算函数值
for i, a in enumerate(a_values):
    for j, d in enumerate(d_values):
        f_values[i, j] = a**2 / (d * (a + d))

# 添加特定的点
i=0
for a, d in specific_values:
    f_value = a**2 / (d * (a + d))
    # 调整z_offset，确保球体至少露出一半

    if i==0:
        ax3.scatter([a], [d], [f_value-z_offset_orange], color='orange',zorder=0, s=70, depthshade=False, label='Tipping point')
        # draw_sphere(ax3, [a, d, f_value], ball_radius, 'orange')  # Tipping Point
    else:
        ax3.scatter([a], [d], [f_value+z_offset_red], color='red', zorder=0, s=70, depthshade=False, label='Initial steady state')
        # draw_sphere(ax3, [a, d, f_value], ball_radius, 'red')  # Steady state point
    i+=1

ax3.legend(
    # frameon=False
    fontsize=11,
    loc='upper right',
    bbox_to_anchor=(1.1, 0.83, 0.01, 0.1),#(横坐标位置，纵坐标位置，宽度，高度)
    labelspacing=0.3,
    handletextpad=0.01,
    handlelength=1.25,
    markerscale=0.8,
)


# 绘制图形
a_mesh, d_mesh = np.meshgrid(a_values, d_values)
# ax3 = plt.axes(projeyouction='3d')
surf = ax3.plot_surface(a_mesh, d_mesh, f_values.T, cmap='viridis', alpha=0.8)


# 设置坐标轴标签（使用英文）
font_properties = {'family': 'Times New Roman', 'color': 'black', 'weight': 'bold'}
ax3.set_xlabel('Misspelling frequency m(t)', labelpad=8,rotation=-18, fontdict=font_properties)
ax3.set_ylabel('Nomenclature frequency n(t)', labelpad=8,rotation=46, fontdict=font_properties)
ax3.set_zlabel('Function value f(m, n)', labelpad=8,rotation=87, fontdict=font_properties)

ax1.set_position([0.1, 0.1, 0.23, 0.8])  # left, bottom, width, height
ax2.set_position([0.425, 0.1, 0.23, 0.8])
ax3.set_position([0.56, 0.06, 0.45, 0.9]) # 留出更大的间距

ax1.text(-0.15, 1.08, 'A', transform=ax1.transAxes, fontsize=23, fontweight='bold', va='top')
ax2.text(-0.13, 1.08, 'B', transform=ax2.transAxes, fontsize=23, fontweight='bold', va='top')
ax3.text2D(0.05, 1, "C", transform=ax3.transAxes, fontsize=23, fontweight='bold', va='top')


ax3.tick_params(axis='x', pad=2)
ax3.tick_params(axis='y', pad=3)
ax3.tick_params(axis='z', pad=5)

ax1.vlines(data.loc[data.index == max_density_index,'NMR'].iloc[0], ymin1, ymax1, 'gray', '--',linewidth=1.5, alpha=0.7,zorder=0) # 垂直虚线
ax1.text(data.loc[data.index == max_density_index, 'NMR'].iloc[0] +0.35, ymin1, "   Initial steady state" , rotation=90)


plt.savefig(r"./chatgpt.svg")
# Display the plot
plt.show()
