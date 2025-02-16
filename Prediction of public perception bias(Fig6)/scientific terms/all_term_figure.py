import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.family'] = 'Times New Roman'


def draw(fig, real, prediction, sheet_name, time, time2, i, color):
    ax = fig.add_subplot(4, 3, i)

    ax.grid(False)
    # 真实数据
    # ax.plot(real[0, :, 0], label="GBNC", linewidth=1)
    plt.fill_between([i for i in range(len(real))], real, -0.01, facecolor=color, alpha=0.13)
    # 预测数据
    # ax.plot(prediction[0, :, 0], label="CfC_output", linewidth=2)
    ax.plot(prediction, linewidth=2, color=color)

    ax.spines['top'].set_linewidth(0.9)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['right'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    ax.set_xlim(0, len(real))

    # 自定义刻度
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['', '', '', '', '', ''])
    #     plt.xticks([250, 350, 450, 500], ['', '', '', ''])
    if i == 12 or i == 3 or i == 6 or i == 9:
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xlim(50, 500)
        plt.xticks([50, 150, 250, 350, 450, 500], ['', '', '', '', '', ''])
    if i == 11 or i == 8 or i == 5 or i == 2:
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xlim(150, 500)
        plt.xticks([150, 250, 350, 450, 500], ['', '', '', '', ''])
    if i == 10 or i == 7 or i == 4 or i == 1:
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xlim(250, 500)
        plt.xticks([250, 350, 450, 500], ['', '', '', ''])
    if i == 12:  # 有x轴
        ax.set_xlabel('Year', fontsize=15, weight='bold')
        plt.xticks([50, 150, 250, 350, 450], ['1550', '1650', '1750', '1850', '1950'], fontsize=12)
    if i == 11:  # 有x轴
        ax.set_xlabel('Year', fontsize=15, weight='bold')
        plt.xticks([150, 250, 350, 450], ['1650', '1750', '1850', '1950'], fontsize=12)
    if i == 10:  # 有x轴
        ax.set_xlabel('Year', fontsize=15, weight='bold')
        plt.xticks([250, 350, 450], ['1750', '1850', '1950'], fontsize=12)
    if i == 1 or i == 4 or i == 7 or i == 10:  # 有y轴
        ax.set_ylabel('Normalized Frenquency', fontsize=15, weight='bold')
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=12)




    ymin, ymax = ax.get_ylim()
    rect = Rectangle((time, ymin), time2 - time, ymax - ymin, facecolor='black', edgecolor='none', alpha=0.1)
    ax.add_patch(rect)
    # ax.vlines(time, ymin, ymax, 'black', '--',linewidth=0.6,alpha=0.2) # 垂直1
    # ax.vlines(time2, ymin, ymax, 'black', '--', linewidth=0.6,alpha=0.2)  # 垂直2
    ax.set_ylim(ymin, ymax)
    ax.text(time, (ymin + ymax) / 2 + 0.2, f'∆t = {time2 - time}', rotation=90, horizontalalignment='right',
            fontsize=11)

    title_obj = ax.set_title(sheet_name, weight='bold', color='white', size=14)
    # 获取标题的位置
    tx, ty = title_obj.get_position()
    bbox = Rectangle((tx - 0.502, ty), width=1.0035, height=0.15, facecolor=color, alpha=1, transform=ax.transAxes,
                     clip_on=False)
    # 将矩形添加到 axes
    ax.add_patch(bbox)
    # 调整标题的 zorder 使其显示在矩形之上
    title_obj.set_zorder(1)


lists = ["liaison", "guarantee", "annually", "misspell", "vacuum", "embarrass", "upholstery", "publicly", "tyranny",
         "buoyant", "atheist", "repetition"]
times = [434, 195, 98, 279, 166, 125, 315, 229, 160, 275, 227, 159]  # 标记训练数据长度
times2 = [442, 203, 108, 313, 181, 140, 345, 251, 183, 316, 251, 197]  # 标记预测数据峰值
colors = ['#148758', '#963972', '#4EB0AE', '#1E6897', '#876287', '#E58D2E', '#74AF4B', '#4D78C8', '#EDB117', '#AD6239',
          '#CE5745', '#1E938D']

print(times)
print(times2)
print([j - i for i, j in zip(times, times2)])

fig = plt.figure(figsize=(16, 13))
plt.subplots_adjust(hspace=0.22, wspace=0.05)

i = 1
for list, time, time2, color in zip(lists, times, times2, colors):
    real = pd.read_excel("./CfC data.xlsx", sheet_name=list, usecols=[1])["real"].tolist()
    prediction = pd.read_excel("./CfC data.xlsx", sheet_name=list, usecols=[2])["predict"].tolist()
    draw(fig, real, prediction, list, time, time2, i, color)
    i += 1
plt.savefig('./figures/all terms_300dpi.jpg', bbox_inches='tight', dpi=300)
plt.show()