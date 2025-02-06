import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.plotting as tp
import io
from contextlib import redirect_stdout

data = pd.read_excel(r"./data.xlsx", usecols=[0, 1, 3, 4, 5, 6, 7, 8])
data['Date'] = pd.to_datetime(data['Date'])


start_date = datetime(2022, 11, 30)
end_date = datetime(2024, 8, 14)
date_list = [end_date - timedelta(days=i) for i in range((end_date - start_date).days + 1)]
output_dir = "./significant_links_output"
os.makedirs(output_dir, exist_ok=True)
var_names = [r'ChatGPT', r'ChatGBT', r'ChatGTP', r'ChadGPT', r'Chad GPT', r'Chat GPT', r'Chat-GPT']

# 循环处理每个时间段
for current_end_date in date_list:
    current_start_date = start_date
    filtered_data = data[(data['Date'] >= current_start_date) & (data['Date'] <= current_end_date)]
    filtered_data = filtered_data.drop(columns=['Date'])
    if filtered_data.empty:
        print(f"No data available for {current_start_date} to {current_end_date}. Skipping...")
        continue

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(filtered_data)

    dataframe = pp.DataFrame(
        normalized_data,
        datatime={0: np.arange(len(normalized_data))},
        var_names=var_names
    )

    # 初始化 PCMCI
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)
    tau_max = 8

    # 运行 PCMCI
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None, alpha_level=0.01)
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=tau_max, fdr_method='fdr_bh')

    # 保存显著链接信息
    output_file = os.path.join(output_dir, f"{current_end_date.strftime('%Y-%m-%d')}.txt")
    with open(output_file, mode="w", encoding="utf-8") as f:
        with io.StringIO() as buf, redirect_stdout(buf):
            pcmci.print_significant_links(
                p_matrix=q_matrix,
                val_matrix=results['val_matrix'],
                alpha_level=0.01
            )
            f.write(buf.getvalue())

    # 绘制网络图
    graph = pcmci.get_graph_from_pmatrix(
        p_matrix=q_matrix,
        alpha_level=0.01,
        tau_min=0,
        tau_max=tau_max,
        link_assumptions=None
    )
    results['graph'] = graph

    tp.plot_graph(
        figsize=(8, 6),
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=var_names,
        link_colorbar_label='edges',
        node_colorbar_label='nodes',
        show_autodependency_lags=False,
        arrow_linewidth=7,
        cmap_edges='YlGnBu',
        cmap_nodes='Oranges',
        vmin_edges=0,
        vmax_edges=1,
        vmin_nodes=-0.2,
        vmax_nodes=1,
    )

    # 绘制时间序列图
    tp.plot_time_series_graph(
        figsize=(10, 5),
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=var_names,
        link_colorbar_label='edges',
        arrow_linewidth=5,
        cmap_edges='YlGnBu',
        vmin_edges=0,
        vmax_edges=1.0,
    )

    print(f"Processed data for {current_start_date} to {current_end_date}. Results saved.")
