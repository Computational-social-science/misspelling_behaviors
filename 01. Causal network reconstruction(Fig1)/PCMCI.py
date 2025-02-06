import numpy as np
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel(r"./data.xlsx", usecols=[0,1,3,4,5,6,7,8])
data['Date'] = pd.to_datetime(data['Date'])

start_date = '2022-11-30'
end_date = '2024-08-14'


data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
data = data.drop(columns=['Date'])
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
T, N = data.shape
var_names = [r'ChatGPT',r'ChatGBT', r'ChatGTP', r'ChadGPT', r'Chad GPT',r'Chat GPT', r'Chat-GPT',]
dataframe = pp.DataFrame(data,
                         datatime = {0:np.arange(len(data))},
                         var_names=var_names)

# 1、时序数据可视化
tp.plot_timeseries(dataframe,time_label="Day")
plt.savefig('plot_timeseries_300dpi.jpg', bbox_inches='tight', dpi=300)

# 2、滞后零散点图
parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=parcorr,
    verbosity=1)
correlations = pcmci.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']
matrix_lags = None
matrix_lags = np.argmax(np.abs(correlations), axis=2)
tp.plot_scatterplots(dataframe=dataframe
                     , add_scatterplot_args={'matrix_lags':matrix_lags}
                     , name='lag_plot_scatterplots_300dpi.jpg')

# 3、核密度估计
matrix = tp.setup_density_matrix(N=dataframe.N,
        var_names=dataframe.var_names)
# Standardize to better compare skewness with gaussianized data
dataframe.values[0] -= dataframe.values[0].mean(axis=0)
dataframe.values[0] /= dataframe.values[0].std(axis=0)
matrix.add_densityplot(dataframe=dataframe,
    matrix_lags=matrix_lags, label_color='red', label="standardized data",
    snskdeplot_args = {'cmap':'Greys', 'alpha':1., 'levels':4})
# Transform data to normal marginals
data_normal = pp.trafo2normal(data)
dataframe_normal = pp.DataFrame(data_normal, var_names=var_names)
matrix.add_densityplot(dataframe=dataframe_normal,
    matrix_lags=matrix_lags, label_color='black', label="gaussianized data",
    snskdeplot_args = {'cmap':'Reds', 'alpha':1., 'levels':4})
matrix.adjustfig(name='plot_densityplots_300dpi.jpg')
tp.plot_densityplots(dataframe=dataframe, add_densityplot_args={'matrix_lags':matrix_lags},name='plot_densityplots_300dpi.jpg')


# 4、滞后函数图
parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=parcorr,
    verbosity=1)
correlations = pcmci.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']

lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names':var_names, 'x_base':5, 'y_base':.5},name='lagged_dependencies_300dpi.jpg')

# 5、网络图
pcmci.verbosity = 1
tau_max = 8
results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None, alpha_level=0.01)
# print("p-values")
# print (results['p_matrix'].round(3))
# print("MCI partial correlations")
# print (results['val_matrix'].round(2))
q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=tau_max, fdr_method='fdr_bh')
pcmci.print_significant_links(
        p_matrix = q_matrix,
        val_matrix = results['val_matrix'],
        alpha_level = 0.01)
graph = pcmci.get_graph_from_pmatrix(p_matrix=q_matrix, alpha_level=0.01,
            tau_min=0, tau_max=tau_max, link_assumptions=None)
results['graph'] = graph
viridis_rev = plt.colormaps['viridis']

viridis_rev = viridis_rev.reversed()


tp.plot_graph(
    figsize=(8, 6),
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=var_names,
    link_colorbar_label='edges',
    node_colorbar_label='nodes',
    show_autodependency_lags=False,
    save_name=f"./figure/plot_graph_{end_date}_1.png",
    arrow_linewidth=7,
    cmap_edges='YlGnBu', # RdBu_r
    cmap_nodes='Oranges',
    vmin_edges=0, # -1
    vmax_edges=1,
    vmin_nodes=-0.2, # -1
    vmax_nodes=1,
    #link_label_fontsize=0
)

# 6、时间序列图
tp.plot_time_series_graph(
    figsize=(10, 5),
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=var_names,
    link_colorbar_label='edges',
    save_name=f"./figure/plot_time_series_graph{end_date}_1.png",
    arrow_linewidth=5,
    cmap_edges='YlGnBu',
    vmin_edges=0, # -1
    vmax_edges=1.0,
)