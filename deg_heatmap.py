import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib
import matplotlib as mpl
import seaborn as sns

time_resolution_names = ['5 min', '15 min', '30 min']
time_resolutions = ['5T', '15T', '30T']
bat_Pmax = ['0.5c0.5dMW', '0.25c0.25dMW', '0.25c0.5dMW', '0.125c0.25dMW', '0.125c0.125dMW', '0.5c1dMW', '1c1dMW', '1c2dMW', '2c2dMW', '2c4dMW', '4c4dMW'] #,
bat_Emax = ['0.25MWh', '0.5MWh', '1MWh', '2MWh', '4MWh'] #
DCACs = ['0.9', '1.0', '1.1', '1.25', '1.5', '2.0']
# ax_pos = [(0,0), (0,1), (0,2), (1,0), ]
# bat_sizes = ['025MWh_025MW', '05MWh_05MW', '1MWh_1MW', '2MWh_1MW', '2MWh_2MW', '4MWh_2MW', '4MWh_4MW'] #
# batt_sizes_names = ['1MWh, 1hr', '2MWh, 1hr', '2MWh, 2hr', '4MWh, 1hr', '4MWh, 2hr']

store_d = {}
final_deg_size_time = pd.DataFrame()
# keys = list()
# ENS_MWh_MW = pd.DataFrame()
for j in DCACs:
    for i in range(len(bat_Emax)):
        for h in range(len(bat_Pmax)):
            ikey = f'{bat_Emax[i]}_{bat_Pmax[h]}'
            for k in time_resolutions:
                # keys = keys.append(ikey)
                if os.path.isfile(f'.\Results\{ikey}_{k}_{j}_SOC.csv'):
                    store_d[(ikey,k)] = pd.read_csv(f'.\Results\{ikey}_{k}_{j}_SOC.csv')
                    final_deg_size_time.loc[k, ikey] = store_d[(ikey, k)]['Capacity_left'].iloc[-1]
                else:
                    continue
                # ENS_MWh_MW.loc[bat_Pmax[i], bat_Emax[i]] = store[(i,k)]['Capacity_left'].iloc[-1]

    print(final_deg_size_time)
    # fig, ax = subplots(nrows=2, ncols=3)
    ax = sns.heatmap(final_deg_size_time, annot=True, fmt='.4f', vmin=79.3, vmax=83.3) #xticklabels=keys, # , yticklabels=time_resolution_names
    ax.set(xlabel='Battery size/type', ylabel='Time resolution of power settlement (minutes)', title=f'DC-to-AC ratio {j}')
    plt.xticks(rotation=10)
    plt.show()
    # final_deg_size_time.to_excel('.\Heatmaps\heatmaps_dfs.xlsx', sheet_name='final_deg_by_size')


    # ## MWh MW plots
    # time_resolution_names = {'1T':'1 min', '5T':'5 min', '15T':'15 min', '30T':'30 min'}
    # time_resolutions = ['1T', '5T', '15T', '30T']
    # time_grans = {'1T':60, '5T':12, '15T':4, '30T':2}
    # bat_Pmax = ['0.25MW', '0.5MW', '1MW', '1MW', '2MW', '2MW', '4MW']
    # bat_Emax = ['0.25MWh', '0.5MWh', '1MWh', '2MWh', '2MWh', '4MWh', '4MWh']
    # # bat_sizes = ['025MWh_025MW', '05MWh_05MW', '1MWh_1MW', '2MWh_1MW', '2MWh_2MW', '4MWh_2MW', '4MWh_4MW']
    #
    # cases = {}
    # cases['0.25MWh'] = ['0.25MW']
    # cases['0.5MWh'] = ['0.25MW', '0.5MW']
    # cases['1MWh'] = ['0.5MW', '1MW']
    # cases['2MWh'] = ['0.5MW', '1MW', '2MW']
    # cases['4MWh'] = ['2MW', '4MW']

    # store = {}
    # ENS_MWh_MW = {}
    # for k in time_resolutions:
    #     ENS_MWh_MW[k] = pd.DataFrame()
    #     for i in bat_Emax:
    #         for j in cases[i]:
    #             if k == '1T':
    #                 store[(k, i, j)] = pd.read_pickle(f'.\Opt_res\{i}_{j}_{k}_Results')
    #                 ENS_MWh_MW[k].loc[j, i] = store[(k, i, j)]['Imb_under'].sum() / time_grans[k]
    #             else:
    #                 store[(k, i, j)] = pd.read_csv(f'.\Opt_res\{i}_{j}_{k}_Results.csv')
    #                 ENS_MWh_MW[k].loc[j, i] = store[(k,i,j)]['Imb_under'].sum() / time_grans[k]
    #
    #
    #
    #     # time_resolution_names = ['1 min', '5 min', '15 min', '30 min']
    #     # time_resolutions = ['1T', '5T', '15T', '30T']
    #     # bat_Pmax = ['025MW', '05MW', '1MW', '1MW', '2MW', '2MW', '4MW'] #,
    #     bat_Emax = ['025MWh', '05MWh', '1MWh', '2MWh', '2MWh', '4MWh', '4MWh'] #
    #     # bat_sizes = ['025MWh_025MW', '05MWh_05MW', '1MWh_1MW', '2MWh_1MW', '2MWh_2MW', '4MWh_2MW', '4MWh_4MW'] #
    #
    #     cases_deg = {}
    #     cases_deg['025MWh'] = ['025MW']
    #     cases_deg['05MWh'] = ['025MW', '05MW']
    #     cases_deg['1MWh'] = ['05MW', '1MW']
    #     cases_deg['2MWh'] = ['05MW', '1MW', '2MW']
    #     cases_deg['4MWh'] = ['2MW', '4MW']
    #
    #     store_deg = {}
    #     final_deg_MWh_MW = {}
    #     for k in time_resolutions:
    #         final_deg_MWh_MW[k] = pd.DataFrame()
    #         for i in bat_Emax:
    #             for j in cases_deg[i]:
    #                 store_deg[(k, i, j)] = pd.read_csv(f'.\Results\{i}_{j}_{k}_SOC.csv')
    #                 final_deg_MWh_MW[k].loc[j, i] = store_deg[(k, i, j)]['Capacity_left'].iloc[-1]
    #
    #     print(final_deg_MWh_MW)
    #
    #
    #     for k in time_resolutions:
    #         # fig, (ax1, ax2) = plt.subplots(1,2)
    #         ax1 = sns.heatmap(final_deg_MWh_MW[k], annot=True, fmt='.4f', xticklabels=bat_Emax, yticklabels=bat_Pmax)
    #         ax1.set(xlabel='Battery E cap.', ylabel='Battery P cap',
    #                 title=f'Remaining capacity after 15 years operation, time res: {time_resolution_names[k]}')
    #         plt.show()
    #         ax2 = sns.heatmap(ENS_MWh_MW[k], annot=True, fmt='.4f', xticklabels=bat_Emax, yticklabels=bat_Pmax)
    #         ax2.set(xlabel='Battery E cap.', ylabel='Battery P cap',
    #                 title=f'ENS over 1 year operation, time res: {time_resolution_names[k]}')
    #         plt.show()
    #
    #         # g2 = sns.heatmap(flights, cmap="YlGnBu", cbar=False, ax=ax2)
    #         # g2.set_ylabel('')
    #         # g2.set_xlabel('')
    #         # g3 = sns.heatmap(flights, cmap="YlGnBu", ax=ax3)
    #         # g3.set_ylabel('')
    #         # g3.set_xlabel('')
    #
    #
    #     # final_deg_MWh_MW.to_excel('.\Heatmaps\heatmaps_dfs.xlsx', sheet_name='final_deg_by_size')