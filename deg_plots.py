import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import matplotlib as mpl
import seaborn as sns
import rainflow

plotting = True
size_time_deg_heatmap = True
MWh_MW_matrices = True
SOC_plots = True
Cycles_plots = 1
battery_plots = True
AC_to_DC = '1.1'
results_folder = r'M:\Documents\PhD\Code\PhD_Projects\IFE_degradation\Results\July_prelim_v1'
deg_results_folder = r'Deg_model_results'
opt_results_folder = r'Opt_model_results'
pattern = re.compile(r'No_cal_deg_(\d+.?\d*)MWh_(\d+.?\d*)MW_(\d+T)_(\d+.\d+)_SOC.csv')


def set_up(results_folder, deg_results_folder, opt_results_folder, pattern):
    deg_dir = os.path.join(results_folder, deg_results_folder)
    opt_dir = os.path.join(results_folder, opt_results_folder)
    time_labels = {'1T': '1 min', '5T': '5 min', '15T': '15 min', '30T': '30 min'}
    time_grans = {'1T': 60, '5T': 12, '15T': 4, '30T': 2}

    batteries = []
    for m, fn in enumerate(os.listdir(deg_dir)):
        if os.path.isfile(os.path.join(deg_dir, str(fn))):
            filename, file_extension = os.path.splitext(fn)
            opt_fn = fn.replace('SOC', 'Results')
            if file_extension == '.csv':

                # Degradation results
                deg_res = pd.read_csv(os.path.join(deg_dir, fn), index_col='time', usecols=['time', 'Capacity_left'])
                remaining_cap = deg_res['Capacity_left'].iloc[-1]
                match = pattern.match(fn)
                E = match.group(1)
                P = match.group(2)
                T = match.group(3)
                DC = match.group(4)

                # Optimisation results
                opt_filename, opt_file_extension = os.path.splitext(opt_fn)
                if int(T[:-1]) == 1:
                    opt_res = pd.read_pickle(os.path.join(opt_dir, opt_filename))
                else:
                    opt_res = pd.read_csv(os.path.join(opt_dir, opt_fn))
                imbalance = (abs(opt_res['Imb_under'].sum()) + abs(opt_res['Imb_over'].sum())) * (int(T[:-1]) /60) # Factored by time granularity since imbalance values are in MWh
                total_battery_delivered = opt_res.q_bat_dis.sum() * (int(T[:-1]) /60)

                dict = {'filename': fn,
                        'E': float(E),
                        'P': float(P),
                        'Energy capacity': f'{E} MWh',
                        'Power capacity': f'{P} MW',
                        'Time resolution': int(T[:-1]),
                        'DC to AC': DC,
                        'Time labels': time_labels[T],
                        'Time granularity': time_grans[T],
                        'Remaining cap': remaining_cap,
                        'Energy not served': imbalance,
                        'Average SOC': opt_res.SOC.mean(),
                        'Total battery discharge': total_battery_delivered
                        }
                batteries.append(dict)
    batt_df = pd.DataFrame(batteries)
    batt_df['Size labels'] = list(zip(batt_df['Energy capacity'], batt_df['Power capacity']))
    batt_df['C-rate'] = batt_df['P']/batt_df['E']
    batt_df = batt_df.sort_values(by=['E', 'C-rate', 'Time resolution'], ascending=[True, False, False]).reset_index(drop=True)
    return batt_df
batt_df = set_up(results_folder, deg_results_folder, opt_results_folder, pattern)

def plot_heatmap_x_battery_size_y_time_resolutions(batt_df):
    temp_df = batt_df[['Remaining cap', 'E', 'P', 'Time resolution']].copy()
    newf = temp_df.pivot(columns=['E', 'P'], index='Time resolution')
    newf = newf.sort_index(ascending=False)

    # Plot on heatmap
    plt.figure(figsize=(9, 5))
    size_time_htmp = sns.heatmap(
        newf, xticklabels=[f"{E:g}MWh, {P:g}MW" for E, P in newf.columns.droplevel(0)])
    size_time_htmp.set(title='Remaining capacity after 15 years operation')

    plt.xticks(rotation=20, ha='right')  # , labels=newf.columns[1:])
    plt.ylabel('Time resolution of settlement (minutes)')
    plt.xlabel('Battery size (MWh/MW)')
    plt.tight_layout()
    plt.show()
if size_time_deg_heatmap:
    plot_heatmap_x_battery_size_y_time_resolutions(batt_df)


#
#     ## MWh MW ENS 1y and deg 15y plots
#     if MWh_MW_matrices:
#         # ENS
#         store_ENS, store_deg = {}, {}
#         ENS_MWh_MW, final_deg_MWh_MW, Tot_dis, Avg_SOC = {}, {}, {}, {}
#         for t in time_resolutions:
#             ENS_MWh_MW[t] = pd.DataFrame()
#             Tot_dis[t] = pd.DataFrame()
#             final_deg_MWh_MW[t] = pd.DataFrame()
#             Avg_SOC[t] = pd.DataFrame()
#             for e in bat_Emax:
#                 for p in bat_Pmax:
#                     if os.path.isfile(f'.\Opt_res\\{e}_{p}_{t}_Results.csv'):
#                         store_ENS[(t, e, p)] = pd.read_csv(f'.\Opt_res\\{e}_{p}_{t}_Results.csv')
#                         ENS_MWh_MW[t].loc[p, e] = store_ENS[(t, e, p)]['Imb_under'].sum() / store_ENS[(t, e, p)]['Bid'].sum()
#                         Tot_dis[t].loc[p, e] = store_ENS[(t, e, p)]['q_bat_dis'].sum() / time_grans[t]
#                         Avg_SOC[t].loc[p,e] = store_ENS[(t, e, p)]['SOC'].mean()
#                     elif os.path.isfile(f'.\Opt_res\\{e}_{p}_{t}_Results'):
#                         store_ENS[(t, e, p)] = pd.read_pickle(f'.\Opt_res\\{e}_{p}_{t}_Results')
#                         ENS_MWh_MW[t].loc[p, e] = store_ENS[(t, e, p)]['Imb_under'].sum() / store_ENS[(t, e, p)]['Bid'].sum()
#                         Tot_dis[t].loc[p, e] = store_ENS[(t, e, p)]['q_bat_dis'].sum() / time_grans[t]
#                         Avg_SOC[t].loc[p, e] = store_ENS[(t, e, p)]['SOC'].mean()
#                     else:
#                         ENS_MWh_MW[t].loc[p, e] = float('NaN')
#                         Tot_dis[t].loc[p, e] = float('NaN')
#                         Avg_SOC[t].loc[p, e] = float('NaN')
#
#                     if os.path.isfile(f'.\Results\\{e}_{p}_{t}_SOC.csv'):
#                         store_deg[(t, e, p)] = pd.read_csv(f'.\Results\\{e}_{p}_{t}_SOC.csv')
#                         final_deg_MWh_MW[t].loc[p, e] = store_deg[(t, e, p)]['Capacity_left'].iloc[-1]
#                     else:
#                         final_deg_MWh_MW[t].loc[p, e] = float('NaN')
#
#         print(final_deg_MWh_MW, ENS_MWh_MW)
#
#         # plot
#         for t in time_resolutions:
#             # fig, axs = plt.subplots(1,2, figsize=(15,12))
#             # sns.heatmap(final_deg_MWh_MW[t], annot=True, fmt='.4f', ax=axs[0,0])
#             # axs[0,0].set(title=f'Remaining capacity (after 15 years)')
#             # # plt.show()
#             # sns.heatmap(ENS_MWh_MW[t], annot=True, fmt='.4f', vmin = 1300, vmax = 2200, cmap = "Purples", ax=axs[0,1])
#             # axs[0,1].set(title=f'Percentage of energy not delivered')
#             # plt.show()
#             # sns.heatmap(Tot_dis[t], annot=True, fmt='.4f', cmap='Oranges', ax=axs[1,0])
#             # axs[1,0].set(title=f'Total discharge (over 1 year)')
#             # # plt.show()
#             # sns.heatmap(Avg_SOC[t], annot=True, fmt='.4f', cmap='Blues', ax=axs[1,1])
#             # axs[1,1].set(title=f'Average SOC')
#
#             fig, axs = plt.subplots(1, 2, figsize=(12, 3))
#             sns.heatmap(final_deg_MWh_MW[t], annot=True, fmt='.4f', ax=axs[0], vmin = 82, vmax = 83.8, cmap='Oranges_r')
#             # sns.cubehelix_palette(as_cmap=True, reverse=True)
#             axs[0].set(title=f'Remaining capacity (after 15 years)', xlabel='Battery E cap.')
#             # plt.show()
#             sns.heatmap(ENS_MWh_MW[t], annot=True, fmt='.4f', cmap="Purples", ax=axs[1], vmin=0.13, vmax=0.185)
#             axs[1].set(title=f'Percentage of energy not delivered', xlabel='Battery E cap.')
#
#             for ax in axs.flat:
#                 ax.set_xticks(ax.get_xticks())  # Ensure the ticks are present
#                 ax.tick_params(rotation=15)
#                 ax.set(xlabel='Battery E cap.', ylabel='Battery P cap')#, xticklabels=bat_Emax, yticklabels=bat_Pmax)
#             # axs[0].set_xlabel('')
#             # axs[0, 1].set_xlabel('')
#
#             # for ax in axs.flat:
#             fig.suptitle(f'Time: {time_resolution_names[t]}')
#             plt.show()
#
#
#     if SOC_plots:
#         SOCs = {}
#         plot_SOC = {}
#         for e in bat_Emax:
#             for p in bat_Pmax:
#                 plot_SOC[(e,p)] = pd.DataFrame()
#                 for t in time_resolutions:
#                     if os.path.isfile(f'Opt_res\{e}_{p}_{t}_Results.csv'):
#                         store_df = pd.read_csv(f'Opt_res\{e}_{p}_{t}_Results.csv', index_col=0, parse_dates=True)
#                         SOCs[(e,p,t)] = store_df['SOC']
#                         plot_SOC[(e, p)][t] = store_df['SOC']
#                         plot_SOC[(e, p)][t] = plot_SOC[(e, p)][t].ffill().bfill()
#                     elif os.path.isfile(f'Opt_res\{e}_{p}_{t}_Results'):
#                         store_df = pd.read_pickle(f'Opt_res\{e}_{p}_{t}_Results')
#                         SOCs[(e,p,t)] = store_df['SOC']
#                         plot_SOC[(e, p)][t] = store_df['SOC']
#                         plot_SOC[(e, p)][t] = plot_SOC[(e, p)][t].ffill().bfill()
#                     else:
#                         continue
#
#                 if plot_SOC[(e, p)].empty:
#                     del plot_SOC[(e, p)]
#                 # else:
#                 #     plot_SOC[(e, p)].plot()
#                 #     plt.title(f'Battery size {e}, {p}')
#                 #     plt.show()
#
# # final_deg_MWh_MW.to_excel('.\Heatmaps\heatmaps_dfs.xlsx', sheet_name='final_deg_by_size')
#
# if Cycles_plots:
#     cycles = {}
#     avg_SOC = {}
#     avg_DoD = {}
#     no_cycles = {}
#     for e in bat_Emax:
#         for p in bat_Pmax:
#             plot_cycles = {}
#             for t in time_resolutions:
#                 if (e,p,t) in SOCs.keys():
#                     plot=1
#                     cycles[(e,p,t)] = rainflow.count_cycles(SOCs[(e,p,t)], binsize=0.1)
#                     avg_SOC[(e,p,t)] = SOCs[(e,p,t)].mean()
#                     # plot_cycles[t] = rainflow.count_cycles(SOCs[e,p,t], binsize=0.1)
#                     cyc_depth, freq, DoD = [], [], []
#                     for item in cycles[e,p,t]:
#                         cyc_depth.append(item[0])
#                         freq.append(item[1])
#                         DoD.append(item[0]*item[1])
#                     no_cycles[(e,p,t)] = sum(freq)
#                     avg_DoD[(e,p,t)] = sum(DoD)/no_cycles[(e,p,t)]
#                     x_axis = np.arange(len(cyc_depth))
#                     plt.bar((x_axis + (time_resolutions.index(t)*0.2)), freq, label=f"Time: {t}. Avg SOC: {round(avg_SOC[(e,p,t)],2)}. No. cycles: {sum(freq)}", width=0.2)
#                 else:
#                     plot=0
#             if plot == 1:
#                 plt.xticks(x_axis, ['%.2f' % elem for elem in cyc_depth])
#                 plt.xlabel('Depth of discharge (relative to SOC)')
#                 plt.ylabel('Frequency')
#                 plt.legend()
#                 plt.title(f'{e,p}')
#                 plt.show()
#
# if battery_plots:
#
#     final_deg = {}
#     bat_deg, dodavg, socavg, cycleno, distot = {}, {}, {}, {}, {}
#     for e in bat_Emax:
#         for p in bat_Pmax:
#             bat_deg[e, p], dodavg[e, p], socavg[e, p], cycleno[e, p], distot[e, p] = pd.Series(dtype=float), \
#                 pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
#             for t in time_resolutions:
#                 # final_deg[(e,p,t)] = final_deg_MWh_MW[t].loc[p,e]
#                 if (e,p,t) in SOCs.keys():
#                     bat_deg[e,p][t] = final_deg_MWh_MW[t].loc[p,e]
#                     dodavg[e,p][t] = avg_DoD[(e,p,t)]
#                     socavg[e,p][t] = avg_SOC[(e,p,t)]
#                     distot[e,p][t] = Tot_dis[t].loc[p,e]
#                     cycleno[e,p][t] = no_cycles[(e,p,t)]
#                     print(bat_deg[e,p][t],
#                     dodavg[e,p][t],
#                     socavg[e,p][t],
#                     distot[e,p][t],
#                     cycleno[e,p][t])
#                 else:
#                     continue
#             # bat_deg[e,p].plot.bar(x=bat_deg[e,p].index)
#             if bat_deg[e,p].empty:
#                 del bat_deg[e,p]
#             else:
#                 fig, ax1 = plt.subplots(figsize=(12, 3))
#
#                 ax1.bar(bat_deg[e,p].index, bat_deg[e,p], label = 'Remaining capacity')
#                 # ax1.set_ylim((min(bat_deg[e,p])-0.25, max(bat_deg[e,p])+0.5))
#                 ax1.set_ylabel('Degradation from bar chart')
#                 # plt.xlabel(
#                 ax1.set_xlabel('Time intervals of settlement (mins)')
#
#                 # Create secondary y-axes for the lines
#                 ax2 = ax1.twinx()
#                 ax3 = ax1.twinx()
#                 ax4 = ax1.twinx()
#                 ax5 = ax1.twinx()
#
#                 # Move the spine of the additional axes
#                 ax3.spines['right'].set_position(('outward', 50))
#                 ax4.spines['right'].set_position(('outward', 100))
#                 ax5.spines['right'].set_position(('outward', 150))
#
#                 # Plot lines on separate y-axes
#                 ax2.plot(dodavg[e,p].index, dodavg[e,p], color='green', label='Average DoD', linestyle='--')
#                 ax3.plot(socavg[e,p].index, socavg[e,p], color='red', label='Average SOC', linestyle='-.')
#                 ax4.plot(distot[e,p].index, distot[e,p], color='purple', label='Total discharge (MWh)', linestyle=':')
#                 ax5.plot(cycleno[e,p].index, cycleno[e,p], color='orange', label='Number of cycles', linestyle='-')
#
#                 # Set labels and tick colors for each y-axis
#                 ax2.set_ylabel('Depth fo discharge', color='green')
#                 # ax2.tick_params('y', colors='green')
#
#                 ax3.set_ylabel('SOC', color='red')
#                 # ax3.tick_params('y', colors='red')
#
#                 ax4.set_ylabel('Discharge total (MWh)', color='purple')
#                 # ax4.tick_params('y', colors='purple')
#
#                 ax5.set_ylabel('no. cycles', color='orange')
#                 # ax5.tick_params('y', colors='orange')
#
#                 # Adding legend
#                 lines, labels = ax1.get_legend_handles_labels()
#                 lines2, labels2 = ax2.get_legend_handles_labels()
#                 lines3, labels3 = ax3.get_legend_handles_labels()
#                 lines4, labels4 = ax4.get_legend_handles_labels()
#                 lines5, labels5 = ax5.get_legend_handles_labels()
#
#                 ax1.legend(lines + lines2 + lines3 + lines4 + lines5,
#                            labels + labels2 + labels3 + labels4 + labels5,
#                            loc='upper left')
#
#                 # Show the plot
#                 plt.title(f'{e, p}')
#                 plt.xlabel('Time intervals of settlement (mins)')
#                 plt.tight_layout()
#                 plt.show()
#
#
#
# #
# # import os
# # import pandas as pd
# # import matplotlib.pyplot as plt
# #
# # # Define configurable parameters
# # RESULTS_DIR = 'Results'  # Directory containing CSV files
# # TIME_RESOLUTION = 1  # Time resolution in years (adjust as needed)
# #
# # def read_data_from_csv(filename):
# #     """Read data from a CSV file and return a DataFrame."""
# #     filepath = os.path.join(RESULTS_DIR, filename)
# #     return pd.read_csv(filepath)
# #
# # def calculate_metrics(data):
# #     """Calculate relevant metrics based on the data."""
# #     # Example calculations (customize as per your requirements)
# #     remaining_capacity = data['Capacity_left'].iloc[-1]
# #     energy_not_delivered = data['Energy'].sum()
# #     total_discharge = data['Discharge'].sum()
# #     average_soc = data['SOC'].mean()
# #     return remaining_capacity, energy_not_delivered, total_discharge, average_soc
# #
# # def create_visualizations(data):
# #     """Create relevant plots for visualization."""
# #     # Example: Create a bar chart showing remaining capacity
# #     plt.figure(figsize=(8, 6))
# #     plt.bar(data['Time'], data['Capacity_left'], color='skyblue')
# #     plt.xlabel('Time (years)')
# #     plt.ylabel('Remaining Capacity')
# #     plt.title('Battery Remaining Capacity Over Time')
# #     plt.grid(True)
# #     plt.show()
# #
# # def main():
# #     # Get list of CSV files in the Results directory
# #     csv_files = [file for file in os.listdir(RESULTS_DIR) if file.lower().endswith('.csv')]
# #
# #     for filename in csv_files:
# #         data = read_data_from_csv(filename)
# #         remaining_capacity, energy_not_delivered, total_discharge, average_soc = calculate_metrics(data)
# #
# #         print(f"Metrics for {filename}:")
# #         print(f"Remaining Capacity: {remaining_capacity:.2f} Wh")
# #         print(f"Energy Not Delivered: {energy_not_delivered:.2f} Wh")
# #         print(f"Total Discharge: {total_discharge:.2f} Wh")
# #         print(f"Average SOC: {average_soc:.2f}")
# #
# #         create_visualizations(data)
# #
# # if __name__ == '__main__':
# #     main()
