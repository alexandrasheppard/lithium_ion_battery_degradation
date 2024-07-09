import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import matplotlib as mpl
import seaborn as sns
import rainflow

plotting = True
size_time_deg_matrix = True
MWh_MW_matrices = True
SOC_plots = True
Cycles_plots = 1
battery_plots = True
AC_to_DC = '1.1'


time_resolutions = ['30T', '15T', '5T', '1T']
time_resolution_names = {'1T': '1 min', '5T': '5 min', '15T': '15 min', '30T': '30 min'}
time_grans = {'1T': 60, '5T': 12, '15T': 4, '30T': 2}
bat_Emax = [1, 1,2,2,4,4]
bat_Pmax = [1, 0.5, 2, 1, 4, 2]

files = []
for m, fn in enumerate(os.listdir('Results')):
    if os.path.isfile(os.path.join('Results', str(fn))):
        filename, file_extension = os.path.splitext(fn)
        if file_extension == '.csv':
            files.append(fn)

print(files)

if plotting:
    if size_time_deg_matrix:
        store_d = {}
        final_deg_size_time = pd.DataFrame()

        for i in range(len(bat_Emax)):
            ikey = f'{bat_Emax[i]}MWh_{bat_Pmax[i]}MW'
            for k in time_resolutions:
                store_d[(ikey,k)] = pd.read_csv(f'.\Results\\No_cal_deg_{ikey}_{k}_{AC_to_DC}_SOC.csv')
                final_deg_size_time.loc[k, ikey] = store_d[(ikey, k)]['Capacity_left'].iloc[-1]

        print(final_deg_size_time)
        ax = sns.heatmap(final_deg_size_time, annot=True, fmt='.2f')#, figsize=(10,4))#, yticklabels=time_resolution_names) #xticklabels=keys,
        ax.set(xlabel='Battery size/type', ylabel='Time resolution of settlement (mins)', title='Remaining capacity after 15 years operation')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()
#
#         # final_deg_size_time.to_excel('.\Heatmaps\heatmaps_dfs.xlsx', sheet_name='final_deg_by_size')
#
    ## MWh MW ENS 1y and deg 15y plots
    # if MWh_MW_matrices:
    #     # ENS
    #     store_ENS, store_deg = {}, {}
    #     ENS_MWh_MW, final_deg_MWh_MW, Tot_dis, Avg_SOC = {}, {}, {}, {}
    #     for t in time_resolutions:
    #         ENS_MWh_MW[t] = pd.DataFrame()
    #         Tot_dis[t] = pd.DataFrame()
    #         final_deg_MWh_MW[t] = pd.DataFrame()
    #         Avg_SOC[t] = pd.DataFrame()
    #         for e in bat_Emax:
    #             for p in bat_Pmax:
    #                 if os.path.isfile(f'.\Opt_res\{e}_{p}_{t}_Results.csv'):
    #                     store_ENS[(t, e, p)] = pd.read_csv(f'.\Opt_res\{e}_{p}_{t}_Results.csv')
    #                     ENS_MWh_MW[t].loc[p, e] = store_ENS[(t, e, p)]['Imb_under'].sum() / store_ENS[(t, e, p)]['Bid'].sum()
    #                     Tot_dis[t].loc[p, e] = store_ENS[(t, e, p)]['q_bat_dis'].sum() / time_grans[t]
    #                     Avg_SOC[t].loc[p,e] = store_ENS[(t, e, p)]['SOC'].mean()
    #                 elif os.path.isfile(f'.\Opt_res\{e}_{p}_{t}_Results'):
    #                     store_ENS[(t, e, p)] = pd.read_pickle(f'.\Opt_res\{e}_{p}_{t}_Results')
    #                     ENS_MWh_MW[t].loc[p, e] = store_ENS[(t, e, p)]['Imb_under'].sum() / store_ENS[(t, e, p)]['Bid'].sum()
    #                     Tot_dis[t].loc[p, e] = store_ENS[(t, e, p)]['q_bat_dis'].sum() / time_grans[t]
    #                     Avg_SOC[t].loc[p, e] = store_ENS[(t, e, p)]['SOC'].mean()
    #                 else:
    #                     ENS_MWh_MW[t].loc[p, e] = float('NaN')
    #                     Tot_dis[t].loc[p, e] = float('NaN')
    #                     Avg_SOC[t].loc[p, e] = float('NaN')
    #
    #                 if os.path.isfile(f'.\Results\{e}_{p}_{t}_SOC.csv'):
    #                     store_deg[(t, e, p)] = pd.read_csv(f'.\Results\{e}_{p}_{t}_SOC.csv')
    #                     final_deg_MWh_MW[t].loc[p, e] = store_deg[(t, e, p)]['Capacity_left'].iloc[-1]
    #                 else:
    #                     final_deg_MWh_MW[t].loc[p, e] = float('NaN')
    #
    #     print(final_deg_MWh_MW, ENS_MWh_MW)
    #
    #     # plot
    #     for t in time_resolutions:
    #         # fig, axs = plt.subplots(1,2, figsize=(15,12))
    #         # sns.heatmap(final_deg_MWh_MW[t], annot=True, fmt='.4f', ax=axs[0,0])
    #         # axs[0,0].set(title=f'Remaining capacity (after 15 years)')
    #         # # plt.show()
    #         # sns.heatmap(ENS_MWh_MW[t], annot=True, fmt='.4f', vmin = 1300, vmax = 2200, cmap = "Purples", ax=axs[0,1])
    #         # axs[0,1].set(title=f'Percentage of energy not delivered')
    #         # plt.show()
    #         # sns.heatmap(Tot_dis[t], annot=True, fmt='.4f', cmap='Oranges', ax=axs[1,0])
    #         # axs[1,0].set(title=f'Total discharge (over 1 year)')
    #         # # plt.show()
    #         # sns.heatmap(Avg_SOC[t], annot=True, fmt='.4f', cmap='Blues', ax=axs[1,1])
    #         # axs[1,1].set(title=f'Average SOC')
    #
    #         fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    #         sns.heatmap(final_deg_MWh_MW[t], annot=True, fmt='.4f', ax=axs[0], vmin = 82, vmax = 83.8, cmap='Oranges_r')
    #         # sns.cubehelix_palette(as_cmap=True, reverse=True)
    #         axs[0].set(title=f'Remaining capacity (after 15 years)', xlabel='Battery E cap.')
    #         # plt.show()
    #         sns.heatmap(ENS_MWh_MW[t], annot=True, fmt='.4f', cmap="Purples", ax=axs[1], vmin=0.13, vmax=0.185)
    #         axs[1].set(title=f'Percentage of energy not delivered', xlabel='Battery E cap.')
    #
    #         for ax in axs.flat:
    #             ax.set_xticks(ax.get_xticks())  # Ensure the ticks are present
    #             ax.tick_params(rotation=15)
    #             ax.set(xlabel='Battery E cap.', ylabel='Battery P cap')#, xticklabels=bat_Emax, yticklabels=bat_Pmax)
    #         # axs[0].set_xlabel('')
    #         # axs[0, 1].set_xlabel('')
    #
    #         # for ax in axs.flat:
    #         fig.suptitle(f'Time: {time_resolution_names[t]}')
    #         plt.show()
#
#
    # if SOC_plots:
    #     SOCs = {}
    #     plot_SOC = {}
    #     for e in bat_Emax:
    #         for p in bat_Pmax:
    #             plot_SOC[(e,p)] = pd.DataFrame()
    #             for t in time_resolutions:
    #                 if os.path.isfile(f'Opt_res\{e}_{p}_{t}_Results.csv'):
    #                     store_df = pd.read_csv(f'Opt_res\{e}_{p}_{t}_Results.csv', index_col=0, parse_dates=True)
    #                     SOCs[(e,p,t)] = store_df['SOC']
    #                     plot_SOC[(e, p)][t] = store_df['SOC']
    #                     plot_SOC[(e, p)][t] = plot_SOC[(e, p)][t].ffill().bfill()
    #                 elif os.path.isfile(f'Opt_res\{e}_{p}_{t}_Results'):
    #                     store_df = pd.read_pickle(f'Opt_res\{e}_{p}_{t}_Results')
    #                     SOCs[(e,p,t)] = store_df['SOC']
    #                     plot_SOC[(e, p)][t] = store_df['SOC']
    #                     plot_SOC[(e, p)][t] = plot_SOC[(e, p)][t].ffill().bfill()
    #                 else:
    #                     continue
    #
    #             if plot_SOC[(e, p)].empty:
    #                 del plot_SOC[(e, p)]
    #             # else:
    #             #     plot_SOC[(e, p)].plot()
    #             #     plt.title(f'Battery size {e}, {p}')
    #             #     plt.show()

# final_deg_MWh_MW.to_excel('.\Heatmaps\heatmaps_dfs.xlsx', sheet_name='final_deg_by_size')
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
if battery_plots:

    final_deg = {}
    bat_deg, dodavg, socavg, cycleno, distot = {}, {}, {}, {}, {}
    for e in bat_Emax:
        for p in bat_Pmax:
            bat_deg[e, p], dodavg[e, p], socavg[e, p], cycleno[e, p], distot[e, p] = pd.Series(dtype=float), \
                pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
            for t in time_resolutions:
                # final_deg[(e,p,t)] = final_deg_MWh_MW[t].loc[p,e]
                if (e,p,t) in SOCs.keys():
                    bat_deg[e,p][t] = final_deg_MWh_MW[t].loc[p,e]
                    dodavg[e,p][t] = avg_DoD[(e,p,t)]
                    socavg[e,p][t] = avg_SOC[(e,p,t)]
                    distot[e,p][t] = Tot_dis[t].loc[p,e]
                    cycleno[e,p][t] = no_cycles[(e,p,t)]
                    print(bat_deg[e,p][t],
                    dodavg[e,p][t],
                    socavg[e,p][t],
                    distot[e,p][t],
                    cycleno[e,p][t])
                else:
                    continue
            # bat_deg[e,p].plot.bar(x=bat_deg[e,p].index)
            if bat_deg[e,p].empty:
                del bat_deg[e,p]
            else:
                fig, ax1 = plt.subplots(figsize=(12, 3))

                ax1.bar(bat_deg[e,p].index, bat_deg[e,p], label = 'Remaining capacity')
                # ax1.set_ylim((min(bat_deg[e,p])-0.25, max(bat_deg[e,p])+0.5))
                ax1.set_ylabel('Degradation from bar chart')
                # plt.xlabel(
                ax1.set_xlabel('Time intervals of settlement (mins)')

                # Create secondary y-axes for the lines
                ax2 = ax1.twinx()
                ax3 = ax1.twinx()
                ax4 = ax1.twinx()
                ax5 = ax1.twinx()

                # Move the spine of the additional axes
                ax3.spines['right'].set_position(('outward', 50))
                ax4.spines['right'].set_position(('outward', 100))
                ax5.spines['right'].set_position(('outward', 150))

                # Plot lines on separate y-axes
                ax2.plot(dodavg[e,p].index, dodavg[e,p], color='green', label='Average DoD', linestyle='--')
                ax3.plot(socavg[e,p].index, socavg[e,p], color='red', label='Average SOC', linestyle='-.')
                ax4.plot(distot[e,p].index, distot[e,p], color='purple', label='Total discharge (MWh)', linestyle=':')
                ax5.plot(cycleno[e,p].index, cycleno[e,p], color='orange', label='Number of cycles', linestyle='-')

                # Set labels and tick colors for each y-axis
                ax2.set_ylabel('Depth fo discharge', color='green')
                # ax2.tick_params('y', colors='green')

                ax3.set_ylabel('SOC', color='red')
                # ax3.tick_params('y', colors='red')

                ax4.set_ylabel('Discharge total (MWh)', color='purple')
                # ax4.tick_params('y', colors='purple')

                ax5.set_ylabel('no. cycles', color='orange')
                # ax5.tick_params('y', colors='orange')

                # Adding legend
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines3, labels3 = ax3.get_legend_handles_labels()
                lines4, labels4 = ax4.get_legend_handles_labels()
                lines5, labels5 = ax5.get_legend_handles_labels()

                ax1.legend(lines + lines2 + lines3 + lines4 + lines5,
                           labels + labels2 + labels3 + labels4 + labels5,
                           loc='upper left')

                # Show the plot
                plt.title(f'{e, p}')
                plt.xlabel('Time intervals of settlement (mins)')
                plt.tight_layout()
                plt.show()
#



















#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Define configurable parameters
# RESULTS_DIR = 'Results'  # Directory containing CSV files
# TIME_RESOLUTION = 1  # Time resolution in years (adjust as needed)
#
# def read_data_from_csv(filename):
#     """Read data from a CSV file and return a DataFrame."""
#     filepath = os.path.join(RESULTS_DIR, filename)
#     return pd.read_csv(filepath)
#
# def calculate_metrics(data):
#     """Calculate relevant metrics based on the data."""
#     # Example calculations (customize as per your requirements)
#     remaining_capacity = data['Capacity_left'].iloc[-1]
#     energy_not_delivered = data['Energy'].sum()
#     total_discharge = data['Discharge'].sum()
#     average_soc = data['SOC'].mean()
#     return remaining_capacity, energy_not_delivered, total_discharge, average_soc
#
# def create_visualizations(data):
#     """Create relevant plots for visualization."""
#     # Example: Create a bar chart showing remaining capacity
#     plt.figure(figsize=(8, 6))
#     plt.bar(data['Time'], data['Capacity_left'], color='skyblue')
#     plt.xlabel('Time (years)')
#     plt.ylabel('Remaining Capacity')
#     plt.title('Battery Remaining Capacity Over Time')
#     plt.grid(True)
#     plt.show()
#
# def main():
#     # Get list of CSV files in the Results directory
#     csv_files = [file for file in os.listdir(RESULTS_DIR) if file.lower().endswith('.csv')]
#
#     for filename in csv_files:
#         data = read_data_from_csv(filename)
#         remaining_capacity, energy_not_delivered, total_discharge, average_soc = calculate_metrics(data)
#
#         print(f"Metrics for {filename}:")
#         print(f"Remaining Capacity: {remaining_capacity:.2f} Wh")
#         print(f"Energy Not Delivered: {energy_not_delivered:.2f} Wh")
#         print(f"Total Discharge: {total_discharge:.2f} Wh")
#         print(f"Average SOC: {average_soc:.2f}")
#
#         create_visualizations(data)
#
# if __name__ == '__main__':
#     main()
