import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import math
import matplotlib as mpl
import seaborn as sns
import rainflow
import matplotlib.dates as mdates
import datetime
from datetime import date, datetime
from dateutil import relativedelta

# Location of input files
# results_folder = r'M:\Documents\PhD\Code\PhD_Projects\IFE_degradation\Results\July_prelim_v1'
results_folder = r'M:\Documents\PhD\Code\PhD_Projects\IFE_degradation\Results\20240806140146\Matric_filling_july_10'
deg_results_folder = r'Deg_model_results'
opt_results_folder = r'Opt_model_results'
start_date = datetime.strptime('2016-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
lifetime_percent = 80
# Naming system for degradation results files
pattern = re.compile(r'(\d+.?\d*)MWh_(\d+.?\d*)MW_(\d+T)_(\d+.\d+)_SOC.csv')

size_time_deg_heatmap = True
battery_plots = False
plot_cycle_data = False
SOC_limits_calculate = False

# Set SOC limits to count above/below
SOC_count_limits = {'lower':0.2,
                    'upper':0.8}

# for plotting P vs E heatmaps for several factors
# factors = []
factors = ['Remaining cap', 'Average SOC', 'Average DOD', 'Number of cycles', 'Total battery discharge', 'Lifetime']
            #'Total battery discharge', 'Energy not served',
desired_heatmaps = {'E':{'x':'Storage duration', 'y':'Time resolution'},
                    'Storage duration':{'x':'E', 'y':'Time resolution'},
                    'Time resolution':{'x':'E', 'y':'Storage duration'}}
if SOC_limits_calculate:
    factors.extend([f'SOC gt {SOC_count_limits['upper']}', f'SOC lt {SOC_count_limits['lower']}'])
colours = {'Remaining cap':'rocket',
           'Average SOC':'Greens',
           'Average DOD':'Blues',
           'Energy not served':'Reds',
           'Number of cycles':'YlOrBr',
           'Total battery discharge':'Purples',
           f'SOC gt {SOC_count_limits['upper']}':'Greys',
           f'SOC lt {SOC_count_limits['lower']}': 'Greys',
           'Lifetime': 'Oranges'}
colour = {'Remaining cap':'C5',
           'Average SOC':'C2',
           'Average DOD':'C0',
           'Energy not served':'C3',
           'Number of cycles':'C1',
           'Total battery discharge':'C4',
           f'SOC gt {SOC_count_limits['upper']}':'C7',
           f'SOC lt {SOC_count_limits['lower']}': 'black'}

def set_up(results_folder, deg_results_folder, opt_results_folder, pattern, lifetime_percent=80):
    deg_dir = os.path.join(results_folder, deg_results_folder)
    opt_dir = os.path.join(results_folder, opt_results_folder)
    time_labels = {'1T': '1 min', '5T': '5 min', '15T': '15 min', '30T': '30 min'}
    time_grans = {'1T': 60, '5T': 12, '15T': 4, '30T': 2}

    batteries = []
    SOC_profiles = {}
    for m, fn in enumerate(os.listdir(deg_dir)):
        if os.path.isfile(os.path.join(deg_dir, str(fn))):
            filename, file_extension = os.path.splitext(fn)
            opt_fn = fn.replace('SOC', 'Results')
            if file_extension == '.csv':

                # Degradation results
                deg_res = pd.read_csv(os.path.join(deg_dir, fn), index_col='time', usecols=['time', 'Capacity_left'])
                remaining_cap = deg_res['Capacity_left'].iloc[-1]
                date_of_lifetime = deg_res[deg_res['Capacity_left']<lifetime_percent]['Capacity_left'].index[0]
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
                        'Date of lifetime': date_of_lifetime,
                        'Energy not served': imbalance,
                        'Average SOC': opt_res.SOC.mean(),
                        'Total battery discharge': total_battery_delivered
                        }
                SOC_profiles[(dict['E'], dict['P'], dict['Time resolution'])] = opt_res.SOC
                batteries.append(dict)
    batt_df = pd.DataFrame(batteries)
    batt_df['Size labels'] = list(zip(batt_df['Energy capacity'], batt_df['Power capacity']))
    batt_df['Case labels'] = list(zip(batt_df['Energy capacity'], batt_df['Power capacity'], batt_df['Time resolution']))
    s_dur_names = {0.5:'2 h', 1:'1 h', 2:'0.5 h', 4:'0.25 h'}
    batt_df['Storage duration'] = batt_df['P'] / batt_df['E']
    for i in range(len(batt_df)):
        batt_df.loc[i, 'Storage_duration_name'] = s_dur_names[batt_df.loc[i, 'Storage duration']]
    batt_df = batt_df.sort_values(by=['E', 'Storage duration', 'Time resolution'], ascending=[True, False, False]).reset_index(drop=True)

    # Calculate lifetime
    for i in range(len(batt_df)):
        delta = relativedelta.relativedelta(datetime.strptime(
            batt_df['Date of lifetime'][i], "%Y-%m-%d %H:%M:%S+00:00"),
            start_date)
        months = delta.years * 12 + delta.months + round(delta.days / 30)
        batt_df.loc[i, 'Lifetime'] = months / 12

    return batt_df, SOC_profiles
batt_df, SOC_profiles = set_up(results_folder, deg_results_folder, opt_results_folder, pattern, lifetime_percent)

# Plotting the overall degradation heatmap, time vs size
def plot_heatmap_deg_x_battery_size_y_time_resolutions(batt_df):
    temp_df = batt_df[['Remaining cap', 'E', 'P', 'Time resolution']].copy()
    newf = temp_df.pivot(columns=['E', 'P'], index='Time resolution')
    newf = newf.sort_index(ascending=False)
    vmin, vmax = newf.min().min(), newf.max().max()

    # Plot on heatmap
    plt.figure(figsize=(11, 4))
    size_time_htmp = sns.heatmap(newf,
                                 vmin=vmin,
                                 vmax=vmax,
                                 xticklabels=[f"{E:g}MWh, {P:g}MW" for E, P in newf.columns.droplevel(0)])
    size_time_htmp.set(title='Remaining capacity after 15 years operation')

    plt.xticks(rotation=20, ha='right')  # , labels=newf.columns[1:])
    plt.ylabel('Time resolution of settlement (minutes)')
    plt.xlabel('Battery size (MWh/MW)')
    plt.tight_layout()
    save_dir = r'M:\Documents\PhD\Papers\EUPVSEC2024\Conference proceedings figures\Draft 2'
    plt.savefig(f'{save_dir}\\Remaining_capacities.png')
    plt.show()
if size_time_deg_heatmap:
    plot_heatmap_deg_x_battery_size_y_time_resolutions(batt_df)

def find_vmin_vmax(batt_df, factors):
    v_mins_maxs = {}
    for factor in list(dict.fromkeys(factors)):
        v_mins_maxs[factor] = {'min': batt_df[factor].min(),
                               'max': batt_df[factor].max()}
    return v_mins_maxs

# Plotting heatmaps for various factors and time resolutions, P vs E heatmap matrices
def make_subplot_dimensions(nrows, no_subplots):
    axs_indexes = []
    if nrows >1:
        ncols = math.ceil(no_subplots / nrows)
        cols = list(range(ncols)) * nrows
        rows = [0] * ncols + [1] * ncols
        for i in range(no_subplots):
            axs_indexes.append((rows[i], cols[i]))
    else:
        ncols = no_subplots
        axs_indexes = list(range(no_subplots))
    return nrows, ncols, axs_indexes
def plot_heatmap_subplot(df, factor, axs_pos, colour, pivot_index='P', pivot_columns='E',  vmins_and_maxs=None, title='Untitled'):
    pivot = df.pivot(index=pivot_index, columns=pivot_columns, values=factor)
    if vmins_and_maxs:
        sns.heatmap(pivot, annot=True, fmt='.2f',
                    ax=axs_pos,
                    vmin=vmins_and_maxs[factor]['min'],
                    vmax=vmins_and_maxs[factor]['max'],
                    cmap=colour)
    else:
        sns.heatmap(pivot, annot=True, fmt='.2f',
                    ax=axs_pos,
                    cmap=colour)
    axs_pos.set_title(title)

def plot_factor_heatmaps(separating_factor, x_axis_factor, y_axis_factor, batt_df, factors, colours, nrows=1):
    separating_factors = batt_df[separating_factor].unique()
    if separating_factor in factors:
        factors.remove(separating_factor)
    v_mins_maxs = find_vmin_vmax(batt_df, factors)

    for f in range(len(factors)):
        # Set up subplots for correct number of time resolutions
        nrows, ncols, axs_indexes = make_subplot_dimensions(nrows, len(separating_factors))

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
        # cbar_ax = fig.add_axes([.91, .3, .03, .4])

        for i in range(len(separating_factors)):
            # Filter data for time resolution
            s_fact = separating_factors[i]
            filtered_df = batt_df[batt_df[separating_factor] == s_fact]

            # Plot subplot
            plot_heatmap_subplot(df=filtered_df,
                                 factor=factors[f],
                                 axs_pos=axs[axs_indexes[i]],
                                 colour=colours[factors[f]],
                                 pivot_index=y_axis_factor,  # Edited
                                 pivot_columns=x_axis_factor,  # Edited
                                 vmins_and_maxs=v_mins_maxs,
                                 title=f"{separating_factor}: {s_fact:g}")

        # Set shared axis labels and title
        # fig.supxlabel('Storage_duration_name')
        # fig.supylabel('T')
        plt.suptitle(f'{factors[f]}')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the main title
        save_dir = r'M:\Documents\PhD\Papers\EUPVSEC2024\Conference proceedings figures\Draft 2'
        plt.savefig(f'{save_dir}\\x4_{separating_factor}_{factors[f]}.png')
        plt.show()

# Three heatmaps of E cap, x axis SD, y axis TR
for key in desired_heatmaps.keys():
    plot_factor_heatmaps(key, desired_heatmaps[key]['x'], desired_heatmaps[key]['y'], batt_df, factors, colours)


# Update battery dataframe with number of cycles and average DoD
def count_cycles(batt_df, SOC_profiles, cycle_binsize=0.1):
    cycle_data = {}
    for key in SOC_profiles.keys():
        e,p,t = key
        cycles = rainflow.count_cycles(SOC_profiles[key], binsize=cycle_binsize)
        cyc_depth, freq, DoD_numerator = [], [], []
        for item in cycles:
            cyc_depth.append(item[0])
            freq.append(item[1])
            DoD_numerator.append(item[0]*item[1])
        no_cycles = sum(freq)
        avg_DoD = sum(DoD_numerator)/no_cycles
        cycle_data[key] = {'no_cycles': no_cycles,
                            'avg_DoD': avg_DoD,
                            'cyc_depth': cyc_depth,
                            'freq': freq}
        assert len(batt_df[(batt_df['E'] == e) &
                           (batt_df['P'] == p) &
                           (batt_df['Time resolution'] == t)]) == 1, \
            'Error in filtering dataframe - there is more than one battery for the given SOC key'
        row_index = batt_df[(batt_df['E'] == e) &
                           (batt_df['P'] == p) &
                           (batt_df['Time resolution'] == t)
                            ].index.values[0]
        batt_df.loc[row_index, 'Number of cycles'] = no_cycles
        batt_df.loc[row_index, 'Average DOD'] = avg_DoD
    return batt_df, cycle_data
batt_df, cycle_data = count_cycles(batt_df, SOC_profiles, cycle_binsize=0.1)

def count_SOC_limit_excedences(batt_df, SOC_profiles, SOC_count_limits={'lower':0.2, 'upper':0.8}):
    for key in SOC_profiles.keys():
        e,p,t = key
        soc_ser = SOC_profiles[key].copy()
        upper_limit = SOC_count_limits['upper']
        lower_limit = SOC_count_limits['lower']
        maxs = []
        mins = []
        for i in range(len(soc_ser) - 1):
            if soc_ser.iloc[i + 1] > soc_ser.iloc[i] and soc_ser.iloc[i + 1] == soc_ser.iloc[i + 2]:
                maxs.append(soc_ser.iloc[i + 1])
            elif soc_ser.iloc[i + 1] > soc_ser.iloc[i] and soc_ser.iloc[i + 1] > soc_ser.iloc[i + 2]:
                maxs.append(soc_ser.iloc[i + 1])
            elif soc_ser.iloc[i + 1] < soc_ser.iloc[i] and soc_ser.iloc[i + 1] == soc_ser.iloc[i + 2]:
                mins.append(soc_ser.iloc[i + 1])
            elif soc_ser.iloc[i + 1] < soc_ser.iloc[i] and soc_ser.iloc[i + 1] < soc_ser.iloc[i + 2]:
                mins.append(soc_ser.iloc[i + 1])

        count_over = len([n for n in maxs if n > 0.8])
        count_under = len([n for n in mins if n< 0.2])

        assert len(batt_df[(batt_df['E'] == e) &
                           (batt_df['P'] == p) &
                           (batt_df['Time resolution'] == t)]) == 1, \
            'Error in filtering dataframe - there is more than one battery for the given SOC key'
        row_index = batt_df[(batt_df['E'] == e) &
                           (batt_df['P'] == p) &
                           (batt_df['Time resolution'] == t)
                            ].index.values[0]
        batt_df.loc[row_index, f'SOC gt {upper_limit}'] = count_over
        batt_df.loc[row_index, f'SOC lt {lower_limit}'] = count_under
    return batt_df
if SOC_limits_calculate:
    batt_df = count_SOC_limit_excedences(batt_df, SOC_profiles, SOC_count_limits=SOC_count_limits)


# Plotting the degradation bar chart with battery factor line graphs for each battery size
def plot_battery_size_overview(batt_df, factors, colour):
    if 'Remaining cap' in factors:
        factors.remove('Remaining cap')
    unique_sizes = batt_df['Size labels'].unique()

    for size in unique_sizes:
        # Filter data for the current battery size
        size_df = batt_df[batt_df['Size labels'] == size]

        fig, ax1 = plt.subplots(figsize=(11, 7))

        # Convert 'Time resolution' to categorical type for even spacing
        time_resolutions = [str(time) for time in list(size_df['Time resolution'].unique())]

        # Bar plot for remaining capacity
        bars = ax1.bar(time_resolutions, size_df['Remaining cap'], color='skyblue', width=0.8,
                       label='Remaining Capacity')
        ax1.set_xlabel('Time Resolution (minutes)')
        ax1.set_ylabel('Remaining Capacity (%)', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')

        # Adjust y-axis limits to show only the ends of the bars
        min_cap = size_df['Remaining cap'].min()
        max_cap = size_df['Remaining cap'].max()
        y_margin = (max_cap - min_cap) * 0.1  # 10% margin
        ax1.set_ylim(min_cap - y_margin, max_cap + y_margin)

        for i in range(len(factors)):
            ax = ax1.twinx()
            ax.plot(time_resolutions, size_df[factors[i]], color=colour[factors[i]], marker='o',
                    label=factors[i])
            ax.spines['right'].set_position(('outward', i * 60))
            ax.set_ylabel(factors[i], color=colour[factors[i]])
            ax.tick_params(axis='y', labelcolor=colour[factors[i]])

        # Title and legends
        fig.suptitle(f'Battery Size: {size[0]}, {size[1]}')
        fig.legend()
        plt.tight_layout()
        save_dir = r'M:\Documents\PhD\Papers\EUPVSEC2024\Conference proceedings figures\Draft 2'
        plt.savefig(f'{save_dir}\\{size[0]}_{size[1]}.png')
        plt.show()
if factors != []:
    if battery_plots:
        plot_battery_size_overview(batt_df, factors, colour)

def plot_cycles(cycle_data):
    x_pos_t = {1:0, 5:1, 15:2, 30:3}
    for (e, p) in set([(k[0],k[1]) for k in cycle_data.keys()]):
        for t in set([(k[2]) for k in cycle_data.keys()]):
            key = (e,p,t)
            cyc_depth = cycle_data[key]['cyc_depth']
            freq = cycle_data[key]['freq']
            x_axis = np.arange(len(cyc_depth))
            plt.bar((x_axis + (x_pos_t[t] * 0.2)), freq,
                    label=f"Time: {t} minutes. Avg DOD: {round(cycle_data[key]['avg_DoD'], 3)}. No. cycles: {sum(freq)}", width=0.2)
        plt.xticks(x_axis, ['%.2f' % elem for elem in cyc_depth])
        plt.xlabel('Depth of discharge (relative to SOC)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Battery size: {e:g}MWh, {p:g}MW')
        save_dir = r'M:\Documents\PhD\Papers\EUPVSEC2024\Conference proceedings figures\Draft 2'
        plt.savefig(f'{save_dir}\cycles_{e:g}MWh_{p:g}MW.png')
        plt.show()
if plot_cycle_data:
    plot_cycles(cycle_data)
def plot_lifetime_bar_chart(batt_df):
    plt.bar(batt_df['Size labels'], batt_df['Lifetime'])
    plt.show()
plot_lifetime_bar_chart(batt_df)


# Plot time series data of SOC profiles

# IDEAS:
# - Plot degradation by factor in waterfall chart
# - Plot 10, 20 and 25 mins
