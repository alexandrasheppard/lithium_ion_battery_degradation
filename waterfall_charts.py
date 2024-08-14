import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os

# Location of input files
folder = r'M:\Documents\PhD\Code\PhD_Projects\IFE_degradation\Results\waterfall'
pattern = re.compile(r'(.*)(\d+.?\d*)MWh_(\d+.?\d*)MW_(\d+T)(.*).csv')

deg_types = ['cal linear - ',
             'cyc linear - ',
             'cal + cyc linear - ',
             'cal + non_linear - ',
             'cyc + non_linear - ',
             'nonlinear only - ',
             'all - ',
             'all2 - ',
             'no_deg - ']

step_order = ['no_deg - ', 'cal linear - ', 'cal + cyc linear - ', 'all - ', 'all - '] # must be 4 elements
labels = ['Full', 'Cal', 'Cyc', 'Nonlinear', 'Remaining']
def set_up(folder, deg_types, pattern):
    dir = folder
    time_labels = {'1T': '1 min', '5T': '5 min', '15T': '15 min', '30T': '30 min'}

    batteries = []
    for m, fn in enumerate(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, str(fn))):
            filename, file_extension = os.path.splitext(fn)
            if file_extension == '.csv':
                deg_res = pd.read_csv(os.path.join(dir, fn), index_col='time', usecols=['time', 'Capacity_left'])
                remaining_cap = deg_res['Capacity_left'].iloc[-1]
                match = pattern.match(fn)
                deg_type = match.group(1)
                assert deg_type in deg_types, "Degradation type not recognised!"
                E = match.group(2)
                P = match.group(3)
                T = match.group(4)

                dict = {'deg_type': deg_type,
                        'filename': fn,
                        'E': float(E),
                        'P': float(P),
                        'Energy capacity': f'{E} MWh',
                        'Power capacity': f'{P} MW',
                        'Time resolution': int(T[:-1]),
                        'Time labels': time_labels[T],
                        'Remaining cap': remaining_cap}

                batteries.append(dict)
            batt_df = pd.DataFrame(batteries)
            batt_df = batt_df.sort_values(by=['deg_type', 'Time resolution']).reset_index(drop=True)
    return batt_df
batt_df = set_up(folder, deg_types, pattern)

def find_values(batt_df, step_order):
    # Find the values for each step
    values = {}
    for t in batt_df['Time resolution'].unique():
        filtered_df = batt_df[batt_df['Time resolution'] == t]
        key = f'{t} min'
        values[key] = []
        for deg_type in step_order:
            values[key].append(filtered_df[filtered_df['deg_type'] == deg_type]['Remaining cap'].values[0])
    steps = {}
    for key in values.keys():
        steps[key] = [j - i for i, j in zip(values[key][:-1], values[key][1:])]
    return values, steps
values, steps = find_values(batt_df, step_order)

def make_1_waterfall_chart(values, steps, labels, folder):
    colors = ['C0', 'skyblue', 'skyblue', 'skyblue', 'C0']  # , 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for key in values.keys():
        start_value, end_value = values[key][0], values[key][-1]

        # Start point for the bars, only showing changes for intermediate steps
        # start_points = [0, start_value - abs(steps[key][0]), start_value - abs(steps[key][0]) - abs(steps[key][1]), 0]
        start_points = [0,
                        start_value - abs(steps[key][0]),
                        start_value - abs(steps[key][0]) - abs(steps[key][1]),
                        start_value - abs(steps[key][0]) - abs(steps[key][1]),
                        0]

        # Heights of the bars
        heights = [start_value, abs(steps[key][0]), abs(steps[key][1]), end_value]
        heights = [start_value, abs(steps[key][0]), abs(steps[key][1]), abs(steps[key][2]), end_value]


        # Plotting the waterfall chart
        fig, ax = plt.subplots(figsize=(5, 4))

        # Plot start and end values
        bars = ax.bar(labels, heights, bottom=start_points, color=colors)

        # Adding labels on the bars for intermediate steps
        for i, (bar, height, start) in enumerate(zip(bars, heights, start_points)):
            # Placing the text at the top of the bars for all steps
            ax.text(bar.get_x() + bar.get_width() / 2, start + height if i == 0 or i == 4 else start,
                    round(height if i == 0 or i == 4 else steps[key][i-1], 2),
                    ha='center', va='bottom' if i == 0 or i == 4 else 'top')

        # Add grid, title, and labels
        ax.set_title(f"Degradation components - {key}")
        ax.set_ylabel("Remianing capacity [%]")
        # ax.set_ylim(bottom=80, top=105)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Show the plot
        plt.savefig(os.path.join(folder, f'Waterfall chart - {key}.jpg'))
        plt.show()


make_1_waterfall_chart(values, steps, labels, folder)