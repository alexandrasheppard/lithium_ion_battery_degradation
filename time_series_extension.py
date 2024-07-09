import pandas as pd
import matplotlib.pyplot as plt
import os

def Extend15Y(SOC_1Y_results_folder):
    for m, fn in enumerate(os.listdir(SOC_1Y_results_folder)):
        if os.path.isfile(os.path.join(SOC_1Y_results_folder, str(fn))):
            filename, file_extension = os.path.splitext(fn)
            if file_extension == '.csv':
                df = pd.read_csv(SOC_1Y_results_folder + fn, parse_dates=[0]).rename(columns = {'Unnamed: 0':'dtm'})
            else:
                series = pd.read_pickle(SOC_1Y_results_folder + fn) #, parse_dates=[0])
                df = pd.DataFrame(series).reset_index(names='dtm')
                filename = fn

            df_y = {}
            df_y[0] = df
            for y in range(14):
                df_y[y+1] = pd.DataFrame()
                df_y[y+1]['dtm'] = df_y[y]['dtm'] + pd.offsets.DateOffset(years=1)
                df_y[y+1]['SOC'] = df_y[y]['SOC']
                df = pd.concat([df, df_y[y+1]]).reset_index(drop=True)

            df.to_pickle(f'M:\Documents\PhD\Code\PhD_Projects\IFE_degradation\input_data\\{filename}')
            print(f'Output file made for {filename} in input_data file')