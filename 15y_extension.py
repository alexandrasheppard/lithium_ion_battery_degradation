import pandas as pd
import matplotlib.pyplot as plt
import os

# counter = 0
for m, fn in enumerate(os.listdir('input_data/1Y_data/')):
    if os.path.isfile(os.path.join('input_data/1Y_data/', str(fn))):
        filename, file_extension = os.path.splitext(fn)
        if file_extension == '.csv':
            # If lasted al the way down to............

            # file processing
            df = pd.read_csv('input_data/1Y_data/' + fn, parse_dates=[0]).rename(columns = {'Unnamed: 0':'dtm'})
        else:
            series = pd.read_pickle('input_data/1Y_data/' + fn) #, parse_dates=[0])
            df = pd.DataFrame(series).reset_index(names='dtm')

        df_y = {}
        df_y[0] = df
        for y in range(14):
            df_y[y+1] = pd.DataFrame()
            df_y[y+1]['dtm'] = df_y[y]['dtm'] + pd.offsets.DateOffset(years=1)
            df_y[y+1]['SOC'] = df_y[y]['SOC']
            df = pd.concat([df, df_y[y+1]]).reset_index(drop=True)
            # testseries = pd.DataFrame()
            # testseries['check'] = df.index
            # testseries.plot()
            # plt.show()

        df.to_pickle(f'input_data/{filename}')
        print(f"Output file made for {filename}")

