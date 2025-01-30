import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_time_line_plots():
    fig, axs = plt.subplots(1, 3, figsize=(7/1.1, 5/1.1))

    battery_size = '1 MWh'

    lifetime = {}
    lifetime[battery_size] = pd.DataFrame()
    lifetime[battery_size]['0.25 h'] = pd.Series(index = [30, 15, 5, 1], data = [4.240930869267625, 4.240930869267625, 3.3319644079397674, 3.3319644079397674])
    lifetime[battery_size]['0.5 h'] = pd.Series(index = [30, 15, 5, 1], data = [4.240930869267625, 4.240930869267625, 3.3319644079397674, 3.482546201232033])
    lifetime[battery_size]['1 h'] = pd.Series(index = [30, 15, 5, 1], data = [4.544832306639288, 5.300479123887748, 3.9370294318959616, 4.090349075975359])
    lifetime[battery_size]['2 h'] = pd.Series(index = [30, 15, 5, 1], data = [6.061601642710472, 6.666666666666667, 4.848733744010952, 4.848733744010952])

    axs[0].plot(lifetime[battery_size], linestyle='--', alpha=0.4)
    for column in lifetime[battery_size].columns:
        axs[0].scatter(lifetime[battery_size].index, lifetime[battery_size][column], s=60, label=f'{column} (points)')
    axs[0].invert_xaxis()
    axs[0].legend(lifetime[battery_size].columns)
    axs[0].set_title('1 MWh')
    axs[0].set_ylabel('Battery lifetime (years)')
    axs[0].set_xticks([30, 15, 5, 1])
    # plt.show()


    battery_size = '2 MWh'

    lifetime = {}
    lifetime[battery_size] = pd.DataFrame()
    lifetime[battery_size]['0.25 h'] = pd.Series(index = [30, 15, 5, 1], data = [7.121149897330596, 5.152635181382615, 5.002053388090349, 5.300479123887748])
    lifetime[battery_size]['0.5 h'] = pd.Series(index = [30, 15, 5, 1], data = [7.121149897330596, 5.152635181382615, 5.152635181382615, 5.300479123887748])
    lifetime[battery_size]['1 h'] = pd.Series(index = [30, 15, 5, 1], data = [7.422313483915127, 7.72621492128679, 5.300479123887748, 5.453798767967146])
    lifetime[battery_size]['2 h'] = pd.Series(index = [30, 15, 5, 1], data = [8.030116358658454, 8.635181382614647, 5.604380561259411, 5.453798767967146])

    axs[1].plot(lifetime[battery_size], linestyle='--', alpha=0.4)
    for column in lifetime[battery_size].columns:
        axs[1].scatter(lifetime[battery_size].index, lifetime[battery_size][column], s=60, label=f'{column} (points)')
    axs[1].invert_xaxis()
    axs[1].legend(lifetime[battery_size].columns)
    axs[1].set_title(battery_size)
    # plt.show()
    axs[1].set_xlabel('Settlement time (minutes)')
    axs[1].set_xticks([30, 15, 5, 1])



    battery_size = '4 MWh'

    lifetime = {}
    lifetime[battery_size] = pd.DataFrame()
    lifetime[battery_size]['0.25 h'] = pd.Series(index = [1, 5, 15, 30], data = [6.209445585215605, 6.513347022587269, 6.513347022587269, 6.8172484599589325])
    lifetime[battery_size]['0.5 h'] = pd.Series(index = [1, 5, 15, 30], data = [6.209445585215605, 6.362765229295003, 6.666666666666667, 6.666666666666667])
    lifetime[battery_size]['1 h'] = pd.Series(index = [1, 5, 15, 30], data = [6.209445585215605, 6.513347022587269, 6.513347022587269, 6.666666666666667])
    lifetime[battery_size]['2 h'] = pd.Series(index = [1, 5, 15, 30], data = [6.209445585215605, 6.362765229295003, 8.331279945242985, 10.91])

    # axs[2].plot(lifetime[battery_size])
    # axs[2].invert_xaxis()
    # axs[2].legend(lifetime[battery_size].columns)
    # axs[2].set_title(battery_size)
    # axs[2].set_yscale('linear')
    # plt.show()

    axs[2].plot(lifetime[battery_size], linestyle='--', alpha=0.4)
    for column in lifetime[battery_size].columns:
        axs[2].scatter(lifetime[battery_size].index, lifetime[battery_size][column], s=60, label=f'{column} (points)')
    axs[2].invert_xaxis()
    axs[2].legend(lifetime[battery_size].columns)
    axs[2].set_title(battery_size)
    axs[2].set_xticks([1, 5, 15, 30])

    axs[0].set_ylim(3, 11)
    axs[1].set_ylim(3, 11)
    axs[2].set_ylim(3, 11)

    plt.show()


plot_time_line_plots()

# input_folder = r'M:\Documents\PhD\Code\PhD_Projects\IFE_degradation\Results\29_1_0.4\Opt_model_results'
# input_folder_new = r'M:\Documents\PhD\Code\PhD_Projects\IFE_degradation\Results\20241106092807\29_10\Opt_model_results'
# input_folder_new_2 = r'M:\Documents\PhD\Code\PhD_Projects\IFE_degradation\Results\20241106094557\29_10\Opt_model_results'
# bat_4MWh_2MW_15T = pd.read_csv(input_folder + r'\4MWh_2MW_15T_1.0_Results.csv')
# bat_4MWh_2MW_15T_new = pd.read_csv(input_folder_new + r'\4MWh_2MW_15T_1_Results.csv')
# bat_4MWh_2MW_15T_new_2 = pd.read_csv(input_folder_new_2 + r'\4MWh_2MW_15T_1_Results.csv')
# bat_4MWh_2MW_30T = pd.read_csv(input_folder + r'\4MWh_2MW_30T_1.0_Results.csv')
# bat_4MWh_2MW_5T = pd.read_csv(input_folder + r'\4MWh_2MW_5T_1.0_Results.csv')
# bat_4MWh_2MW_1T = pd.read_pickle(input_folder + r'\4MWh_2MW_1T_1.0_Results')
#
# # Create a normalized x-axis for each series
# x_30T = np.linspace(0, 1, len(bat_4MWh_2MW_30T))
# x_15T = np.linspace(0, 1, len(bat_4MWh_2MW_15T))
# x_15T_new = np.linspace(0, 1, len(bat_4MWh_2MW_15T_new))
# x_15T_new_2 = np.linspace(0, 1, len(bat_4MWh_2MW_15T_new_2))
# x_5T = np.linspace(0, 1, len(bat_4MWh_2MW_5T))
# x_1T = np.linspace(0, 1, len(bat_4MWh_2MW_1T))
#
# # Plot with normalized x-axis
# # plt.plot(x_1T, bat_4MWh_2MW_1T['SOC'], label='1T')
# # plt.plot(x_5T, bat_4MWh_2MW_5T['SOC'], label='5T')
# plt.plot(x_15T, bat_4MWh_2MW_15T['SOC'], label='15T')
# plt.plot(x_15T, bat_4MWh_2MW_15T_new['SOC'], label='15T_new')
# plt.plot(x_15T, bat_4MWh_2MW_15T_new_2['SOC'], label='15T_new_2')
# # plt.plot(x_30T, bat_4MWh_2MW_30T['SOC'], label='30T')
#
#
#
# # Add legend and show plot
# plt.xlabel("Normalized Time (0 to 1)")
# plt.ylabel("SOC")
# plt.legend()
# plt.xlim(-0.01, 0.1)
# plt.show()
#
