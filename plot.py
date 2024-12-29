import math
import pandas as pd
import matplotlib.pyplot as plt

from typing import List

def tensorboard_smoothing(values: List[float], smooth: float = 0.9) -> List[float]:
    """
    values: List[`value` of Item]. You don't need to pass in `step`
    """
    # [0.81 0.9 1]. res[2] = (0.81 * values[0] + 0.9 * values[1] + values[2]) / 2.71
    norm_factor = 1
    x = 0
    res = []
    for i in range(len(values)):
        if math.isnan(values[i]):
            res.append(float('nan'))
            continue
        v = values[i]
        x = x * smooth + v  # Exponential decay
        res.append(x / norm_factor)
        #
        norm_factor *= smooth
        norm_factor += 1
    return res

def plot_reproduce_result():
    # 读取CSV文件
    df1 = pd.read_csv('ray_results/reproduce/reproduce_CPPO.csv')
    df2 = pd.read_csv('ray_results/reproduce/reproduce_IPPO.csv')
    df3 = pd.read_csv('ray_results/reproduce/reproduce_MAPPO.csv')

    # 计算iteration (the index of the row)
    df1['Iteration'] = df1.index + 1
    df2['Iteration'] = df2.index + 1
    df3['Iteration'] = df3.index + 1

    # 绘制折线图
    plt.plot(df1['Iteration'], df1['Value'], color='#FFE2D9') # 橙色 CPPO
    plt.plot(df2['Iteration'], df2['Value'], color='#DFF5E0') # 绿色 IPPO
    plt.plot(df3['Iteration'], df3['Value'], color='#F2DCF5') # 紫色 MAPPO

    # smooth
    values = df1['Value'].values
    smoothed_values = tensorboard_smoothing(values, smooth=0.7)
    plt.plot(df1['Iteration'], smoothed_values, label='CPPO', color='#FF7043')

    values = df2['Value'].values
    smoothed_values = tensorboard_smoothing(values, smooth=0.7)
    plt.plot(df2['Iteration'], smoothed_values, label='IPPO', color='#66BB6A')

    values = df3['Value'].values
    smoothed_values = tensorboard_smoothing(values, smooth=0.7)
    plt.plot(df3['Iteration'], smoothed_values, label='MAPPO', color='#AB47BC')

    plt.xlabel('Training Iteration', fontsize=12)
    plt.ylabel('Episode reward mean', fontsize=12)
    plt.legend()
    plt.show()
    
def plot_agent_num_comparison():
    # 读取CSV文件
    IPPO_dfs, CPPO_dfs, MAPPO_dfs = [], [], []
    for i in range(3, 6):
        IPPO_dfs.append(pd.read_csv(f'ray_results/agent_num/IPPO_agent{i}.csv'))
        CPPO_dfs.append(pd.read_csv(f'ray_results/agent_num/CPPO_agent{i}.csv'))
        MAPPO_dfs.append(pd.read_csv(f'ray_results/agent_num/MAPPO_agent{i}.csv'))
        
    # 计算iteration (the index of the row)
    for i in range(3):
        IPPO_dfs[i]['Iteration'] = IPPO_dfs[i].index + 1
        CPPO_dfs[i]['Iteration'] = CPPO_dfs[i].index + 1
        MAPPO_dfs[i]['Iteration'] = MAPPO_dfs[i].index + 1
        
    # 绘制折线图，一个算法绘制在一个图，绘制成3个水平摆放的图
    COLOR = ["#FFE2D9", "#DFF5E0", "#F2DCF5"]
    COLOR_S = ["#FF7043", "#66BB6A", "#AB47BC"]
    ALG_NAME = ['CPPO', 'MAPPO', 'IPPO']
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for k, dfs in enumerate([CPPO_dfs, MAPPO_dfs, IPPO_dfs]):
        for i in range(3):
            axs[k].plot(dfs[i]['Iteration'], dfs[i]['Value'], color=COLOR[i], alpha=0.6)
            values = dfs[i]['Value'].values
            smoothed_values = tensorboard_smoothing(values, smooth=0.8)
            axs[k].plot(dfs[i]['Iteration'], smoothed_values, label=f'{ALG_NAME[k]}_Agent{i + 3}', color=COLOR_S[i])
            axs[k].set_xlabel(f'{ALG_NAME[k]} Training Iteration', fontsize=12)
            axs[k].set_ylabel('Episode reward mean', fontsize=12)
            axs[k].legend(loc='upper left')
            #限制y轴范围
            axs[k].set_ylim(0, 17.5)
    
    # 紧凑布局
    plt.tight_layout()
    plt.show()
    
def plot_il_results():
    # 读取CSV文件
    df1 = pd.read_csv('ray_results/improvement/il/reproduce_IPPO.csv')
    df2 = pd.read_csv('ray_results/improvement/il/IPPO_impr.csv')

    # 计算iteration (the index of the row)
    df1['Iteration'] = df1.index + 1
    df2['Iteration'] = df2.index + 1

    # 绘制折线图
    plt.plot(df1['Iteration'], df1['Value'], color='#FFE2D9') # 橙色 IPPO
    plt.plot(df2['Iteration'], df2['Value'], color='#DFF5E0') # 绿色 IPPO_IL

    # smooth
    values = df1['Value'].values
    smoothed_values = tensorboard_smoothing(values, smooth=0.7)
    plt.plot(df1['Iteration'], smoothed_values, label='IPPO', color='#FF7043')

    values = df2['Value'].values
    smoothed_values = tensorboard_smoothing(values, smooth=0.7)
    plt.plot(df2['Iteration'], smoothed_values, label='IPPO-IL', color='#66BB6A')

    plt.xlabel('Training Iteration', fontsize=12)
    plt.ylabel('Episode reward mean', fontsize=12)
    plt.legend()
    plt.show()
    
def plot_noise_results():
    # 读取CSV文件
    df1 = pd.read_csv('ray_results/improvement/noise/reproduce_IPPO.csv')
    df2 = pd.read_csv('ray_results/improvement/noise/IPPO_impr.csv')
    df3 = pd.read_csv('ray_results/improvement/noise/reproduce_MAPPO.csv')
    df4 = pd.read_csv('ray_results/improvement/noise/MAPPO_impr.csv')
    
    # 计算iteration (the index of the row)
    df1['Iteration'] = df1.index + 1
    df2['Iteration'] = df2.index + 1
    df3['Iteration'] = df3.index + 1
    df4['Iteration'] = df4.index + 1
    
    # 绘制折线图，一个算法绘制在一个图，绘制成2个水平摆放的图
    COLOR = ["#FFE2D9", "#DFF5E0", "#F2DCF5"]
    COLOR_S = ["#FF7043", "#66BB6A", "#AB47BC"]
    ALG_NAME = ['IPPO', 'IPPO_Noise', 'MAPPO', 'MAPPO_Noise']
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for k, dfs in enumerate([[df1, df2], [df3, df4]]):
        for i in range(2):
            axs[k].plot(dfs[i]['Iteration'], dfs[i]['Value'], color=COLOR[i], alpha=0.7)
            values = dfs[i]['Value'].values
            smoothed_values = tensorboard_smoothing(values, smooth=0.8)
            axs[k].plot(dfs[i]['Iteration'], smoothed_values, label=f'{ALG_NAME[k*2+i]}', color=COLOR_S[i])
            axs[k].set_xlabel(f'{ALG_NAME[k]} Training Iteration', fontsize=12)
            axs[k].set_ylabel('Episode reward mean', fontsize=12)
            axs[k].legend(loc='upper left')
            #限制y轴范围
            axs[k].set_ylim(0, 15)
    
    # 紧凑布局
    plt.tight_layout()
    plt.show()
    
def plot_min_max():
    # 读取CSV文件
    IPPO_naive_min = pd.read_csv('ray_results/improvement/noise/IPPO_naive_min.csv')
    IPPO_noise_min = pd.read_csv('ray_results/improvement/noise/IPPO_noise_min.csv')
    
    # 计算iteration (the index of the row)
    IPPO_naive_min['Iteration'] = IPPO_naive_min.index + 1
    IPPO_noise_min['Iteration'] = IPPO_noise_min.index + 1
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(IPPO_naive_min['Iteration'], IPPO_naive_min['Value'], label='IPPO_naive_min', color='#FF7043')
    ax.plot(IPPO_noise_min['Iteration'], IPPO_noise_min['Value'], label='IPPO_noise_min', color='#66BB6A')
    
    # 添加标签和图例
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Episode reward min')
    ax.legend()
    plt.show()
    
if __name__ == '__main__':
    # plot_reproduce_result()
    # plot_agent_num_comparison()
    # plot_il_results()
    # plot_noise_results()
    plot_min_max()
    
        