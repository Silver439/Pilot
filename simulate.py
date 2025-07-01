import time
import numpy as np
from util.data_generate import *
from util.methods import *
import pandas as pd

# 实验参数
n = 100  # 实验重复次数
J = 10000  # 抽样次数
J1 = 100000 # 计算结果抽样次数
p, m = 100, 10  # 问题规模参数
#group_sizes = [10, 10, 10, 10, 10]
group_sizes = [15, 10, 20, 10, 15, 10, 10, 10]  # 分组大小

# 定义6种算法及其对应的索引获取函数
methods = [
    ('smallest', lambda mu: np.argsort(mu)[:m]),  # 直接取最小值
    ('cluster', lambda mu: find_min_mu_in_clusters(mu, cluster_subjects(R, n_clusters=m, method='average'))),  # 基于层次聚类的方法
    ('LCB_max', lambda mu: LCB(p, m, mu, std, R, Sigma, J=J, num=10, method='max')[1]),  # 多目标优化+LCB
    ('improvement', lambda mu: improvement(p, m, mu, std, R, Sigma)[1]),  # 基于集合贡献值的方法
    ('SAA', lambda mu: SAA(p, m, mu, std, R, Sigma, J=J)[1]),  # SAA
    ('mixed', lambda mu: mixed(p, m, mu, std, R, Sigma, J=J)[1])  # SAA+集合贡献值筛选
]

rankings = np.zeros(6)  # 存储每种方法获胜次数
times = np.zeros(6)    # 存储每种方法总耗时

# 初始化输出文件
with open('output.csv', 'w') as f:
    f.write(','.join([m[0] for m in methods]) + '\n')

for seed in range(n):
    # 生成数据
    mu, std, R, Sigma = data4(p=p, seed=seed, group_sizes=group_sizes)
    np.random.seed(seed)
    
    # 运行所有方法
    B_list, indices_list = [], []
    for i, (name, get_indices) in enumerate(methods):
        start = time.perf_counter()
        indices = get_indices(mu)  # 获取选中的索引
        B = np.linalg.cholesky(Sigma[np.ix_(indices, indices)])  # Cholesky分解
        times[i] += time.perf_counter() - start  # 累计时间
        
        B_list.append(B)
        indices_list.append(indices)
    
    # 模拟计算各方法表现
    G = np.zeros(6)
    for _ in range(J1):
        z = np.random.randn(m)
        for i in range(6):
            G[i] += np.min(mu[indices_list[i]] + B_list[i] @ z)
    G /= J1  # 计算平均值

    # 计算本次实验的名次
    ranks = np.argsort(np.argsort(G)) + 1
    rankings += ranks
    
    # 记录结果
    with open('output.csv', 'a') as f:
        f.write(','.join([f"{x:.6f}" for x in G]) + '\n')

    print(G)

# 记录并输出结果
data = pd.read_csv('output.csv')
smallest = np.mean(data['smallest'].to_numpy())
cluster = np.mean(data['cluster'].to_numpy())
LCB_max = np.mean(data['LCB_max'].to_numpy())
improvement = np.mean(data['improvement'].to_numpy())
SAA = np.mean(data['SAA'].to_numpy())
mixed = np.mean(data['mixed'].to_numpy())
results = [smallest,cluster,LCB_max,improvement,SAA,mixed]
avg_ranks = rankings / n

result_df = pd.DataFrame({
    '算法名称': [m[0] for m in methods],
    '平均名次': avg_ranks.round(2),
    '平均优化结果': results,
    '平均用时(秒)': times / n
})
result_df.to_csv('algorithm_results.csv', float_format='%.6f')