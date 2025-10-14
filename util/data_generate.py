import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import eigh
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy
import math    

def Emin2(id1,id2,mu,std,R): #计算两个正态随机变量的最小值期望
    mu1 = mu[id1]
    mu2 = mu[id2]
    std1 = std[id1]
    std2 = std[id2]
    rou = R[id1,id2]
    theta = math.sqrt(std1*std1+std2*std2-2*rou*std1*std2)+1e-6
    result = mu1*norm.cdf((mu2-mu1)/theta)+mu2*norm.cdf((mu1-mu2)/theta)-theta*norm.pdf((mu2-mu1)/theta)
    return result

def data1(p=50, seed=42, group_sizes=[10, 10, 10, 10, 10]):
    # 1. 参数设置
    p = p  # 科目总2数
    n_samples = 1000
    seed = seed 
    group_sizes = group_sizes
    np.random.seed(seed)

    # 2. 生成均值（70-90分）
    mu = np.random.uniform(70, 90, size=p)

    # 3. 生成协方差矩阵
    Sigma = np.zeros((p, p))
    start = 0
    for size in group_sizes:
        end = start + size
        block_corr = np.random.uniform(0.6, 1)
        block = np.ones((size, size)) * block_corr
        np.fill_diagonal(block, 1)
        Sigma[start:end, start:end] = block 
        start = end

    # 结合方差
    variances = np.random.uniform(25, 49, size=p)
    D = np.diag(np.sqrt(variances))
    Sigma = D @ Sigma @ D

    # 4. 生成数据并截断
    X = np.random.multivariate_normal(mu, Sigma, size=n_samples)
    X = np.clip(X, 0, 100)

    mu = mu
    std = np.sqrt(np.diag(Sigma))
    R = np.corrcoef(X[:, :].T)
     
    return mu,std,R,Sigma

def data2(p=50, seed=42, group_sizes=[10, 10, 10, 10, 10]):
    # 1. 参数设置
    p = p  # 科目总数
    n_samples = 1000  
    seed = seed 
    group_sizes = group_sizes
    np.random.seed(seed)

    n_groups = len(group_sizes)
    starts = np.cumsum([0] + group_sizes[:-1])

    # 生成协方差矩阵（两步法）
    Sigma = np.eye(p)

    # 第一步：强组内相关
    for i in range(n_groups):
        start, end = starts[i], starts[i] + group_sizes[i]
        block_corr = np.random.uniform(0.4, 0.7)  # 更高的组内相关
        block = np.ones((group_sizes[i], group_sizes[i])) * block_corr
        np.fill_diagonal(block, 1)
        Sigma[start:end, start:end] = block

    # 第二步：弱组间相关
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            start_i, end_i = starts[i], starts[i] + group_sizes[i]
            start_j, end_j = starts[j], starts[j] + group_sizes[j]
            inter_corr = np.random.uniform(-0.3, 0.3)  # 更弱的组间相关
            Sigma[start_i:end_i, start_j:end_j] = inter_corr
            Sigma[start_j:end_j, start_i:end_i] = inter_corr

    # 保证正定性
    min_eigval = np.min(eigh(Sigma, eigvals_only=True))
    if min_eigval <= 0:
        Sigma += (0.1 - min_eigval) * np.eye(p)

    # 添加方差（更小的方差范围）
    variances = np.random.uniform(25, 49, size=p)
    D = np.diag(np.sqrt(variances))
    Sigma = D @ Sigma @ D

    # 生成数据（更集中的均值）
    mu = np.random.uniform(70, 90, size=p)
    X = np.random.multivariate_normal(mu, Sigma, size=n_samples)
    X = np.clip(X, 0, 100)

    mu = mu
    std = np.sqrt(np.diag(Sigma))
    R = np.corrcoef(X[:, :].T)
    return mu,std,R,Sigma

def data3(p=50, seed=42, group_sizes=[10, 10, 10, 10, 10]):
    # 1. 参数设置
    p = p  # 科目总数
    n_samples = 1000
    seed = seed 
    group_sizes = group_sizes  
    np.random.seed(seed)

    # 2. 生成均值（第一组最低，其他组较高）
    mu = np.zeros(p)
    p1 = group_sizes[0]
    # 第一组（10科）：低均值 70-75
    mu[:p1] = np.random.uniform(70, 75, size=p1)  
    # 其他组（40科）：较高均值 75-90
    mu[p1:] = np.random.uniform(75, 90, size=p-p1)  

    # 3. 构建协方差矩阵（组内高相关，组间独立）
    Sigma = np.zeros((p, p))
    start = 0
    for size in group_sizes:
        end = start + size
        # 组内高相关（0.75-1.0）
        block_corr = np.random.uniform(0.75, 1.0)
        block = np.ones((size, size)) * block_corr
        np.fill_diagonal(block, 1)  # 对角线=1
        Sigma[start:end, start:end] = block
        start = end

    # 添加随机方差（标准差 5-7）
    variances = np.random.uniform(25, 49, size=p)  # 方差=25-49 → 标准差=5-7
    D = np.diag(np.sqrt(variances))
    Sigma = D @ Sigma @ D  # 最终协方差矩阵

    # 4. 生成数据并截断（0-100分）
    X = np.random.multivariate_normal(mu, Sigma, size=n_samples)
    X = np.clip(X, 0, 100)

    mu = mu
    std = np.sqrt(np.diag(Sigma))
    R = np.corrcoef(X[:, :].T)
    return mu,std,R,Sigma

def random_correlation_matrix(p):
    """生成随机正定相关矩阵"""
    A = np.random.uniform(-0.9, 0.9, size=(p, p))  # 限制范围避免极端值
    A = (A + A.T) / 2  # 确保对称
    np.fill_diagonal(A, 1)  # 对角线设为1
    
    # 调整矩阵使其正定
    min_eigenvalue = np.min(np.linalg.eigvals(A))
    if min_eigenvalue < 1e-8:  # 如果最小特征值太小
        A += (0.1 - min_eigenvalue) * np.eye(p)  # 调整
    return A

def data4(p=50, seed=42, group_sizes=[10, 10, 10, 10, 10]):
    np.random.seed(seed)
    
    # 1. 生成均值 (70-90分)
    mu = np.random.uniform(70, 90, size=p)
    
    # 2. 生成随机正定相关矩阵
    R = random_correlation_matrix(p)
    
    # 3. 生成随机标准差 (5-7分)
    std_devs = np.random.uniform(5, 7, size=p)
    D = np.diag(std_devs)
    Sigma = D @ R @ D  # 协方差矩阵
    
    # 4. 生成数据并截断到 [0, 100]
    X = np.random.multivariate_normal(mu, Sigma, size=1000)
    X = np.clip(X, 0, 100)
    
    return mu, std_devs, np.corrcoef(X.T), Sigma


