import numpy as np
import pandas as pd
import csv
from util.methods import improvement, update_subject_stats
from util.data_generate import data1, data4

# Parameters
p, m, J, beta = 50, 8, 100000, 2
group_sizes = [10] * 5
seed = 1
n = 200 # Number of iterations
num_repeats = 30  # Number of repetitions
csv_path = str(p)+'_'+str(m)+'_'+str(beta)+'.csv'
column_names = [f'Step_{i}' for i in range(1, n+1)]

# 初始化CSV文件并写入列名
with open(csv_path, 'w') as f:
    f.write(','.join(['Repetition'] + column_names) + '\n')

# Main loop for repetitions
for repeat in range(num_repeats):
    np.random.seed(seed + repeat)  # Different seed for each repetition
    
    G = []
    # Data generation
    mu1, std1, R1, Sigma1 = data4(p=p, seed=seed+repeat, group_sizes=group_sizes)
    mu2, std2, R2, Sigma2 = data1(p=p, seed=seed+repeat, group_sizes=group_sizes)
    mu = np.random.multivariate_normal(mu1, Sigma1, size=1)[0]
    mu_0 = 80 * np.ones(p)
    mu_hat, var_hat, obs_num = np.zeros(p), np.ones(p), np.zeros(p)

    # Initial observation
    indices = improvement(p, m, mu_0, std2, R2, Sigma2)[1]
    obs_num[indices] += 1
    obs = np.random.multivariate_normal(mu[indices], Sigma2[np.ix_(indices, indices)], size=1)[0]

    # Initial update
    Gamma = Sigma2[np.ix_(indices, indices)]
    inv_mat = np.linalg.inv(Sigma1[np.ix_(indices, indices)] + Gamma)
    delta = (obs - mu_0[indices]).reshape(-1, 1)

    cov_x = Sigma1[:, indices]
    adjustments_mean = (cov_x @ inv_mat @ delta).flatten()
    adjustments_var = np.einsum('ij,jk,ki->i', cov_x, inv_mat, cov_x.T)
    mu_hat = mu_0 + adjustments_mean
    var_hat = np.diag(Sigma1) - adjustments_var
    X_n = indices

    # Main loop
    for _ in range(n):
        U, ids = improvement(p, m, mu_hat-beta*np.sqrt(var_hat), std2, R2, Sigma2)
        obs_num[ids] += 1
        indices = np.vstack((indices, ids))
        
        new_obs = np.random.multivariate_normal(mu[ids], Sigma2[np.ix_(ids, ids)], size=1)[0]
        X_n, obs = update_subject_stats(obs_num, X_n, obs, ids, new_obs)
        
        # Gamma matrix calculation
        has_X = np.array([np.any(indices == x, axis=1) for x in X_n])
        counts = has_X @ has_X.T
        obs_counts = obs_num[X_n].reshape(-1, 1) @ obs_num[X_n].reshape(1, -1)
        Gamma = (Sigma2[np.ix_(X_n, X_n)] * counts / obs_counts)
        Gamma = (Gamma + Gamma.T) / 2
        
        # Update estimates
        COV_XX = Sigma1[np.ix_(X_n, X_n)]
        inv_mat = np.linalg.inv(COV_XX + Gamma)
        delta = (obs - mu_0[X_n]).reshape(-1, 1)
        
        cov_x = Sigma1[:, X_n]
        adjustments_mean = (cov_x @ inv_mat @ delta).flatten()
        adjustments_var = np.einsum('ij,jk,ki->i', cov_x, inv_mat, cov_x.T)
        mu_hat = mu_0 + adjustments_mean
        var_hat = np.diag(Sigma1) - adjustments_var
        
        # Calculate G
        z_samples = np.random.randn(J, m)
        G.append(np.mean(np.min(mu[ids] + (U @ z_samples.T).T, axis=1)))

    # Final calculation
    U, ids = improvement(p, m, mu, std2, R2, Sigma2)
    z_samples = np.random.randn(J, m)
    G_real = np.mean(np.min(mu[ids] + (U @ z_samples.T).T, axis=1))
    result = np.array(G) - G_real
    
    # Store results
    with open(csv_path, 'a') as f:
        row_data = [str(repeat)] + [f'{x:.6f}' for x in result]  # 格式化数据
        f.write(','.join(row_data) + '\n')

# Convert to DataFrame and save to CSV
print(f"Results saved to {csv_path}")