import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal
from util.data_generate import *
from util.methods import *
import csv

# Parameters
p, m = 50, 5
group_sizes = [10]*5
seed, J, n_iter, num_repeats = 1, 10000, 200, 30
csv_path = f'{p}_{m}_TS.csv'
csv_path2 = f'{p}_{m}_TS_post.csv'

# Initialize CSV
with open(csv_path, 'w') as f:
    f.write(','.join(['Repetition'] + [f'Step_{i}' for i in range(1, n_iter+1)]) + '\n')
with open(csv_path2, 'w') as f:
    f.write(','.join(['Repetition'] + [f'Step_{i}' for i in range(1, n_iter+1)]) + '\n')

for repeat in tqdm(range(num_repeats)):
    # Initial setup
    seed = seed + repeat
    Sigma1 = generate_cov_matrix(p,seed)
    mu2, std2, R2, Sigma2 = data4(p=p, seed=seed, group_sizes=group_sizes)
    mu = np.random.multivariate_normal(mu2, Sigma1)
    
    # Initialize variables
    mu_0 = np.full(p, 80)
    mu_hat, var_hat, obs_num = np.zeros(p), np.ones(p), np.zeros(p)
    
    # First observation
    _, ids = improvement(p, m, mu_0, std2, R2, Sigma2, J=J)
    obs_num[ids] += 1
    indices = ids.copy()
    obs = np.random.multivariate_normal(mu[ids], Sigma2[np.ix_(ids, ids)])
    X_n = ids.copy()
    
    # Update estimates
    Gamma = Sigma2[np.ix_(ids, ids)]
    inv_mat = np.linalg.inv(Sigma1[np.ix_(ids, ids)] + Gamma)
    delta = (obs - mu_0[ids]).reshape(-1, 1)
    
    COV_x = Sigma1[:, ids]
    adjustments = (COV_x @ inv_mat @ delta).flatten()
    mu_hat = mu_0 + adjustments

    Sigma_XX = np.eye(p)
    COV_Xn = Sigma1[:, X_n]
    temp = COV_Xn @ inv_mat @ COV_Xn.T
    Sigma_XX = Sigma1 - temp

    Sigma_XX = np.triu(Sigma_XX)  # Take upper triangle
    Sigma_XX += Sigma_XX.T  # Make symmetric
    np.fill_diagonal(Sigma_XX, np.diag(Sigma_XX)/2)  # Correct diagonal

    # Main loop
    G = []
    G_post = []
    for _ in range(1, n_iter+1):

        mu_TS = np.random.multivariate_normal(mu_hat, Sigma_XX)
        U, ids = improvement(p, m, mu_TS, std2, R2, Sigma2, J=J)
        obs_num[ids] += 1
        indices = np.vstack((indices, ids))
        
        newobs = np.random.multivariate_normal(mu[ids], Sigma2[np.ix_(ids, ids)])
        X_n, obs = update_subject_stats(obs_num, X_n, obs, ids, newobs)
        
        # Update Gamma matrix
        len_Xn = len(X_n)
        Gamma = np.zeros((len_Xn, len_Xn))
        for i in range(len_Xn):
            for j in range(i, len_Xn):
                count = np.sum((indices == X_n[i]).any(1) & (indices == X_n[j]).any(1))
                Gamma[i,j] = Sigma2[X_n[i],X_n[j]] * count / (obs_num[X_n[i]] * obs_num[X_n[j]])
        Gamma += Gamma.T
        np.fill_diagonal(Gamma, np.diag(Gamma)/2)
        
        # Update estimates
        COV_XX = Sigma1[np.ix_(X_n, X_n)]
        inv_mat = np.linalg.inv(COV_XX + Gamma)
        delta = (obs - mu_0[X_n]).reshape(-1, 1)
        
        COV_x = Sigma1[:, X_n]
        adjustments = (COV_x @ inv_mat @ delta).flatten()
        mu_hat = mu_0 + adjustments

        Sigma_XX = np.eye(p)
        COV_Xn = Sigma1[:, X_n]
        temp = COV_Xn @ inv_mat @ COV_Xn.T
        Sigma_XX = Sigma1 - temp

        Sigma_XX = np.triu(Sigma_XX)  # Take upper triangle
        Sigma_XX += Sigma_XX.T  # Make symmetric
        np.fill_diagonal(Sigma_XX, np.diag(Sigma_XX)/2)  # Correct diagonal

        # Calculate G
        z_samples = np.random.randn(J*10, m)
        G.append(np.mean(np.min(mu[ids] + (U @ z_samples.T).T, axis=1)))
        
        # Calculate G_post
        U2, ids2 = improvement(p, m, mu_hat, std2, R2, Sigma2, J=J)
        z_samples = np.random.randn(J*10, m)
        G_post.append(np.mean(np.min(mu[ids2] + (U2 @ z_samples.T).T, axis=1)))
    
    # Calculate G_real
    U, ids = improvement(p, m, mu, std2, R2, Sigma2, J=J)
    z_samples = np.random.randn(J*10, m)
    G_real = np.mean(np.min(mu[ids] + (U @ z_samples.T).T, axis=1))
    
    # Save results
    with open(csv_path, 'a') as f:
        f.write(','.join([str(repeat)] + [f'{x:.6f}' for x in np.array(G)-G_real]) + '\n')
    with open(csv_path2, 'a') as f:
        f.write(','.join([str(repeat)] + [f'{x:.6f}' for x in np.array(G_post)-G_real]) + '\n')

print(f"Results saved to {csv_path} and {csv_path2}")