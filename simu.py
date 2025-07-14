import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal
from util.data_generate import *
from util.methods import *
import csv

p= 50
group_sizes = [10]*5
seed, J, n_iter, num_repeats = 1, 10000, 200, 30
np.random.seed(seed=seed)
betas = [0,2,20]
Sigma1 = generate_cov_matrix(p)
mu1 = 70 * np.ones(p)
mu2, std2, R2, Sigma2 = data1(p=p, seed=seed, group_sizes=group_sizes)
mu = np.random.multivariate_normal(mu1, Sigma1)

for m in [5,8,12]:
    for beta in betas:

        csv_path = f'{p}_{m}_{beta}.csv'
        csv_path1 = f'{p}_{m}_{beta}_pre.csv'
        csv_path2 = f'{p}_{m}_{beta}_post.csv'

        # Initialize CSV
        with open(csv_path, 'w') as f:
            f.write(','.join(['Repetition'] + [f'Step_{i}' for i in range(1, n_iter+1)]) + '\n')
        with open(csv_path1, 'w') as f:
            f.write(','.join(['Repetition'] + [f'Step_{i}' for i in range(1, n_iter+1)]) + '\n')
        with open(csv_path2, 'w') as f:
            f.write(','.join(['Repetition'] + [f'Step_{i}' for i in range(1, n_iter+1)]) + '\n')

        for repeat in tqdm(range(num_repeats)):
            # Initial setup
            seed = seed + repeat
            
            # Initialize variables
            mu_0 = np.full(p, 60)
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
            adjustments = np.diag(COV_x @ inv_mat @ COV_x.T)
            var_hat = np.diag(Sigma1) - adjustments
            
            # Main loop
            G = [] #用于计算累积遗憾
            G_pre = [] 
            G_post = []
            for _ in range(1, n_iter+1):
                U, ids = improvement(p, m, mu_hat-beta*np.sqrt(var_hat), std2, R2, Sigma2, J=J)
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
                adjustments = np.diag(COV_x @ inv_mat @ COV_x.T)
                var_hat = np.diag(Sigma1) - adjustments

                # Calculate G
                z_samples = np.random.randn(J*10, m)
                G.append(np.mean(np.min(mu[ids] + (U @ z_samples.T).T, axis=1)))

                Gmin = 1000
                idmin = []
                z_samples2 = np.random.randn(J, m)
                for id in indices:
                    sub_Sigma = Sigma2[np.ix_(id, id)]
                    B = np.linalg.cholesky(sub_Sigma)
                    Gtemp = np.mean(np.min(mu_hat[id] + (B @ z_samples2.T).T, axis=1))
                    if Gtemp < Gmin:
                        idmin = id.copy()
                        Gmin = Gtemp
                ids1 = idmin

                sub_Sigma = Sigma2[np.ix_(ids1, ids1)]
                U1 = np.linalg.cholesky(sub_Sigma)
                G_pre.append(np.mean(np.min(mu[ids1] + (U1 @ z_samples.T).T, axis=1)))
                
                # Calculate G_post
                U2, ids2 = improvement(p, m, mu_hat, std2, R2, Sigma2, J=J)
                #z_samples = np.random.randn(J*10, m)
                G_post.append(np.mean(np.min(mu[ids2] + (U2 @ z_samples.T).T, axis=1)))

            # Calculate G_real
            U3, ids3 = improvement(p, m, mu, std2, R2, Sigma2, J=J)
            z_samples = np.random.randn(J*10, m)
            G_real = np.mean(np.min(mu[ids3] + (U3 @ z_samples.T).T, axis=1))
            
            # Save results
            with open(csv_path, 'a') as f:
                f.write(','.join([str(repeat)] + [f'{x:.6f}' for x in np.array(G)-G_real]) + '\n')
            with open(csv_path1, 'a') as f:
                f.write(','.join([str(repeat)] + [f'{x:.6f}' for x in np.array(G_pre)-G_real]) + '\n')
            with open(csv_path2, 'a') as f:
                f.write(','.join([str(repeat)] + [f'{x:.6f}' for x in np.array(G_post)-G_real]) + '\n')

        print(f"Results saved to {csv_path} and {csv_path2}")