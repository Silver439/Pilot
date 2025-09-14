import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal
from util.data_generate import *
from util.methods import *
import csv
#random200
p = 200
#betas = ['smallest','cluster','dynamic']
betas = ['dynamic',]
seed, J, n_iter, num_repeats = 1, 10000, 150, 30
np.random.seed(seed=seed)
Sigma1 = np.loadtxt('Data_generate/Sigma1.txt')
Sigma1 = Sigma1 * 4

Sigma2 = np.loadtxt('Data_generate/group_cov.txt')
std2 = np.sqrt(np.diag(Sigma2))
R2 = Sigma2 / np.outer(std2, std2)
mu = np.loadtxt('Data_generate/mu.txt')

for m in [10,]:
    U3, ids3 = SAA(p, m, mu, std2, R2, Sigma2, J=J)
    z_samples = np.random.randn(J*10, m)
    G_real = np.mean(np.min(mu[ids3] + (U3 @ z_samples.T).T, axis=1))
    cluster_labels = cluster_subjects(R2, n_clusters=m, method='average')
    for beta in betas:

        csv_path = f'{p}_{m}_{beta}.csv'
        csv_path1 = f'{p}_{m}_{beta}_pre.csv'
        #csv_path2 = f'{p}_{m}_{beta}_post.csv'

        # Initialize CSV
        with open(csv_path, 'w') as f:
            f.write(','.join(['Repetition'] + [f'Step_{i}' for i in range(1, n_iter+1)]) + '\n')
        with open(csv_path1, 'w') as f:
            f.write(','.join(['Repetition'] + [f'Step_{i}' for i in range(1, n_iter+1)]) + '\n')
        # with open(csv_path2, 'w') as f:
        #     f.write(','.join(['Repetition'] + [f'Step_{i}' for i in range(1, n_iter+1)]) + '\n')

        for repeat in tqdm(range(num_repeats)):
            # Initial setup
            seed = seed + repeat
            
            # Initialize variables
            mu_0 = np.full(p, 70)
            mu_hat, var_hat, obs_num = np.zeros(p), np.ones(p), np.zeros(p)
            
            # Initial observation
            N_0 = math.ceil(p / m)
            for i in range(N_0):
                ids = efficient_initialization(p, m, obs_num, mu_0, std2, R2, Sigma2, J)
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
            #G_post = []
            for t in range(N_0+1, n_iter+1):
                if beta == 'dynamic':
                    dynamic_beta = np.sqrt(6 * np.log(t) / np.maximum(obs_num, 1))
                    U, ids = improvement(p, m, mu_hat - dynamic_beta * np.sqrt(var_hat), std2, R2, Sigma2, J=J)
                elif beta == 'cluster': 
                    ids = find_min_mu_in_clusters(mu_hat, cluster_labels)
                    sub_Sigma_cluster = Sigma2[np.ix_(ids, ids)]
                    U = np.linalg.cholesky(sub_Sigma_cluster) 
                else:
                    ids = np.argsort(mu_hat)[:m]
                    sub_Sigma_smallest = Sigma2[np.ix_(ids, ids)]
                    U = np.linalg.cholesky(sub_Sigma_smallest) 
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
                #U2, ids2 = improvement(p, m, mu_hat, std2, R2, Sigma2, J=J)
                #z_samples = np.random.randn(J*10, m)
                #G_post.append(np.mean(np.min(mu[ids2] + (U2 @ z_samples.T).T, axis=1)))
            
            # Save results
            with open(csv_path, 'a') as f:
                f.write(','.join([str(repeat)] + [f'{x:.6f}' for x in np.array(G)-G_real]) + '\n')
            with open(csv_path1, 'a') as f:
                f.write(','.join([str(repeat)] + [f'{x:.6f}' for x in np.array(G_pre)-G_real]) + '\n')
            # with open(csv_path2, 'a') as f:
            #     f.write(','.join([str(repeat)] + [f'{x:.6f}' for x in np.array(G_post)-G_real]) + '\n')
