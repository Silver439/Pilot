import numpy as np
import copy
from util.data_generate import Emin2
from scipy.cluster import hierarchy

def improvement(p,m,mu,std,R,Sigma):
    choosen_index = [] #选定科目
    remain_index = list(range(0,p)) #剩余科目
    choosen_index.append(int(np.argmin(mu)))
    remain_index.remove(np.argmin(mu))

    for j in range(m-1):
        best_improvement = -1 #全局最优
        best_id = -1
        for id1 in remain_index:
            improvement_min = 100 #局部最小
            for id2 in choosen_index:
                temp = mu[id2]-Emin2(id1,id2,mu,std,R)
                if temp < improvement_min:
                    improvement_min = temp
            if improvement_min > best_improvement:
                best_improvement = improvement_min
                best_id = id1
        choosen_index.append(best_id)
        remain_index.remove(best_id)

    sub_Sigma = Sigma[np.ix_(choosen_index, choosen_index)]
    B = np.linalg.cholesky(sub_Sigma) #对协方差矩阵进行cholesky分解
    return B,choosen_index

def LCB(p,m,mu,std,R,Sigma,J=10000,num=10,method = 'max'):
    B,choosen_index = improvement(p,2,mu,std,R,Sigma)
    remain_index = np.setdiff1d(np.arange(p), choosen_index)
    Gmin = 100
    bestid = -1
    for t in range(m-2):
        rou = np.ones(p)
        for idr in remain_index:
            correlations = R[idr, choosen_index]
            if method=='mean':
                corr = np.mean(correlations)
            else:
                corr = np.max(correlations)
            rou[idr] = corr
        remain_mu = mu[remain_index]
        remain_std = std[remain_index]
        remain_rou = rou[remain_index]
        n = len(remain_index)
        mu_matrix = remain_mu.reshape(1, n) < remain_mu.reshape(n, 1)
        std_matrix = remain_std.reshape(1, n) > remain_std.reshape(n, 1)
        rou_matrix = remain_rou.reshape(1, n) < remain_rou.reshape(n, 1)
        dominated_matrix = mu_matrix & std_matrix & rou_matrix
        is_dominated = np.any(dominated_matrix, axis=0)
        nondominated_index = remain_index[~is_dominated]
        if len(nondominated_index) > num:
             weights = np.random.dirichlet(np.ones(3))
             w_mu,w_std,w_rou = weights[0], weights[0], weights[1]
             sub_mu = mu[nondominated_index]
             sub_std = std[nondominated_index]
             sub_rou = rou[nondominated_index]
             norm_mu = (sub_mu - np.min(sub_mu)) / (np.max(sub_mu) - np.min(sub_mu))
             norm_std = (sub_std - np.min(sub_std)) / (np.max(sub_std) - np.min(sub_std))
             norm_rou = (sub_rou - np.min(sub_rou)) / (np.max(sub_rou) - np.min(sub_rou))
             weighted_scores = w_mu * norm_mu - w_std * norm_std + w_rou * norm_rou
             selected_positions = np.argsort(weighted_scores)[:num]
             final_indices = nondominated_index[selected_positions]
        else:
            final_indices = nondominated_index
        if len(final_indices)==1:
            choosen_index.append(int(final_indices[0])) 
            ids = np.where(remain_index == final_indices[0])[0]
            remain_index = np.delete(remain_index, ids)
            Gmin = 100
        else:
            for i in final_indices:
                #print(final_indices)
                copy_index = copy.deepcopy(choosen_index)
                copy_index.append(i)
                sub_Sigma = Sigma[np.ix_(copy_index, copy_index)]
                sub_B = np.linalg.cholesky(sub_Sigma) #对协方差矩阵进行cholesky分解
                G_cul = 0
                for j in range(J):
                    z = np.random.randn(len(copy_index))
                    temp = mu[copy_index] + sub_B @ z
                    G_cul += np.min(temp)
                G_temp = G_cul/J
                if G_temp < Gmin:
                    Gmin = G_temp
                    bestid = i
            Gmin = 100
            choosen_index.append(int(bestid)) 
            ids = np.where(remain_index == bestid)[0]
            remain_index = np.delete(remain_index, ids)

    sub_Sigma = Sigma[np.ix_(choosen_index, choosen_index)]
    U = np.linalg.cholesky(sub_Sigma) 
    return U,choosen_index

def SAA(p,m,mu,std,R,Sigma,J=10000):
    B,submodular_index = improvement(p,2,mu,std,R,Sigma)
    Gmin = 100
    bestid = -1
    for t in range(m-2):
        for i in range(p):
            if i not in submodular_index:
                copy_index = copy.deepcopy(submodular_index)
                copy_index.append(i)
                sub_Sigma = Sigma[np.ix_(copy_index, copy_index)]
                sub_B = np.linalg.cholesky(sub_Sigma) #对协方差矩阵进行cholesky分解
                G_cul = 0
                for j in range(J):
                    z = np.random.randn(len(copy_index))
                    temp = mu[copy_index] + sub_B @ z
                    G_cul += np.min(temp)
                G = G_cul/J
                if G<Gmin:
                    Gmin = G
                    bestid = i
        Gmin = 100
        submodular_index.append(bestid)
    sub_Sigma = Sigma[np.ix_(submodular_index, submodular_index)]
    U = np.linalg.cholesky(sub_Sigma) 
    return U,submodular_index

def mixed(p,m,mu,std,R,Sigma,J=10000):
    Gmin = 100
    bestid = -1
    J = J
    B,mixed_index = improvement(p,2,mu,std,R,Sigma)
    for j in range(m-2):
        imp = []
        for id1 in range(p):
            if id1 not in mixed_index:
                improvement_min = 100 #局部最小
                for id2 in mixed_index:
                    temp = mu[id2]-Emin2(id1,id2,mu,std,R)
                    if temp < improvement_min:
                        improvement_min = temp
                imp.append(improvement_min)
            else:
                imp.append(0)
        indices = np.argsort(imp)[-10:][::-1] #获取指标列表
        for i in indices:
            copy_index = copy.deepcopy(mixed_index)
            copy_index.append(i)
            sub_Sigma = Sigma[np.ix_(copy_index, copy_index)]
            sub_B = np.linalg.cholesky(sub_Sigma) #对协方差矩阵进行cholesky分解
            G_cul = 0
            for j in range(J):
                z = np.random.randn(len(copy_index))
                temp = mu[copy_index] + sub_B @ z
                G_cul += np.min(temp)
            G_temp = G_cul/J
            if G_temp < Gmin:
                Gmin = G_temp
                bestid = i
        Gmin = 100
        mixed_index.append(int(bestid)) 
    sub_Sigma = Sigma[np.ix_(mixed_index, mixed_index)]
    U = np.linalg.cholesky(sub_Sigma)    
    return U,mixed_index

def cluster_subjects(R, n_clusters=None, method='single'):
    """
    使用层次聚类对科目进行分组
    
    参数:
    R - 相关系数矩阵 (p×p)
    n_clusters - 期望的聚类数量(可选)
    method - 链接方法: 'single', 'complete', 'average', 'ward'等
    
    返回:
    聚类标签数组
    """
    # 将相关矩阵转换为距离矩阵(1-绝对相关系数)
    distance_matrix = 1 - np.abs(R)
    
    # 执行层次聚类
    linkage_matrix = hierarchy.linkage(distance_matrix, method=method)
    
    # 如果指定了聚类数量
    if n_clusters is not None:
        cluster_labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    else:
        # 或者使用距离阈值
        cluster_labels = hierarchy.fcluster(linkage_matrix, 0.8, criterion='distance')
    
    # # 绘制树状图
    # plt.figure(figsize=(12, 8))
    # dendrogram = hierarchy.dendrogram(linkage_matrix, labels=np.arange(1, R.shape[0]+1))
    # plt.title('Subject Clustering Dendrogram')
    # plt.xlabel('Subject Index')
    # plt.ylabel('Distance (1 - |correlation|)')
    # plt.show()
    
    return cluster_labels

def find_min_mu_in_clusters(mu, cluster_labels):
    """
    找出每个聚类中平均分(mu)最小的科目指标
    
    参数:
    mu - 各科目平均分数组(形状为(p,))
    cluster_labels - 聚类标签数组(形状为(p,))
    
    返回:
    包含每类最小mu科目指标的列表
    """
    # 获取所有唯一的聚类标签
    unique_clusters = np.unique(cluster_labels)
    
    # 存储每类中mu最小的科目指标
    min_mu_indices = []
    
    for cluster in unique_clusters:
        # 获取当前类别的所有科目索引
        cluster_indices = np.where(cluster_labels == cluster)[0]
        
        # 获取这些科目的mu值
        cluster_mu = mu[cluster_indices]
        
        # 找到mu最小的科目在cluster_indices中的位置
        min_pos_in_cluster = np.argmin(cluster_mu)
        
        # 获取全局索引(在原始mu数组中的位置)
        global_index = cluster_indices[min_pos_in_cluster]
        
        min_mu_indices.append(int(global_index))
    
    return min_mu_indices