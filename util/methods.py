import numpy as np
import copy
from util.data_generate import Emin2
from scipy.cluster import hierarchy

def improvement(p,m,mu,std,R,Sigma,J=10000):
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

def update_subject_stats(obs_num, X_n, obs, ids, newobs):

    # Convert X_n and obs to dictionary for faster lookup
    X_n_dict = {subject: avg for subject, avg in zip(X_n, obs)}
    
    # Update averages for the new student's subjects
    for subject, score in zip(ids, newobs):
        if subject in X_n_dict:
            # Compute new average: (old_avg * (old_count - 1) + new_score) / old_count
            old_avg = X_n_dict[subject]
            old_count = obs_num[subject] - 1  # because obs_num is already updated
            new_avg = (old_avg * old_count + score) / obs_num[subject]
            X_n_dict[subject] = new_avg
        else:
            # New subject: just add its score (count is 1, which is correct since obs_num is updated)
            X_n_dict[subject] = score
    
    # Convert back to numpy arrays
    X_n_new = np.array(list(X_n_dict.keys()))
    obs_new = np.array(list(X_n_dict.values()))
    
    return X_n_new, obs_new

def generate_cov_matrix(n):
    """
    生成n×n的协方差矩阵:
    - 对角线元素为100
    - 非对角线元素为随机非负值
    - 矩阵是对称且正定的(确保是有效的协方差矩阵)
    """
    # 生成随机非负矩阵
    A = np.random.uniform(0, 5, size=(n, n))  # 调整0-10的范围根据需要
    
    # 构造对称矩阵
    cov = (A + A.T) / 2
    
    # 将对角线设置为100
    np.fill_diagonal(cov, 25)
    
    # 确保矩阵是正定的(协方差矩阵必须满足)
    # 通过添加一个足够大的单位矩阵实现
    min_eig = np.min(np.real(np.linalg.eigvals(cov)))
    if min_eig <= 0:
        cov += (np.eye(n) * (-min_eig + 0.1))
    
    return cov

def assign_variances(means, min_var=25, max_var=144, correlation_strength=0.7):
    """
    为给定的均值分配方差，使其与均值正相关
    
    参数:
    - means: 均值数组
    - min_var: 最小方差 (默认25)
    - max_var: 最大方差 (默认144)
    - correlation_strength: 均值与方差的相关强度 (0-1之间)
    
    返回:
    - 方差数组
    """
    # 归一化均值到[0,1]范围
    normalized_means = (means - np.min(means)) / (np.max(means) - np.min(means))
    
    # 创建与均值正相关的基础方差
    base_var = min_var + (max_var - min_var) * normalized_means
    
    # 添加随机性，相关性强度决定随机性的程度
    # 这里使用beta分布来确保方差保持在指定范围内
    random_component = np.random.beta(a=1, b=1/correlation_strength, size=len(means))
    random_component = min_var + (max_var - min_var) * random_component
    
    # 混合基础方差和随机成分
    variances = correlation_strength * base_var + (1 - correlation_strength) * random_component
    
    # 确保方差在指定范围内
    variances = np.clip(variances, min_var, max_var)
    
    return variances

def efficient_initialization(p, m, obs_num, mu_0, std2, R2, Sigma2, J):
    """
    高效初始化方法：以最少的次数覆盖所有模块
    
    参数:
    p: 总模块数
    m: 每次选择的模块数
    obs_num: 观测次数计数器
    mu_0, std2, R2, Sigma2, J: improvement函数所需参数
    
    返回:
    ids: 选择的模块索引
    """
    unobserved_mask = (obs_num == 0)
    unobserved_count = np.sum(unobserved_mask)
    
    if unobserved_count > 0:
        # 优先选择包含最多未观测模块的臂
        unobserved_arms = np.where(unobserved_mask)[0]
        
        if len(unobserved_arms) >= m:
            # 如果未观测模块足够，随机选择m个
            ids = np.random.choice(unobserved_arms, m, replace=False)
        else:
            # 如果未观测模块不足m个，先选所有未观测模块，再补充已观测模块
            ids = unobserved_arms.tolist()
            observed_arms = np.where(~unobserved_mask)[0]
            additional_needed = m - len(ids)
            
            if len(observed_arms) > 0 and additional_needed > 0:
                additional_ids = np.random.choice(observed_arms, additional_needed, replace=False)
                ids = np.concatenate([ids, additional_ids])
            else:
                # 如果所有模块都已观测或没有足够模块，使用improvement函数
                from util.methods import improvement
                _, ids = improvement(p, m, mu_0, std2, R2, Sigma2, J=J)
    else:
        # 所有模块都已观测过，使用improvement函数
        from util.methods import improvement
        _, ids = improvement(p, m, mu_0, std2, R2, Sigma2, J=J)
    
    return ids

def cholesky_update(U_old, A_new):
    """
    只更新Cholesky分解的最后一行
    
    参数:
    U_old: 旧的Cholesky分解（下三角矩阵，n×n）
    A_new: 新的协方差矩阵（n×n）
    返回:
    更新后的Cholesky分解
    """
    n = U_old.shape[0]
    U_new = U_old.copy()
    
    for i in range(n):
        off_diagonal_sum = 0.0
        for k in range(i):
            off_diagonal_sum += U_new[n-1, k] * U_new[i, k]
        
        U_new[n-1, i] = (A_new[n-1, i] - off_diagonal_sum) / U_new[i, i]
    
    diagonal_sum = 0.0
    for k in range(n-1):
        diagonal_sum += U_new[n-1, k] ** 2
    U_new[n-1, n-1] = np.sqrt(A_new[n-1, n-1] - diagonal_sum)
    
    return U_new

def calculate_course_contributions_generalized(mu, Sigma2, selected_courses, candidate, z_samples):
    """
    泛化版本：计算用候选课程替换最后一门课程后的平均贡献值
    
    参数:
    mu: 所有课程的均值向量
    Sigma2: 协方差矩阵
    selected_courses: 当前选择的课程列表
    candidate: 替换课程
    z_samples: 采样点
    """
    n_selected = len(selected_courses)
    last_course_idx = n_selected - 1  # 最后一门课的索引
    
    # 1. 生成初始样本并找到有效行
    current_mu = mu[selected_courses]
    current_Sigma = Sigma2[np.ix_(selected_courses, selected_courses)]
    U_current = np.linalg.cholesky(current_Sigma)
    
    samples = current_mu + (U_current @ z_samples.T).T
    
    # 找到有效行：最后一门课是该行最小值的行
    last_col_values = samples[:, last_course_idx:last_course_idx+1]  # 保持二维形状
    other_cols_values = np.delete(samples, last_course_idx, axis=1)  # 删除最后一列
    
    is_last_course_min = np.all(last_col_values < other_cols_values, axis=1)
    valid_indices = np.where(is_last_course_min)[0]
    
    if len(valid_indices) == 0:
        print("警告: 没有找到有效样本（最后一门课从未是最小值）")
        return {}
    
    # 记录有效行中其他课程的最小值（第二小的值）
    second_smallest_values = np.min(other_cols_values[valid_indices], axis=1)
    z_valid = z_samples[valid_indices, :]
    
    print(f"找到 {len(valid_indices)} 个有效样本（最后一门课是最小值）")
    
    # 2. 遍历候选课程计算贡献值
    improvement_scores = {}
    
    # 创建新的课程组合：用候选课程替换最后一门课
    new_courses = selected_courses.copy()
    new_courses[last_course_idx] = candidate
    
    # 计算新的协方差矩阵
    new_Sigma = Sigma2[np.ix_(new_courses, new_courses)]
    
    # 计算新的Cholesky分解的最后一行
    U_new_last_row = cholesky_update(U_current, new_Sigma)
    
    # 计算新课程的值：mu_candidate + U_new_last_row @ z^T
    new_course_values = mu[candidate] + np.dot(z_valid, U_new_last_row)
    
    # 计算贡献值：新课程值 - 第二小的值
    contributions = new_course_values - second_smallest_values
    
    # 取平均作为该候选课程的平均贡献值
    improvement_scores[candidate] = np.mean(contributions)
    
    return improvement_scores