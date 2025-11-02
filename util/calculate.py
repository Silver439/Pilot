import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.linalg import cholesky, solve_triangular
import itertools
from typing import List, Tuple, Optional

class MultivariateNormalMinimum:
    """
    计算多元正态随机变量最小值的精确期望
    """
    
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        参数:
            mean: 均值向量 (m,)
            cov: 协方差矩阵 (m, m)
        """
        self.mean = mean
        self.cov = cov
        self.m = len(mean)
        
        # 验证输入
        assert cov.shape == (self.m, self.m), "协方差矩阵维度不匹配"
        assert np.allclose(cov, cov.T), "协方差矩阵必须对称"
        
    def _compute_W_k_parameters(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 W^(k) 的均值和协方差矩阵
        
        W_j^(k) = Y_k - Y_j for j ≠ k
        W_k^(k) = Y_k
        """
        # 构建变换矩阵 A
        A = np.zeros((self.m, self.m))
        
        # 对于 j ≠ k: W_j = Y_k - Y_j
        for j in range(self.m):
            if j != k:
                A[j, k] = 1   # Y_k
                A[j, j] = -1  # -Y_j
            else:
                # j = k: W_k = Y_k
                A[k, k] = 1
        
        # 计算 W^(k) 的均值
        mean_W = A @ self.mean
        
        # 计算 W^(k) 的协方差
        cov_W = A @ self.cov @ A.T
        
        return mean_W, cov_W
    
    def _compute_conditional_parameters(self, mean_W: np.ndarray, cov_W: np.ndarray, 
                                      i: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算给定 W_i = c_i 时其他变量的条件分布参数
        """
        m = len(mean_W)
        indices = list(range(m))
        indices.remove(i)
        
        # 提取子矩阵
        Sigma_ii = cov_W[i, i]
        Sigma_i_rest = cov_W[i, indices]
        Sigma_rest_rest = cov_W[np.ix_(indices, indices)]
        
        # 计算条件协方差
        cond_cov = Sigma_rest_rest - np.outer(Sigma_i_rest, Sigma_i_rest) / Sigma_ii
        
        return cond_cov
    
    def _compute_g_vector(self, c: np.ndarray, mean_W: np.ndarray, 
                         cov_W: np.ndarray, i: int) -> np.ndarray:
        """
        计算 g_{·i}^{(k)} 向量
        """
        m = len(mean_W)
        indices = list(range(m))
        indices.remove(i)
        
        Sigma_ii = cov_W[i, i]
        g = np.zeros(m-1)
        
        for idx, j in enumerate(indices):
            g[idx] = (c[j] - mean_W[j]) - (c[i] - mean_W[i]) * cov_W[i, j] / Sigma_ii
        
        return g
    
    def compute_R(self, max_dim: int = 21, method: str = 'scipy', 
                 n_samples: int = 10000, seed: int = 1) -> float:
        """
        计算 R(S) = E[min(Y_1, ..., Y_m)]
        
        参数:
            max_dim: 使用精确方法的最大维度
            method: 高维CDF计算方法 ('monte_carlo' 或 'scipy')
            n_samples: Monte Carlo采样数
        """
        if self.m <= max_dim:
            return self._compute_R_exact(method, n_samples, seed)
        else:
            print(f"维度 {self.m} 太高，使用Monte Carlo近似")
            return self._compute_R_monte_carlo(n_samples)
    
    def _compute_R_exact(self, method: str = 'scipy', n_samples: int = 10000, seed: int = 1) -> float:
        """
        使用命题中的精确公式计算 R(S)
        """
        R_total = 0.0
        
        for k in range(self.m):
            # 1. 计算 W^(k) 的参数
            mean_W, cov_W = self._compute_W_k_parameters(k)
            
            # 2. 定义 c^(k) 向量
            c = np.full(self.m, 0.0)  # 对于 j ≠ k, c_j = 0
            c[k] = 1e10  # 对于 k, c_k = +∞ (用大数近似)
            
            # 3. 计算 p_k = P(W^(k) ≤ c^(k))
            if method == 'scipy':
                # 使用scipy的多维正态CDF (仅适用于低维)
                try:
                    p_k = multivariate_normal(mean=mean_W, cov=cov_W, seed=seed).cdf(c)
                except:
                    p_k = self._monte_carlo_cdf(mean_W, cov_W, c, n_samples//10)
            else:
                p_k = self._monte_carlo_cdf(mean_W, cov_W, c, n_samples//10)
            
            # 4. 计算求和项
            sum_term = 0.0
            for i in range(self.m):
                if cov_W[i, i] <= 1e-10:  # 避免数值问题
                    continue
                    
                # 计算边际密度
                marginal_pdf = norm.pdf(c[i], loc=mean_W[i], scale=np.sqrt(cov_W[i, i]))
                
                # 计算条件分布参数
                cond_cov = self._compute_conditional_parameters(mean_W, cov_W, i)
                g_vector = self._compute_g_vector(c, mean_W, cov_W, i)
                
                # 计算条件概率
                if len(g_vector) == 0:  # 一维情况
                    cond_prob = 1.0
                elif method == 'scipy':
                    try:
                        cond_prob = multivariate_normal(mean=np.zeros(len(g_vector)), 
                                                      cov=cond_cov).cdf(g_vector)
                    except:
                        cond_prob = self._monte_carlo_cdf(np.zeros(len(g_vector)), 
                                                         cond_cov, g_vector, n_samples//20)
                else:
                    cond_prob = self._monte_carlo_cdf(np.zeros(len(g_vector)), 
                                                     cond_cov, g_vector, n_samples//20)
                
                sum_term += cov_W[i, k] * marginal_pdf * cond_prob
            
            # 5. 累加到总结果
            R_total += self.mean[k] * p_k - sum_term
        
        return R_total
    
    def _monte_carlo_cdf(self, mean: np.ndarray, cov: np.ndarray, 
                        c: np.ndarray, n_samples: int) -> float:
        """
        使用Monte Carlo计算多元正态CDF P(X ≤ c)
        """
        try:
            # Cholesky分解生成样本
            L = cholesky(cov, lower=True)
            samples = mean + np.random.randn(n_samples, len(mean)) @ L.T
            prob = np.mean(np.all(samples <= c, axis=1))
            return max(prob, 1e-10)  # 避免数值下溢
        except:
            # 如果Cholesky失败，使用简单采样
            samples = np.random.multivariate_normal(mean, cov, n_samples)
            prob = np.mean(np.all(samples <= c, axis=1))
            return max(prob, 1e-10)
    
    def _compute_R_monte_carlo(self, n_samples: int = 10000) -> float:
        """
        使用直接Monte Carlo计算 E[min(Y)]
        """
        try:
            samples = np.random.multivariate_normal(self.mean, self.cov, n_samples)
            min_values = np.min(samples, axis=1)
            return np.mean(min_values)
        except:
            # 如果协方差矩阵不是正定的，使用特征值分解
            eigvals, eigvecs = np.linalg.eigh(self.cov)
            eigvals = np.maximum(eigvals, 1e-10)  # 确保正定
            cov_corrected = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            samples = np.random.multivariate_normal(self.mean, cov_corrected, n_samples)
            min_values = np.min(samples, axis=1)
            return np.mean(min_values)
