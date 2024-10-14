import numpy as np
from sklearn.neighbors import NearestNeighbors

# 假设目标函数 F(theta, x, xi) 依赖于随机变量 xi
def F(theta, x, xi):
    # 这里是一个简单的目标函数
    return np.sum((theta - x)**2) + xi  # 二次损失函数加上噪声项 xi

# 目标函数的计算：在多个样本上取期望
def f(theta, x, num_samples):
    np.random.seed(1)
    xi_samples = np.random.normal(0, 1, num_samples)  # 随机样本
    costs = [F(theta, x, xi) for xi in xi_samples]
    return np.mean(costs)

# 目标函数的梯度：这里假设梯度是通过 F(theta, x, xi) 的导数计算
def grad_f(theta, x, num_samples):
    np.random.seed(1)
    xi_samples = np.random.normal(0, 1, num_samples)  # 随机样本
    grads = [2 * (theta - x) for xi in xi_samples]  # F 对 theta 的导数
    return np.mean(grads, axis=0)  # 对每个样本的梯度取平均

# SGD with Fixed Learning Rate
def sgd_fixed_lr(theta_0, x, eta_0 , T, num_samples):
    theta = theta_0
    theta_values = []
    for t in range(1, T + 1):
        # 计算当前梯度
        g_t = grad_f(theta, x, num_samples)
        # 使用固定学习率更新参数
        theta = theta - eta_0 * g_t
        theta_values.append(theta)
    return theta_values

# Offline stage
def offline_stage_sgd(n, T, eta_0, num_samples, x_range):
    # 随机生成 n 个协变量点
    covariates_points = np.random.uniform(x_range[0], x_range[1], size=(n, 4))  # 4维协变量空间
    average_theta = []  # 存储每个协变量点的平均参数

    # 对每个协变量点，运行 SGD
    for i, x_i in enumerate(covariates_points):
        print(f"Running SGD for covariate point {i+1}/{n}, x = {x_i}")
        theta_0 = np.zeros(4)  # 初始参数为0的4维向量
        theta_values = sgd_fixed_lr(theta_0, x_i, eta_0, T, num_samples)
        
        # 计算平均参数 \bar{\theta}_T(x_i)
        avg_theta_i = np.mean(theta_values, axis=0)
        average_theta.append(avg_theta_i)
    
    return covariates_points, np.array(average_theta)

# Online stage (KNN algorithm)
def online_stage_knn(average_theta, covariates_points, x, k):
    # 使用KNN算法找到x的最近邻
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(covariates_points)
    distances, indices = nbrs.kneighbors([x])
    
    # 计算 \hat{\theta}(x)
    nearest_neighbors_theta = average_theta[indices[0]]
    estimated_theta = np.mean(nearest_neighbors_theta, axis=0)
    
    return estimated_theta

# 示例用法
if __name__ == "__main__":
    n = 5  # 协变量点的数量
    T = 50  # 每个协变量点上的SGD迭代次数
    eta_0 = 0.01  # 固定学习率
    num_samples = 100  # 样本数目
    k = 3  # 最近邻数
    x_range = (0, 10)  # 协变量点的范围

    # Offline stage
    covariates_points, average_theta = offline_stage_sgd(n, T, eta_0, num_samples, x_range)
    print("\nOffline stage completed.\n")
    
    # Online stage
    x_new = np.array([5.0, 5.0, 5.0, 5.0])  # 新的协变量点
    estimated_theta = online_stage_knn(average_theta, covariates_points, x_new, k)
    print(f"Estimated theta for new point {x_new}: {estimated_theta}")
