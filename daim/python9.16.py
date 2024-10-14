import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel  # RBF kernel
import seaborn as sns
import pandas as pd
from itertools import product

# 示例目标函数 f(theta, x, xi)
def F(theta, x ,xi):
    
    return np.sum((theta - x)**2) + xi



def f(theta, x, num_samples):
    np.random.seed(1)
    xi_samples = np.random.normal(0, 1, num_samples)  # 随机样本
    #xi_samples = np.random.normal(0, 0, num_samples)
    costs = [F(theta, x, xi) for xi in xi_samples]
    return np.mean(costs)


def grad_f(theta, x, num_samples):
    np.random.seed(1)
    xi_samples = np.random.normal(0, 1, num_samples)  # 随机样本
    grads = [4 * (theta - x)**3 for xi in xi_samples]  # F 对 theta 的导数
    return np.mean(grads, axis=0)  # 对每个样本的梯度取平均
# SGD with Learning Rate Decay

def sgd_with_lr_decay( theta_0, x, eta_0, gamma, T):
    theta = theta_0
    
    theta_values = []  # 记录每次迭代的theta值
    for t in range(1, T + 1):
        
        #grad = grad_f(theta, x, num_samples)

        grad = 2 * (theta - x)
        # 学习率衰减
        #eta_t = eta_0 / t        #(1 + gamma * t)
        eta_t = eta_0 / (1 + gamma * t)
        # 更新参数
        #eta_t = eta_0 
        theta = theta - eta_t * grad
        
        
        theta_values.append(theta)
        
    
    return theta_values



# 随机梯度下降算法（SGD）的示例实现
# def sgd(theta_init, x, lr, num_iters):
#     theta = theta_init
#     for _ in range(num_iters):
#         grad = 2 * (theta - x)  # 目标函数 f 的梯度
#         theta = theta - lr * grad
#     return theta


# def sgd(theta_init, x, lr, num_iters,threshold=1e-6):
#     theta = theta_init
#     for t in range(1, num_iters + 1):  # t 从 1 开始，代表当前迭代的次数
#         grad = 2 * (theta - x)  # 目标函数 f 的梯度
#         # 使用梯度的L2范数判断是否小于阈值
#         # 使用 numpy 的 L2 范数计算
#         grad_norm = np.linalg.norm(grad)
#         if grad_norm < threshold:
#             print(f"梯度在迭代 {t} 时达到阈值 {threshold}，停止训练。")
#             break
#         theta = theta - (lr / t) * grad  # 动态调整学习率
#     return theta



def grid_sample(d, lowbound, upbound, point):
    # 根据给定的总点数和维度，计算每个维度的点数 n
    n = int(round(point ** (1.0 / d))) +1 # 每个维度上的点数
    
    # 在每个维度生成 n 个均匀间隔的点
    grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
    #print(len(grid_1d))
    # 在 d 维空间生成网格点
    grid_points = np.array(list(product(grid_1d, repeat=d)))
    
    # 如果生成的点数大于目标点数，则随机选择 point 个点
    if len(grid_points) > point:
        indices = np.random.choice(len(grid_points), size=point, replace=False)
        grid_points = grid_points[indices]
    #print(grid_points)
    return grid_points


# 离线阶段实现
def offline_stage(n, T, eta_0, gamma, covariate_dim):
    #np.random.seed(1)
    covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
    #covariates_points = np.random.uniform(low = lowbound , high=upbound, size=(n, covariate_dim))
    #print(n, covariates_points.shape)
    theta_estimates = []
    for i in range(n):
        x_i = covariates_points[i]
        
        
        np.random.seed(1)
        theta_init = np.random.randn(covariate_dim)  # 初始化θ

        
        theta_values = sgd_with_lr_decay( theta_init, x_i, eta_0, gamma, T)
        
        theta_bar = np.mean(theta_values, axis=0)
        theta_estimates.append(theta_bar)
    return covariates_points, np.array(theta_estimates)


# 在线阶段实现
def knn_online_stage(covariates_points, theta_estimates, x, k):
    # 计算新协变量点 x 与离线阶段得到的协变量点集之间的欧氏距离
    distances = euclidean_distances([x], covariates_points).flatten()
    
    # 找到 k 个最近邻的协变量点的索引
    nearest_neighbors_indices = np.argsort(distances)[:k]
    
    # 获取 k 个最近邻点的 θ 的估计值
    nearest_theta_estimates = theta_estimates[nearest_neighbors_indices]
    
    # 计算 θ 的平均值
    theta_hat = np.mean(nearest_theta_estimates, axis=0)
    
    return theta_hat






#KRR算法的在线阶段实现
def krr_online_stage(covariates_points, theta_estimates, x, lambda_param=1e-4):
    K_phi = rbf_kernel(covariates_points, covariates_points)  # 核矩阵
    k_phi_x = rbf_kernel([x], covariates_points).flatten()    # 新点与训练点之间的核
    theta_x = np.dot(k_phi_x, np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))) @ theta_estimates
    return theta_x

# 自定义实现的KRR算法
# def krr_online_stage(covariates_points, theta_estimates, x, lambda_param=1e-4, gamma=1.0):
#     # 计算核矩阵K_phi
#     n = covariates_points.shape[0]
#     K_phi = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             K_phi[i, j] = np.exp(-gamma * np.linalg.norm(covariates_points[i] - covariates_points[j])**2)
    
#     # 计算核向量k_phi_x
#     k_phi_x = np.array([np.exp(-gamma * np.linalg.norm(x - covariates_points[i])**2) for i in range(n)])
    
#     # 计算KRR的预测值
#     theta_x = np.dot(k_phi_x, np.linalg.inv(K_phi + lambda_param * np.eye(n)) @ theta_estimates)
    
#     return theta_x




def true_theta_function(x, num_samples):
    def objective(theta):
        return f(theta, x, num_samples)
    initial_guess = np.zeros(covariate_dim)
    result = minimize(objective, initial_guess, method='BFGS').x
    return result

# 将数据转换为适合Seaborn使用的格式
def prepare_data_for_plot(n_values, mse_by_k, mse_krr, k_values):
    data = []
    for k in k_values:
        for i, n in enumerate(n_values):
            data.append({"n": n, "MSE": mse_by_k[k][i], "Method": f"k-NN (k={k})"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_krr[i], "Method": "KRR"})
    return pd.DataFrame(data)




if __name__ == '__main__':
    lowbound = 0
    upbound = 4
    total_budget = 5000
    eta_0 = 0.1  # 初始学习率
    gamma = 0.01  # 学习率衰减率
    covariate_dim = 3
    num_samples = 1000  # Number of samples for true theta calculation
    # 生成测试数据集 (在线阶段需要的)
    num_test_points = 1000
    np.random.seed(1)
    test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    true_theta_values = np.array([true_theta_function(x, num_samples) for x in test_points])
    
    
    n_values = np.arange(10,201,10)
    #n_values = np.arange(100,401,100)
 
    # 不同的 k 值
    k_values = [1,  5, 10]  # 可以根据需要调整

    # 存储每个k值下的MSE
    mse_by_k = {k: [] for k in k_values}
    mse_krr = []
    # 遍历不同的n值
    for n in n_values:
        T = total_budget // n  # 计算对应的 T 值
        
        # 离线阶段
        covariates_points, theta_estimates = offline_stage(n, T, eta_0, gamma, covariate_dim)
        
        # 遍历不同的k值
        for k in k_values:
            # 在线阶段，计算预测值和真实值的 MSE
            predicted_theta_values = np.array([knn_online_stage(covariates_points, theta_estimates, x, k) for x in test_points])
            mse = mean_squared_error(np.array(true_theta_values), np.array(predicted_theta_values))
            
            mse_by_k[k].append(mse)
            print(f"k = {k}, n = {n}, T = {T}, MSE = {mse}")

        predicted_theta_values = np.array([krr_online_stage(covariates_points, theta_estimates, x) for x in test_points])
        mse = mean_squared_error(np.array(true_theta_values), np.array(predicted_theta_values))
        mse_krr.append(mse)
        print(f"KRR, n = {n}, T = {T}, MSE = {mse}")



    # 准备数据
    df = prepare_data_for_plot(n_values, mse_by_k, mse_krr, k_values)

    # 绘制MSE随n变化的曲线图
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='MSE', hue='Method', marker='o')

    plt.xlabel('Number of covariate points (n)')
    plt.ylabel('MSE')
    plt.title('MSE vs Number of covariate points (n) for different methods')
    plt.grid(True)
    # 使用 plt.text() 在图像中添加 total_budget 和 covariate_dim
    plt.text(0.95, 0.05, f'Total Budget: {total_budget}\nCovariate Dimension: {covariate_dim}', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    plt.show()